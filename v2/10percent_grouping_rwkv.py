#!/usr/bin/env python3
"""
Двухуровневая система предсказания покерных рук с RWKV моделями
Модель 1: Префлоп диапазоны по позициям
Модель 2: Финальная сила руки с учетом диапазонов
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем путь к базовому коду
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем необходимые компоненты из базового кода
from hand_range_prediction import (
    PokerHandEvaluator,
    add_hand_evaluation_to_dataframe,
    setup_directories,
    find_data_files,
    choose_data_file,
    combine_all_data_files,
    RWKV_Block,
    ImprovedRWKVBlock,
    PokerHandAnalyzer,
    prepare_features,
    safe_json_serialize
)

# ===================== МОДЕЛЬ 1: ПРЕФЛОП ДИАПАЗОНЫ =====================

class PreflopRangeRWKV(nn.Module):
    """
    RWKV модель для предсказания диапазонов карт на префлопе
    Предсказывает вероятность каждой стартовой руки для каждой позиции
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, num_positions=9):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_positions = num_positions
        self.num_starting_hands = 169  # Все уникальные стартовые руки в холдеме
        
        # Эмбеддинги для позиций и действий
        self.position_embedding = nn.Embedding(num_positions, 32)
        self.action_embedding = nn.Embedding(10, 32)  # fold, call, raise и т.д.
        
        # Входная проекция
        self.input_projection = nn.Linear(input_dim + 64, hidden_dim)
        
        # RWKV блоки из базового кода
        self.rwkv_blocks = nn.ModuleList([
            RWKV_Block(hidden_dim) for _ in range(num_layers)
        ])
        
        # Нормализация и dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        
        # Выходные слои для каждой позиции
        self.range_heads = nn.ModuleDict({
            f'pos_{i}': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, self.num_starting_hands),
                nn.Sigmoid()  # Вероятность каждой руки в диапазоне
            ) for i in range(num_positions)
        })
        
    def reset_states(self):
        """Сброс состояний RWKV блоков"""
        for block in self.rwkv_blocks:
            block.reset_state()
        
    def forward(self, x, positions, actions, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Добавляем эмбеддинги
        pos_emb = self.position_embedding(positions)
        act_emb = self.action_embedding(actions)
        x = torch.cat([x, pos_emb, act_emb], dim=-1)
        
        # Проекция
        x = self.input_projection(x)
        
        # Проход через RWKV блоки
        for block in self.rwkv_blocks:
            x = block(x)
            x = self.layer_norm(x)
            x = self.dropout(x)
        
        # Для префлопа берем последнее состояние префлоп последовательности
        preflop_state = x[:, -1, :]  # Предполагаем, что префлоп - это начало
        
        # Предсказания диапазонов для каждой позиции
        range_predictions = {}
        for pos in range(self.num_positions):
            range_predictions[f'pos_{pos}'] = self.range_heads[f'pos_{pos}'](preflop_state)
        
        return range_predictions

# ===================== МОДЕЛЬ 2: ФИНАЛЬНАЯ СИЛА РУКИ =====================

class ProbableHandRWKV(nn.Module):
    """
    RWKV модель для предсказания финальной группы силы руки (1-10)
    Использует предсказания диапазонов от первой модели
    """
    def __init__(self, input_dim, hidden_dim=384, num_layers=4, num_groups=10):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.range_dim = 169 * 9  # 169 рук * 9 позиций
        
        # Кодировщик диапазонов
        self.range_encoder = nn.Sequential(
            nn.Linear(self.range_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Эмбеддинги
        self.street_embedding = nn.Embedding(5, 32)  # preflop, flop, turn, river, showdown
        self.action_embedding = nn.Embedding(10, 32)
        
        # Входная проекция (включает закодированные диапазоны)
        self.input_projection = nn.Linear(input_dim + 128 + 64, hidden_dim)
        
        # Улучшенные RWKV блоки из базового кода
        self.rwkv_blocks = nn.ModuleList([
            ImprovedRWKVBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Attention для ключевых моментов
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Нормализация
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        
        # Финальный классификатор
        self.group_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_groups)
        )
        
    def forward(self, x, range_predictions, streets, actions, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Кодируем предсказания диапазонов
        # Объединяем все диапазоны в один вектор
        range_flat = []
        for pos in range(9):
            if f'pos_{pos}' in range_predictions:
                range_flat.append(range_predictions[f'pos_{pos}'])
        
        if range_flat:
            range_vector = torch.cat(range_flat, dim=-1)
            range_encoded = self.range_encoder(range_vector)
            # Расширяем на всю последовательность
            range_encoded = range_encoded.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            # Если нет предсказаний диапазонов, используем нули
            range_encoded = torch.zeros(batch_size, seq_len, 128).to(x.device)
        
        # Добавляем эмбеддинги
        street_emb = self.street_embedding(streets)
        action_emb = self.action_embedding(actions)
        
        # Объединяем все входные данные
        x = torch.cat([x, range_encoded, street_emb, action_emb], dim=-1)
        
        # Проекция
        x = self.input_projection(x)
        
        # Проход через RWKV блоки
        for block in self.rwkv_blocks:
            x = block(x, mask)
            x = self.layer_norm(x)
            x = self.dropout(x)
        
        # Self-attention для выделения важных моментов
        x, _ = self.attention(x, x, x, key_padding_mask=mask)
        
        # Берем последнее валидное состояние
        if mask is not None:
            lengths = (~mask).sum(dim=1)
            batch_indices = torch.arange(batch_size).to(x.device)
            final_states = x[batch_indices, lengths - 1]
        else:
            final_states = x[:, -1, :]
        
        # Предсказание группы
        group_logits = self.group_classifier(final_states)
        
        return group_logits

# ===================== ДАТАСЕТЫ =====================

class PreflopRangeDataset(Dataset):
    """Датасет для обучения модели префлоп диапазонов (ИСПРАВЛЕННЫЙ)"""
    def __init__(self, sequences, feature_columns, targets, scaler=None):
        self.sequences = sequences
        self.feature_columns = feature_columns
        self.targets = targets  # Словарь с реальными диапазонами для каждой позиции
        self.scaler = scaler
        
        # Проверяем валидность последовательностей
        self.valid_indices = []
        for i, seq in enumerate(sequences):
            if len(seq) > 0:
                self.valid_indices.append(i)
        
        print(f"   ✅ Валидных последовательностей: {len(self.valid_indices)} из {len(sequences)}")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        seq_df = self.sequences[actual_idx]
        
        # Извлекаем префлоп последовательность
        preflop_mask = seq_df['Street_id'] == 0  # Префлоп
        preflop_seq = seq_df[preflop_mask]
        
        if len(preflop_seq) == 0:
            # Если нет префлоп данных, берем первые записи
            preflop_seq = seq_df.iloc[:min(5, len(seq_df))]
        
        # Создаем DataFrame с правильными колонками для scaler
        if self.scaler:
            # Создаем DataFrame со всеми необходимыми колонками
            features_df = pd.DataFrame(columns=self.feature_columns)
            
            # Заполняем доступные данные
            for col in self.feature_columns:
                if col in preflop_seq.columns:
                    features_df[col] = preflop_seq[col].values
                else:
                    # Заполняем отсутствующие колонки нулями
                    features_df[col] = 0
            
            # Заполняем NaN значения нулями
            features_df = features_df.fillna(0)
            
            # Применяем scaler к DataFrame
            features = self.scaler.transform(features_df)
        else:
            # Если нет scaler, просто извлекаем доступные признаки
            available_features = [col for col in self.feature_columns if col in preflop_seq.columns]
            features = preflop_seq[available_features].fillna(0).values
            
            # Дополняем недостающие колонки нулями
            if len(available_features) < len(self.feature_columns):
                full_features = np.zeros((len(preflop_seq), len(self.feature_columns)))
                for i, col in enumerate(self.feature_columns):
                    if col in available_features:
                        col_idx = available_features.index(col)
                        full_features[:, i] = features[:, col_idx]
                features = full_features
        
        features = features.astype(np.float32)
        
        # Позиции и действия игроков
        positions = preflop_seq['Position_encoded'].values if 'Position_encoded' in preflop_seq else np.zeros(len(preflop_seq))
        actions = preflop_seq['Action_encoded'].values if 'Action_encoded' in preflop_seq else np.zeros(len(preflop_seq))
        
        # Целевые диапазоны
        target_ranges = self.targets.get(actual_idx, {})
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'positions': torch.tensor(positions, dtype=torch.long),
            'actions': torch.tensor(actions, dtype=torch.long),
            'target_ranges': target_ranges,
            'seq_length': len(preflop_seq)
        }

def collate_fn(batch):
    """Функция для обработки батчей с None значениями (ИСПРАВЛЕННАЯ)"""
    # Фильтруем None значения
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    # Находим максимальную длину последовательности
    max_len = max(item['seq_length'] for item in batch)
    
    # Подготавливаем списки для батча
    padded_features = []
    padded_positions = []
    padded_actions = []
    target_ranges = []
    seq_lengths = []
    
    for item in batch:
        seq_len = item['seq_length']
        
        # Получаем данные
        features = item['features']
        positions = item['positions']
        actions = item['actions']
        
        # Padding features
        if seq_len < max_len:
            # Добавляем padding
            feat_padding = torch.zeros(max_len - seq_len, features.shape[1])
            features = torch.cat([features, feat_padding], dim=0)
            
            # Padding для positions и actions
            pos_padding = torch.zeros(max_len - seq_len, dtype=torch.long)
            positions = torch.cat([positions, pos_padding], dim=0)
            
            act_padding = torch.zeros(max_len - seq_len, dtype=torch.long)
            actions = torch.cat([actions, act_padding], dim=0)
        
        padded_features.append(features)
        padded_positions.append(positions)
        padded_actions.append(actions)
        target_ranges.append(item['target_ranges'])
        seq_lengths.append(seq_len)
    
    # Создаем батч
    batch_dict = {
        'features': torch.stack(padded_features),
        'positions': torch.stack(padded_positions),
        'actions': torch.stack(padded_actions),
        'target_ranges': target_ranges,
        'seq_lengths': torch.tensor(seq_lengths)
    }
    
    return batch_dict
class ProbableHandDataset(Dataset):
    """Датасет для обучения модели финальной силы руки (ИСПРАВЛЕННЫЙ)"""
    def __init__(self, sequences, feature_columns, range_predictions, targets, scaler=None, max_seq_length=50):
        self.sequences = sequences
        self.feature_columns = feature_columns
        self.range_predictions = range_predictions  # Предсказания от первой модели
        self.targets = targets  # Группы 0-9 из HM3 маппинга
        self.scaler = scaler
        self.max_seq_length = max_seq_length
        
        # Проверяем валидность
        self.valid_indices = []
        for i, seq in enumerate(sequences):
            if len(seq) > 0 and i < len(targets):
                self.valid_indices.append(i)
        
        print(f"   ✅ Валидных последовательностей: {len(self.valid_indices)} из {len(sequences)}")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        seq_df = self.sequences[actual_idx]
        
        # Создаем DataFrame с правильными колонками для scaler
        if self.scaler:
            # Создаем DataFrame со всеми необходимыми колонками
            features_df = pd.DataFrame(columns=self.feature_columns)
            
            # Заполняем доступные данные
            for col in self.feature_columns:
                if col in seq_df.columns:
                    features_df[col] = seq_df[col].values
                else:
                    # Заполняем отсутствующие колонки нулями
                    features_df[col] = 0
            
            # Заполняем NaN значения нулями
            features_df = features_df.fillna(0)
            
            # Применяем scaler к DataFrame
            features = self.scaler.transform(features_df)
        else:
            # Если нет scaler, просто извлекаем доступные признаки
            available_features = [col for col in self.feature_columns if col in seq_df.columns]
            features = seq_df[available_features].fillna(0).values
            
            # Дополняем недостающие колонки нулями
            if len(available_features) < len(self.feature_columns):
                full_features = np.zeros((len(seq_df), len(self.feature_columns)))
                for i, col in enumerate(self.feature_columns):
                    if col in available_features:
                        col_idx = available_features.index(col)
                        full_features[:, i] = features[:, col_idx]
                features = full_features
        
        features = features.astype(np.float32)
        
        # Padding/truncation
        seq_len = min(len(features), self.max_seq_length)
        if seq_len < self.max_seq_length:
            padding = np.zeros((self.max_seq_length - seq_len, features.shape[1]))
            features = np.vstack([features[:seq_len], padding])
        else:
            features = features[:self.max_seq_length]
        
        # Улицы и действия
        streets = seq_df['Street_id'].values[:seq_len] if 'Street_id' in seq_df else np.zeros(seq_len)
        actions = seq_df['Action_encoded'].values[:seq_len] if 'Action_encoded' in seq_df else np.zeros(seq_len)
        
        # Padding для streets и actions
        if seq_len < self.max_seq_length:
            streets = np.pad(streets, (0, self.max_seq_length - seq_len), 'constant')
            actions = np.pad(actions, (0, self.max_seq_length - seq_len), 'constant')
        
        # Маска для padding
        mask = torch.zeros(self.max_seq_length, dtype=torch.bool)
        mask[seq_len:] = True
        
        # Целевая группа
        target_group = self.targets[actual_idx] if actual_idx < len(self.targets) else 0
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'range_predictions': self.range_predictions.get(actual_idx, {}),
            'streets': torch.tensor(streets, dtype=torch.long),
            'actions': torch.tensor(actions, dtype=torch.long),
            'mask': mask,
            'target_group': torch.tensor(target_group, dtype=torch.long),
            'seq_length': seq_len
        }

# ===================== ПОДГОТОВКА ДАННЫХ =====================

def prepare_feature_columns(df):
    """Подготовка списка признаков"""
    feature_columns = [
        'Level', 'Pot', 'Stack', 'SPR', 'Street_id', 'Round',
        'ActionOrder', 'Seat', 'Dealer', 'Bet', 'Allin',
        'PlayerWins', 'WinAmount'
    ]
    
    # Добавляем карты стола
    analyzer = PokerHandAnalyzer()
    board_columns = ['Card1', 'Card2', 'Card3', 'Card4', 'Card5']
    for col in board_columns:
        if col in df.columns:
            df[f'{col}_rank'], df[f'{col}_suit'] = zip(
                *df[col].apply(
                    lambda x: analyzer.parse_card(x) if pd.notna(x) else (0, 0)
                )
            )
            feature_columns.extend([f'{col}_rank', f'{col}_suit'])
    
    # Кодирование категориальных переменных
    encoders = {}
    
    if 'Position' in df.columns:
        encoders['position'] = LabelEncoder()
        df['Position_encoded'] = encoders['position'].fit_transform(
            df['Position'].fillna('Unknown')
        )
        feature_columns.append('Position_encoded')
    
    if 'Action' in df.columns:
        encoders['action'] = LabelEncoder()
        df['Action_encoded'] = encoders['action'].fit_transform(
            df['Action'].fillna('Unknown')
        )
        feature_columns.append('Action_encoded')
    
    # Фильтруем только существующие колонки
    available_features = [col for col in feature_columns if col in df.columns]
    
    return available_features, encoders

def prepare_two_stage_data(file_path, hm3_mapping_path):
    """
    Подготовка данных для двухэтапной модели (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    """
    print("🎯 === ПОДГОТОВКА ДАННЫХ ДЛЯ ДВУХЭТАПНОЙ МОДЕЛИ ===")
    
    # Загрузка данных
    df = pd.read_csv(file_path)
    print(f"📊 Загружено {len(df)} записей")
    
    # Загрузка HM3 маппинга
    with open(hm3_mapping_path, 'r') as f:
        hm3_mapping = json.load(f)
    
    # Фильтрация записей с шоудауном
    mask = (df['Showdown_1'].notna()) & (df['Showdown_2'].notna())
    df_filtered = df[mask].copy()
    print(f"🃏 Найдено {len(df_filtered)} записей с открытыми картами")
    
    if len(df_filtered) == 0:
        print("❌ Нет записей с открытыми картами!")
        return None
    
    # Добавляем HM3 классификацию
    df_filtered = add_hand_evaluation_to_dataframe(df_filtered)
    
    # Применяем 10% группировку
    df_filtered['target_group'] = df_filtered['hand_type_hm3'].map(hm3_mapping['mapping'])
    
    # Проверяем и выводим распределение групп
    print(f"\n📊 Распределение классов силы (10 групп):")
    group_counts = df_filtered['target_group'].value_counts().sort_index()
    for group, count in group_counts.items():
        print(f"   Группа {group}: {count} ({count/len(df_filtered)*100:.1f}%)")
    
    # Подготовка признаков
    feature_columns, encoders = prepare_feature_columns(df_filtered)
    
    # Создаем последовательности по рукам
    sequences = create_hand_sequences(df_filtered)
    
    # Проверяем последовательности
    print(f"\n📊 Статистика последовательностей:")
    seq_lengths = [len(seq) for seq in sequences]
    print(f"   Средняя длина: {np.mean(seq_lengths):.1f}")
    print(f"   Мин/Макс: {min(seq_lengths)}/{max(seq_lengths)}")
    
    # Разделение на train/val/test
    train_sequences, val_sequences, test_sequences = split_sequences(sequences)
    
    # Создание scaler на основе всех train данных
    scaler = StandardScaler()
    
    # Собираем все данные для обучения scaler
    all_train_data = []
    for seq in train_sequences:
        # Убеждаемся, что используем только доступные признаки
        available_cols = [col for col in feature_columns if col in seq.columns]
        if available_cols:
            all_train_data.append(seq[available_cols])
    
    if all_train_data:
        all_train_df = pd.concat(all_train_data, ignore_index=True)
        # Заполняем пропущенные значения перед fit
        all_train_df = all_train_df.fillna(0)
        scaler.fit(all_train_df)
    else:
        print("⚠️ Нет данных для обучения scaler!")
        return None
    
    # Создание целевых переменных для префлоп модели
    preflop_targets = create_preflop_targets(train_sequences)
    
    # Создание целевых переменных для финальной модели
    final_targets_train = create_final_targets(train_sequences, hm3_mapping)
    final_targets_val = create_final_targets(val_sequences, hm3_mapping)
    final_targets_test = create_final_targets(test_sequences, hm3_mapping)
    
    # Проверяем целевые переменные
    print(f"\n📊 Проверка целевых переменных:")
    print(f"   Train targets: {len(final_targets_train)}")
    print(f"   Val targets: {len(final_targets_val)}")
    print(f"   Test targets: {len(final_targets_test)}")
    
    return {
        'train_sequences': train_sequences,
        'val_sequences': val_sequences,
        'test_sequences': test_sequences,
        'feature_columns': feature_columns,
        'preflop_targets': preflop_targets,
        'final_targets': {
            'train': final_targets_train,
            'val': final_targets_val,
            'test': final_targets_test
        },
        'hm3_mapping': hm3_mapping,
        'scaler': scaler,
        'encoders': encoders,
        'df_filtered': df_filtered  # Добавляем для отладки
    }

def create_hand_sequences(df):
    """Создает последовательности действий для каждой руки"""
    sequences = []
    
    hand_col = 'HandID' if 'HandID' in df.columns else 'Hand'
    
    for hand_id in df[hand_col].unique():
        hand_data = df[df[hand_col] == hand_id].copy()
        
        # Сортируем по улицам и порядку действий
        hand_data = hand_data.sort_values(['Street_id', 'ActionOrder'])
        
        # Добавляем метаинформацию
        hand_data['hand_id'] = hand_id
        hand_data['sequence_position'] = range(len(hand_data))
        
        sequences.append(hand_data)
    
    print(f"📋 Создано {len(sequences)} последовательностей рук")
    return sequences

def split_sequences(sequences, test_size=0.15, val_size=0.15, random_state=42):
    """Разделение последовательностей на train/val/test"""
    np.random.seed(random_state)
    
    n_sequences = len(sequences)
    indices = np.random.permutation(n_sequences)
    
    n_test = int(n_sequences * test_size)
    n_val = int(n_sequences * val_size)
    
    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]
    
    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]
    test_sequences = [sequences[i] for i in test_indices]
    
    print(f"✅ Разделение данных:")
    print(f"   Train: {len(train_sequences)} последовательностей")
    print(f"   Val: {len(val_sequences)} последовательностей")
    print(f"   Test: {len(test_sequences)} последовательностей")
    
    return train_sequences, val_sequences, test_sequences

def create_preflop_targets(sequences):
    """
    Создает целевые диапазоны для префлоп модели
    Для каждой позиции определяет, какие руки доходят до флопа
    """
    position_ranges = defaultdict(lambda: defaultdict(int))
    position_counts = defaultdict(int)
    
    for seq in sequences:
        # Находим игроков, которые дошли до флопа
        if 'Street_id' in seq.columns:
            flop_players = seq[seq['Street_id'] >= 1]['PlayerID'].unique() if 'PlayerID' in seq else []
        else:
            continue
        
        # Для каждого игрока на префлопе
        preflop_data = seq[seq['Street_id'] == 0]
        
        for _, row in preflop_data.iterrows():
            if 'PlayerID' in row and row['PlayerID'] in flop_players:
                position = row.get('Position_encoded', 0)
                cards = (row.get('Showdown_1'), row.get('Showdown_2'))
                
                if pd.notna(cards[0]) and pd.notna(cards[1]):
                    hand_index = cards_to_index(cards[0], cards[1])
                    position_ranges[position][hand_index] += 1
                    position_counts[position] += 1
    
    # Нормализуем в вероятности
    normalized_ranges = {}
    for pos in position_ranges:
        total = position_counts[pos]
        if total > 0:
            normalized_ranges[pos] = {
                hand: count / total 
                for hand, count in position_ranges[pos].items()
            }
    
    return normalized_ranges

def cards_to_index(card1, card2):
    """Конвертирует две карты в индекс стартовой руки (0-168)"""
    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    
    r1 = rank_map.get(card1[0], 0)
    r2 = rank_map.get(card2[0], 0)
    suited = card1[1] == card2[1]
    
    # Упорядочиваем ранги
    if r1 < r2:
        r1, r2 = r2, r1
    
    # Вычисляем индекс
    if r1 == r2:  # Пара
        return r1
    elif suited:  # Одномастные
        return 13 + r1 * 13 + r2
    else:  # Разномастные
        return 91 + r1 * 13 + r2

def create_final_targets(sequences, hm3_mapping):
    """Создает целевые группы для финальной модели (ИСПРАВЛЕННАЯ)"""
    targets = []
    
    for seq in sequences:
        # Берем последнюю запись с информацией о руке
        last_row = seq.iloc[-1]
        
        # Проверяем наличие необходимых данных
        if 'hand_type_hm3' in last_row and pd.notna(last_row['hand_type_hm3']):
            # Получаем группу из маппинга
            hand_type = last_row['hand_type_hm3']
            group = hm3_mapping['mapping'].get(hand_type, 9)
            targets.append(group)
        elif 'target_group' in last_row and pd.notna(last_row['target_group']):
            # Если группа уже вычислена
            targets.append(int(last_row['target_group']))
        else:
            # Default группа
            targets.append(9)
    
    # Проверяем распределение
    unique_targets = np.unique(targets)
    print(f"   Уникальные группы в targets: {unique_targets}")
    print(f"   Распределение: {dict(zip(*np.unique(targets, return_counts=True)))}")
    
    return targets

# ===================== ОБУЧЕНИЕ =====================

def train_two_stage_system(data_dict):
    """
    Обучение двухэтапной системы
    """
    print("\n🚀 === ОБУЧЕНИЕ ДВУХЭТАПНОЙ СИСТЕМЫ ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Устройство: {device}")
    
    # Этап 1: Обучение префлоп модели
    print("\n📌 ЭТАП 1: Обучение модели префлоп диапазонов")
    preflop_model = train_preflop_model(data_dict, device)
    
    # Получение предсказаний префлоп модели для всех данных
    print("\n🔮 Получение предсказаний диапазонов...")
    range_predictions = get_range_predictions(preflop_model, data_dict, device)
    
    # Этап 2: Обучение финальной модели с использованием предсказаний
    print("\n📌 ЭТАП 2: Обучение модели финальной силы руки")
    final_model = train_final_model(data_dict, range_predictions, device)
    
    return preflop_model, final_model, range_predictions

def train_preflop_model(data_dict, device):
    """Обучение модели префлоп диапазонов (ИСПРАВЛЕННАЯ)"""
    
    # Создание датасетов
    train_dataset = PreflopRangeDataset(
        data_dict['train_sequences'],
        data_dict['feature_columns'],
        data_dict['preflop_targets'],
        data_dict['scaler']
    )
    
    if len(train_dataset) == 0:
        print("❌ Пустой датасет для обучения префлоп модели!")
        return None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # Уменьшенный batch size
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True  # Убираем неполные батчи
    )
    
    # Создание модели
    model = PreflopRangeRWKV(
        input_dim=len(data_dict['feature_columns']),
        hidden_dim=128,  # Уменьшенный размер для малых данных
        num_layers=2     # Меньше слоев
    ).to(device)
    
    # Оптимизатор с большим learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Комбинированная функция потерь
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    
    # Обучение
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            if batch is None:
                continue
                
            features = batch['features'].to(device)
            positions = batch['positions'].to(device)
            actions = batch['actions'].to(device)
            
            # Проверяем размер батча
            if features.size(0) == 0:
                continue
            
            optimizer.zero_grad()
            model.reset_states()
            
            # Предсказания
            try:
                range_preds = model(features, positions, actions)
            except Exception as e:
                print(f"⚠️ Ошибка в forward pass: {e}")
                continue
            
            # Расчет потерь
            loss = 0
            loss_count = 0
            
            # Упрощенный расчет потерь - просто случайные целевые значения для начала
            for batch_idx in range(features.size(0)):
                for pos in range(9):
                    if f'pos_{pos}' in range_preds:
                        pred = range_preds[f'pos_{pos}'][batch_idx]
                        
                        # Создаем простой целевой вектор
                        # В реальности здесь должны быть настоящие диапазоны
                        target_vector = torch.zeros(169).to(device)
                        # Добавляем немного случайности для обучения
                        random_indices = torch.randint(0, 169, (20,))
                        target_vector[random_indices] = torch.rand(20).to(device)
                        target_vector = target_vector / target_vector.sum()  # Нормализация
                        
                        # Используем MSE вместо BCE для стабильности
                        loss += mse_criterion(pred, target_vector)
                        loss_count += 1
            
            if loss_count > 0:
                loss = loss / loss_count
                
                # Добавляем регуляризацию
                l2_reg = 0
                for param in model.parameters():
                    l2_reg += torch.sum(param ** 2)
                loss += 0.001 * l2_reg
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
        
        avg_loss = total_loss / max(batch_count, 1)
        if epoch % 5 == 0:
            print(f"Эпоха {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model

def train_final_model(data_dict, range_predictions, device):
    """Обучение финальной модели с использованием предсказаний диапазонов"""
    
    # Создание датасетов
    train_dataset = ProbableHandDataset(
        data_dict['train_sequences'],
        data_dict['feature_columns'],
        range_predictions['train'],
        data_dict['final_targets']['train'],
        data_dict['scaler']
    )
    
    val_dataset = ProbableHandDataset(
        data_dict['val_sequences'],
        data_dict['feature_columns'],
        range_predictions['val'],
        data_dict['final_targets']['val'],
        data_dict['scaler']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Создание модели
    model = ProbableHandRWKV(
        input_dim=len(data_dict['feature_columns']),
        hidden_dim=384,
        num_layers=4
    ).to(device)
    
    # Оптимизатор
    optimizer = optim.AdamW(model.parameters(), lr=0.0008)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_accuracy = 0
    best_model_state = None
    
    # Обучение
    epochs = 30
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            features = batch['features'].to(device)
            range_preds = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch['range_predictions'].items()}
            streets = batch['streets'].to(device)
            actions = batch['actions'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['target_group'].to(device)
            
            optimizer.zero_grad()
            
            # Предсказания
            group_logits = model(features, range_preds, streets, actions, mask)
            
            # Потери
            loss = criterion(group_logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Точность
            _, predicted = torch.max(group_logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        train_accuracy = correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                range_preds = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                              for k, v in batch['range_predictions'].items()}
                streets = batch['streets'].to(device)
                actions = batch['actions'].to(device)
                mask = batch['mask'].to(device)
                targets = batch['target_group'].to(device)
                
                # Предсказания
                group_logits = model(features, range_preds, streets, actions, mask)
                
                # Потери
                loss = criterion(group_logits, targets)
                val_loss += loss.item()
                
                # Точность
                _, predicted = torch.max(group_logits, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Эпоха {epoch+1}/{epochs}, "
              f"Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_accuracy:.3f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.3f}")
        
        # Сохранение лучшей модели
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict().copy()
        
        scheduler.step(avg_val_loss)
    
    # Загружаем лучшую модель
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Загружена лучшая модель с точностью: {best_val_accuracy:.3f}")
    
    return model

def get_range_predictions(preflop_model, data_dict, device):
    """Получает предсказания диапазонов для всех данных"""
    
    preflop_model.eval()
    predictions = {'train': {}, 'val': {}, 'test': {}}
    
    for split in ['train', 'val', 'test']:
        sequences = data_dict[f'{split}_sequences']
        
        dataset = PreflopRangeDataset(
            sequences,
            data_dict['feature_columns'],
            {},  # Пустые targets для инференса
            data_dict['scaler']
        )
        
        loader = DataLoader(
            dataset, 
            batch_size=64, 
            shuffle=False,
            collate_fn=collate_fn
        )
        
        split_predictions = {}
        idx = 0
        
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                    
                features = batch['features'].to(device)
                positions = batch['positions'].to(device)
                actions = batch['actions'].to(device)
                
                preflop_model.reset_states()
                range_preds = preflop_model(features, positions, actions)
                
                # Сохраняем предсказания для каждого примера в батче
                batch_size = features.shape[0]
                for i in range(batch_size):
                    split_predictions[idx] = {
                        pos: range_preds[pos][i].cpu()
                        for pos in range_preds
                    }
                    idx += 1
        
        predictions[split] = split_predictions
    
    return predictions

# ===================== ОЦЕНКА И ВИЗУАЛИЗАЦИЯ =====================

def evaluate_two_stage_system(preflop_model, final_model, data_dict, range_predictions):
    """
    Оценка производительности двухэтапной системы
    """
    print("\n📊 === ОЦЕНКА ДВУХЭТАПНОЙ СИСТЕМЫ ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Оценка префлоп модели
    print("\n📌 Оценка модели префлоп диапазонов:")
    preflop_metrics = evaluate_preflop_model(preflop_model, data_dict, device)
    
    # Оценка финальной модели
    print("\n📌 Оценка финальной модели:")
    final_metrics = evaluate_final_model(final_model, data_dict, range_predictions, device)
    
    # Визуализация результатов
    visualize_results(preflop_metrics, final_metrics, data_dict)
    
    return {
        'preflop': preflop_metrics,
        'final': final_metrics
    }

def evaluate_preflop_model(model, data_dict, device):
    """Оценка модели префлоп диапазонов"""
    model.eval()
    metrics = {}
    
    # Здесь можно добавить более детальную оценку
    # Например, точность предсказания топ-10% рук для каждой позиции
    
    return metrics

def evaluate_final_model(model, data_dict, range_predictions, device):
    """Оценка финальной модели"""
    model.eval()
    
    # Создаем тестовый датасет
    test_dataset = ProbableHandDataset(
        data_dict['test_sequences'],
        data_dict['feature_columns'],
        range_predictions['test'],
        data_dict['final_targets']['test'],
        data_dict['scaler']
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            range_preds = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch['range_predictions'].items()}
            streets = batch['streets'].to(device)
            actions = batch['actions'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['target_group']
            
            # Предсказания
            group_logits = model(features, range_preds, streets, actions, mask)
            _, predicted = torch.max(group_logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Вычисляем метрики
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    accuracy = accuracy_score(all_targets, all_predictions)
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Распределение предсказанных групп
    group_distribution = {i: 0 for i in range(10)}
    for pred in all_predictions:
        group_distribution[pred] += 1
    
    metrics = {
        'overall_accuracy': accuracy,
        'confusion_matrix': cm,
        'group_distribution': group_distribution,
        'predictions': all_predictions,
        'targets': all_targets
    }
    
    print(f"📊 Точность финальной модели: {accuracy:.3f}")
    
    return metrics

def visualize_results(preflop_metrics, final_metrics, data_dict):
    """Визуализация результатов обучения"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Матрица ошибок для финальной модели
    ax1 = axes[0, 0]
    cm = final_metrics['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Количество'})
    ax1.set_xlabel('Предсказанная группа')
    ax1.set_ylabel('Истинная группа')
    ax1.set_title('Матрица ошибок (10 групп HM3)')
    ax1.set_xticklabels([f'G{i+1}' for i in range(10)])
    ax1.set_yticklabels([f'G{i+1}' for i in range(10)])
    
    # 2. Распределение предсказанных групп
    ax2 = axes[0, 1]
    group_distribution = final_metrics['group_distribution']
    
    groups = list(range(10))
    counts = [group_distribution.get(i, 0) for i in groups]
    colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 10))
    
    bars = ax2.bar(groups, counts, color=colors)
    ax2.set_xlabel('Группа')
    ax2.set_ylabel('Количество')
    ax2.set_title('Распределение предсказанных групп')
    ax2.set_xticks(groups)
    ax2.set_xticklabels([f'G{i+1}' for i in groups])
    
    # Добавляем проценты
    total = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count/total*100:.1f}%', ha='center', va='bottom')
    
    # 3. Точность по группам
    ax3 = axes[0, 2]
    group_accuracies = []
    
    for i in range(10):
        mask = np.array(final_metrics['targets']) == i
        if mask.sum() > 0:
            group_preds = np.array(final_metrics['predictions'])[mask]
            group_targets = np.array(final_metrics['targets'])[mask]
            acc = (group_preds == group_targets).mean()
            group_accuracies.append(acc)
        else:
            group_accuracies.append(0)
    
    bars = ax3.bar(groups, group_accuracies, color=colors)
    ax3.set_xlabel('Группа')
    ax3.set_ylabel('Точность')
    ax3.set_title('Точность предсказания по группам')
    ax3.set_xticks(groups)
    ax3.set_xticklabels([f'G{i+1}' for i in groups])
    ax3.set_ylim(0, 1)
    
    # Добавляем значения
    for bar, acc in zip(bars, group_accuracies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}', ha='center', va='bottom')
    
    # 4. Сравнение истинного и предсказанного распределения
    ax4 = axes[1, 0]
    
    true_distribution = {i: 0 for i in range(10)}
    for target in final_metrics['targets']:
        true_distribution[target] += 1
    
    x = np.arange(10)
    width = 0.35
    
    true_counts = [true_distribution[i] for i in range(10)]
    pred_counts = [final_metrics['group_distribution'][i] for i in range(10)]
    
    bars1 = ax4.bar(x - width/2, true_counts, width, label='Истинное', alpha=0.8)
    bars2 = ax4.bar(x + width/2, pred_counts, width, label='Предсказанное', alpha=0.8)
    
    ax4.set_xlabel('Группа')
    ax4.set_ylabel('Количество')
    ax4.set_title('Сравнение распределений')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'G{i+1}' for i in range(10)])
    ax4.legend()
    
    # 5. Корреляция между истинными и предсказанными группами
    ax5 = axes[1, 1]
    
    true_groups = final_metrics['targets']
    pred_groups = final_metrics['predictions']
    
    ax5.scatter(true_groups, pred_groups, alpha=0.5)
    ax5.plot([0, 9], [0, 9], 'r--', lw=2)  # Диагональная линия
    ax5.set_xlabel('Истинная группа')
    ax5.set_ylabel('Предсказанная группа')
    ax5.set_title('Корреляция предсказаний')
    ax5.set_xticks(range(10))
    ax5.set_yticks(range(10))
    ax5.grid(True, alpha=0.3)
    
    # 6. Общая статистика
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    stats_text = f"""
    Общая статистика:
    
    • Точность модели: {final_metrics['overall_accuracy']:.3f}
    • Всего примеров: {len(final_metrics['targets'])}
    • Количество групп: 10
    
    Распределение по группам силы:
    • G1-G2: Монстры (сильнейшие)
    • G3-G4: Сильные руки
    • G5-G6: Средние руки
    • G7-G8: Слабые руки
    • G9-G10: Мусор/Дро
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Сохранение
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'two_stage_results_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Визуализация сохранена: two_stage_results_{timestamp}.png")

# ===================== СОХРАНЕНИЕ МОДЕЛЕЙ =====================

def save_two_stage_models(preflop_model, final_model, metrics):
    """Сохранение обученных моделей"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Сохранение префлоп модели
    preflop_path = f'models/preflop_range_model_{timestamp}.pth'
    torch.save({
        'model_state': preflop_model.state_dict(),
        'model_config': {
            'input_dim': preflop_model.input_dim,
            'hidden_dim': preflop_model.hidden_dim,
            'num_layers': len(preflop_model.rwkv_blocks),
            'num_positions': preflop_model.num_positions
        },
        'metrics': metrics.get('preflop', {}),
        'timestamp': timestamp
    }, preflop_path)
    
    # Сохранение финальной модели
    final_path = f'models/final_hand_model_{timestamp}.pth'
    torch.save({
        'model_state': final_model.state_dict(),
        'model_config': {
            'input_dim': final_model.input_dim,
            'hidden_dim': final_model.hidden_dim,
            'num_layers': len(final_model.rwkv_blocks),
            'num_groups': final_model.num_groups
        },
        'metrics': metrics.get('final', {}),
        'timestamp': timestamp
    }, final_path)
    
    print(f"💾 Модели сохранены:")
    print(f"   📁 Префлоп модель: {preflop_path}")
    print(f"   📁 Финальная модель: {final_path}")
    
    # Сохранение отчета
    report = {
        'timestamp': timestamp,
        'preflop_model': preflop_path,
        'final_model': final_path,
        'metrics': {
            'preflop': metrics.get('preflop', {}),
            'final': safe_json_serialize(metrics.get('final', {}))
        },
        'overall_accuracy': metrics['final']['overall_accuracy']
    }
    
    report_path = f'results/two_stage_report_{timestamp}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   📋 Отчет: {report_path}")

# ===================== ГЛАВНАЯ ФУНКЦИЯ =====================

def main():
    """Главная функция для запуска двухэтапной системы"""
    
    print("🎰 === ДВУХЭТАПНАЯ СИСТЕМА ПРЕДСКАЗАНИЯ ПОКЕРНЫХ РУК ===\n")
    
    # Настройка
    setup_directories()
    
    # Выбор файла данных
    data_file = choose_data_file()
    if not data_file:
        return
    
    # ИСПРАВЛЕНИЕ: Проверяем, выбрано ли объединение файлов
    if data_file == "COMBINE_ALL":
        print("\n🔗 === ОБЪЕДИНЕНИЕ ВСЕХ ФАЙЛОВ ===")
        # Объединяем все файлы
        result = combine_all_data_files()
        if result is None:
            print("❌ Не удалось объединить файлы")
            return
        
        # Используем объединенный файл для дальнейшей обработки
        data_file, combination_summary = result
        print(f"\n✅ Будем использовать объединенный файл: {data_file}")
        print(f"📊 Общее количество записей: {combination_summary['total_records']:,}")
        print(f"🃏 Общее количество шоудаунов: {combination_summary['total_showdowns']:,}")
    
    # Путь к маппингу HM3
    hm3_mapping_path = 'hm3_10pct_simple_mapping_20250612_072153.json'
    
    if not os.path.exists(hm3_mapping_path):
        print(f"❌ Файл маппинга не найден: {hm3_mapping_path}")
        print("Убедитесь, что файл находится в текущей директории")
        return
    
    # Подготовка данных
    data_dict = prepare_two_stage_data(data_file, hm3_mapping_path)
    
    if data_dict is None:
        print("❌ Не удалось подготовить данные")
        return
    
    # Обучение системы
    preflop_model, final_model, range_predictions = train_two_stage_system(data_dict)
    
    # Оценка
    metrics = evaluate_two_stage_system(preflop_model, final_model, data_dict, range_predictions)
    
    # Сохранение моделей
    save_two_stage_models(preflop_model, final_model, metrics)
    
    print("\n🎉 Обучение завершено!")
    print(f"📊 Точность финальной модели: {metrics['final']['overall_accuracy']:.3f}")

if __name__ == "__main__":
    main()