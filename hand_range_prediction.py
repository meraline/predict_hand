"""
hand_range_prediction.py - Модель RWKV для предсказания диапазонов рук в покере
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle
import json
import glob
from datetime import datetime


# ---------------------- 1. Модель RWKV ----------------------
class RWKV_Block(nn.Module):
    def __init__(self, hidden_dim):
        super(RWKV_Block, self).__init__()
        self.hidden_dim = hidden_dim

        # Параметры time-mixing
        self.time_decay = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.time_mix_k = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(hidden_dim) * 0.5)

        # Слои
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Состояние
        self.register_buffer("state", None, persistent=False)

    def reset_state(self):
        self.state = None

    def _init_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim).to(self.time_decay.device)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        if self.state is None or self.state.size(0) != batch_size:
            self.state = self._init_state(batch_size)

        output = []
        for t in range(seq_len):
            xt = x[:, t]

            # Time-mixing
            k = self.key(xt * self.time_mix_k + self.state * (1 - self.time_mix_k))
            v = self.value(xt * self.time_mix_v + self.state * (1 - self.time_mix_v))
            r = torch.sigmoid(
                self.receptance(
                    xt * self.time_mix_r + self.state * (1 - self.time_mix_r)
                )
            )

            # Обновление состояния
            self.state = xt + self.state * torch.exp(-torch.exp(self.time_decay))

            # Вычисление выхода
            out = r * self.output(v)
            output.append(out)

        return torch.stack(output, dim=1)


class HandRangeRWKV(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(HandRangeRWKV, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.rwkv_layers = nn.ModuleList(
            [RWKV_Block(hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        # Выходные слои для разных предсказаний
        self.hand_strength_head = nn.Linear(
            hidden_dim, 10
        )  # 10 уровней силы руки (0-9)
        self.category_head = nn.Linear(hidden_dim, 9)  # 9 категорий рук
        self.specific_hand_head = nn.Linear(hidden_dim, 13)  # 13 рангов карт

        # Sigmoid только для specific_hand
        self.sigmoid = nn.Sigmoid()

    def reset_states(self):
        for layer in self.rwkv_layers:
            layer.reset_state()

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        for layer in self.rwkv_layers:
            residual = x
            x = layer(x)
            x = residual + x
            x = self.norm(x)
            x = self.dropout(x)

        x = x.squeeze(1)  # [batch_size, hidden_dim]

        # Многозадачный выход
        hand_strength = self.hand_strength_head(x)
        category_logits = self.category_head(x)
        specific_hand = self.sigmoid(self.specific_hand_head(x))

        return {
            "hand_strength": hand_strength,
            "category_probs": category_logits,
            "specific_hand": specific_hand,
        }


# ---------------------- 2. Анализ покерных рук ----------------------
class PokerHandAnalyzer:
    """Класс для анализа силы покерных рук"""

    RANK_MAP = {
        "A": 14,
        "K": 13,
        "Q": 12,
        "J": 11,
        "T": 10,
        "9": 9,
        "8": 8,
        "7": 7,
        "6": 6,
        "5": 5,
        "4": 4,
        "3": 3,
        "2": 2,
    }

    SUIT_MAP = {"s": 0, "h": 1, "d": 2, "c": 3}

    HAND_CATEGORIES = {
        "premium_pair": ["AA", "KK", "QQ"],
        "strong_pair": ["JJ", "TT", "99"],
        "medium_pair": ["88", "77", "66", "55"],
        "small_pair": ["44", "33", "22"],
        "premium_ace": ["AKs", "AKo", "AQs", "AQo"],
        "strong_ace": ["AJs", "AJo", "ATs", "ATo"],
        "suited_connector": ["KQs", "QJs", "JTs", "T9s", "98s", "87s", "76s"],
        "offsuit_broadway": ["KQo", "QJo", "JTo", "KJo"],
    }

    @staticmethod
    def get_all_categories():
        """Возвращает все доступные категории"""
        return list(PokerHandAnalyzer.HAND_CATEGORIES.keys()) + ["other"]

    @staticmethod
    def get_category_mapping():
        """Возвращает маппинг категорий для JSON"""
        category_mapping = {
            "premium_pair": 0,
            "strong_pair": 1,
            "medium_pair": 2,
            "small_pair": 3,
            "premium_ace": 4,
            "strong_ace": 5,
            "suited_connector": 6,
            "offsuit_broadway": 7,
            "other": 8,
        }
        return category_mapping

    @staticmethod
    def get_rank_names():
        """Возвращает названия рангов для JSON"""
        return ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]

    @staticmethod
    def parse_card(card_str):
        """Парсинг карты из строки (например, 'As' -> (14, 0))"""
        if pd.isna(card_str) or card_str == "" or len(card_str) < 2:
            return None, None

        rank_char = card_str[0].upper()
        suit_char = card_str[1].lower()

        rank = PokerHandAnalyzer.RANK_MAP.get(rank_char)
        suit = PokerHandAnalyzer.SUIT_MAP.get(suit_char)

        return rank, suit

    @staticmethod
    def analyze_hand_strength(card1, card2):
        """Анализ силы стартовой руки (префлоп)"""
        rank1, suit1 = PokerHandAnalyzer.parse_card(card1)
        rank2, suit2 = PokerHandAnalyzer.parse_card(card2)

        if rank1 is None or rank2 is None:
            return 0, "unknown"

        # Нормализуем порядок карт (старшая первой)
        if rank1 < rank2:
            rank1, rank2 = rank2, rank1
            suit1, suit2 = suit2, suit1

        is_suited = suit1 == suit2
        is_pair = rank1 == rank2

        # Определение категории и силы
        hand_str = PokerHandAnalyzer._format_hand_string(
            rank1, rank2, is_suited, is_pair
        )
        category = PokerHandAnalyzer._get_hand_category(hand_str)
        strength = PokerHandAnalyzer._calculate_hand_strength(
            rank1, rank2, is_suited, is_pair
        )

        return strength, category

    @staticmethod
    def _format_hand_string(rank1, rank2, is_suited, is_pair):
        """Форматирование руки в стандартную строку"""
        rank_chars = {v: k for k, v in PokerHandAnalyzer.RANK_MAP.items()}

        if is_pair:
            return f"{rank_chars[rank1]}{rank_chars[rank2]}"
        else:
            suffix = "s" if is_suited else "o"
            return f"{rank_chars[rank1]}{rank_chars[rank2]}{suffix}"

    @staticmethod
    def _get_hand_category(hand_str):
        """Определение категории руки"""
        for category, hands in PokerHandAnalyzer.HAND_CATEGORIES.items():
            if hand_str in hands:
                return category
        return "other"

    @staticmethod
    def _calculate_hand_strength(rank1, rank2, is_suited, is_pair):
        """Расчет силы руки (0-9 шкала)"""
        if is_pair:
            if rank1 >= 14:  # AA
                return 9
            elif rank1 >= 13:  # KK
                return 8
            elif rank1 >= 12:  # QQ
                return 7
            elif rank1 >= 11:  # JJ
                return 6
            elif rank1 >= 10:  # TT
                return 5
            elif rank1 >= 9:  # 99
                return 4
            elif rank1 >= 8:  # 88
                return 3
            elif rank1 >= 7:  # 77
                return 2
            elif rank1 >= 6:  # 66
                return 1
            else:
                return 0
        else:
            # Непарные руки
            high_card_bonus = max(0, rank1 - 10)
            second_card_bonus = max(0, rank2 - 8)
            suited_bonus = 1 if is_suited else 0
            connector_bonus = 1 if abs(rank1 - rank2) == 1 else 0

            base_strength = (
                high_card_bonus + second_card_bonus + suited_bonus + connector_bonus
            )
            return min(9, max(0, base_strength))


# ---------------------- 3. Подготовка данных ----------------------
class HandRangeDataset(Dataset):
    def __init__(self, features, targets=None):
        # Исправление: правильно обрабатываем DataFrame и numpy массивы
        if hasattr(features, "values"):
            self.features = torch.tensor(
                features.values.astype(np.float32), dtype=torch.float32
            )
        else:
            self.features = torch.tensor(
                features.astype(np.float32), dtype=torch.float32
            )

        self.has_targets = targets is not None

        if self.has_targets:
            self.hand_strength = torch.tensor(
                targets["hand_strength"], dtype=torch.long
            )
            self.category_probs = torch.tensor(
                targets["category_probs"], dtype=torch.float32
            )
            self.specific_hand = torch.tensor(
                targets["specific_hand"], dtype=torch.float32
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.has_targets:
            return (
                self.features[idx],
                {
                    "hand_strength": self.hand_strength[idx],
                    "category_probs": self.category_probs[idx],
                    "specific_hand": self.specific_hand[idx],
                },
            )
        else:
            return self.features[idx]


def create_target_variables(df):
    """Создание всех целевых переменных на основе РЕАЛЬНЫХ карт"""
    n_samples = len(df)

    # Целевые переменные
    targets = {
        "hand_strength": df["hand_strength"].values.astype(int),
        "category_probs": np.zeros((n_samples, 9)),  # 9 категорий
        "specific_hand": np.zeros((n_samples, 13)),  # 13 рангов
    }

    # Кодирование категорий в one-hot
    category_mapping = PokerHandAnalyzer.get_category_mapping()

    for i, category in enumerate(df["hand_category"]):
        cat_idx = category_mapping.get(category, 8)  # 8 = other
        targets["category_probs"][i, cat_idx] = 1.0

    # Для specific_hand используем реальные карты игрока
    analyzer = PokerHandAnalyzer()

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            card1 = row["Showdown_1"]
            card2 = row["Showdown_2"]

            if pd.notna(card1) and pd.notna(card2):
                rank1, suit1 = analyzer.parse_card(card1)
                rank2, suit2 = analyzer.parse_card(card2)

                if rank1 is not None and rank2 is not None:
                    # Преобразуем ранги в индексы (A=14 -> 12, K=13 -> 11, ..., 2=2 -> 0)
                    rank1_idx = max(0, min(12, rank1 - 2))
                    rank2_idx = max(0, min(12, rank2 - 2))

                    targets["specific_hand"][i, rank1_idx] = 0.6
                    targets["specific_hand"][i, rank2_idx] = 0.6

                    # Немного вероятности на соседние ранги
                    for offset in [-1, 1]:
                        for rank_idx in [rank1_idx, rank2_idx]:
                            neighbor_idx = rank_idx + offset
                            if 0 <= neighbor_idx < 13:
                                targets["specific_hand"][i, neighbor_idx] = 0.1
        except Exception as e:
            # В случае ошибки, равномерное распределение
            targets["specific_hand"][i, :] = 1 / 13

    # Нормализуем specific_hand чтобы сумма была 1
    for i in range(n_samples):
        row_sum = targets["specific_hand"][i, :].sum()
        if row_sum > 0:
            targets["specific_hand"][i, :] /= row_sum
        else:
            targets["specific_hand"][i, :] = 1 / 13

    return targets


def prepare_hand_range_data(file_path, include_hole_cards=True):
    """Подготовка данных для предсказания диапазонов рук"""
    print(f"Загрузка данных из {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Загружено {len(df)} строк данных")

    # Фильтруем только записи с известными картами
    mask = (df["Showdown_1"].notna()) & (df["Showdown_2"].notna())
    df_filtered = df[mask].copy().reset_index(drop=True)

    print(f"Найдено {len(df_filtered)} записей с открытыми картами для обучения")

    if len(df_filtered) == 0:
        print("Ошибка: не найдено записей с открытыми картами!")
        return None

    # Анализ рук и создание целевых переменных
    analyzer = PokerHandAnalyzer()
    hand_analysis = df_filtered.apply(
        lambda row: analyzer.analyze_hand_strength(
            row["Showdown_1"], row["Showdown_2"]
        ),
        axis=1,
    )
    df_filtered["hand_strength"] = [x[0] for x in hand_analysis]
    df_filtered["hand_category"] = [x[1] for x in hand_analysis]

    print(
        f"Анализ рук завершен. Модель {'ВИДИТ' if include_hole_cards else 'НЕ ВИДИТ'} карты игрока в признаках."
    )

    # Подготовка признаков
    feature_columns = [
        "Level",
        "Pot",
        "Stack",
        "SPR",
        "Street_id",
        "Round",
        "ActionOrder",
        "Seat",
        "Dealer",
        "Bet",
        "Allin",
        "PlayerWins",
        "WinAmount",
    ]

    # Добавляем информацию о картах стола
    board_columns = ["Card1", "Card2", "Card3", "Card4", "Card5"]
    for col in board_columns:
        if col in df_filtered.columns:
            df_filtered[f"{col}_rank"], df_filtered[f"{col}_suit"] = zip(
                *df_filtered[col].apply(
                    lambda x: analyzer.parse_card(x) if pd.notna(x) else (0, 0)
                )
            )
            feature_columns.extend([f"{col}_rank", f"{col}_suit"])

    # Добавляем карты игрока ТОЛЬКО если это модель с картами
    if include_hole_cards:
        print("Добавляем карты игрока в признаки...")
        df_filtered["hole1_rank"], df_filtered["hole1_suit"] = zip(
            *df_filtered["Showdown_1"].apply(
                lambda x: analyzer.parse_card(x) if pd.notna(x) else (0, 0)
            )
        )
        df_filtered["hole2_rank"], df_filtered["hole2_suit"] = zip(
            *df_filtered["Showdown_2"].apply(
                lambda x: analyzer.parse_card(x) if pd.notna(x) else (0, 0)
            )
        )
        feature_columns.extend(["hole1_rank", "hole1_suit", "hole2_rank", "hole2_suit"])
    else:
        print("Карты игрока НЕ включены в признаки")

    # Кодирование категориальных переменных
    if "Position" in df_filtered.columns:
        position_encoder = LabelEncoder()
        df_filtered["Position_encoded"] = position_encoder.fit_transform(
            df_filtered["Position"].fillna("Unknown")
        )
        feature_columns.append("Position_encoded")

    if "Action" in df_filtered.columns:
        action_encoder = LabelEncoder()
        df_filtered["Action_encoded_feature"] = action_encoder.fit_transform(
            df_filtered["Action"].fillna("Unknown")
        )
        feature_columns.append("Action_encoded_feature")

    if "TypeBuyIn" in df_filtered.columns:
        buyin_encoder = LabelEncoder()
        df_filtered["TypeBuyIn_encoded"] = buyin_encoder.fit_transform(
            df_filtered["TypeBuyIn"].fillna("Unknown")
        )
        feature_columns.append("TypeBuyIn_encoded")

    # Обработка проблемных значений
    print("Обработка проблемных значений...")

    # Обработка SPR
    if "SPR" in df_filtered.columns:
        pot_zero_mask = (df_filtered["Pot"] == 0) | (df_filtered["Pot"].isnull())
        if pot_zero_mask.any():
            df_filtered.loc[pot_zero_mask, "SPR"] = 100.0

        spr_95th = df_filtered["SPR"].quantile(0.95)
        extreme_spr_mask = df_filtered["SPR"] > spr_95th * 2
        if extreme_spr_mask.any():
            df_filtered.loc[extreme_spr_mask, "SPR"] = spr_95th

        spr_median = df_filtered["SPR"].median()
        df_filtered["SPR"] = df_filtered["SPR"].fillna(spr_median)

    # Обработка остальных колонок
    for col in feature_columns:
        if col in df_filtered.columns:
            if df_filtered[col].dtype in ["int64", "float64"]:
                df_filtered[col] = df_filtered[col].replace([np.inf, -np.inf], np.nan)
                median_val = df_filtered[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df_filtered[col] = df_filtered[col].fillna(median_val)
            else:
                df_filtered[col] = df_filtered[col].fillna(0)

    # Отбор доступных признаков
    available_features = [col for col in feature_columns if col in df_filtered.columns]
    X = df_filtered[available_features].copy()

    print(f"Используется {len(available_features)} признаков")

    # Создание целевых переменных
    y = create_target_variables(df_filtered)

    # ИСПРАВЛЕНИЕ: Проверяем количество уникальных классов для стратификации
    unique_strengths = np.unique(y["hand_strength"])
    min_class_count = min(
        [np.sum(y["hand_strength"] == strength) for strength in unique_strengths]
    )

    # Если слишком мало данных для стратификации, не используем ее
    use_stratify = len(df_filtered) >= 10 and min_class_count >= 2

    if not use_stratify:
        print(
            f"⚠️  Отключена стратификация из-за малого количества данных (min_class_count={min_class_count})"
        )

    # Разделение данных
    X_train, X_test, y_hand_train, y_hand_test = train_test_split(
        X,
        y["hand_strength"],
        test_size=0.2,
        random_state=42,
        stratify=y["hand_strength"] if use_stratify else None,
    )

    # Создаем целевые переменные для train/test
    train_indices = X_train.index
    test_indices = X_test.index

    y_train = {
        "hand_strength": y["hand_strength"][train_indices],
        "category_probs": y["category_probs"][train_indices],
        "specific_hand": y["specific_hand"][train_indices],
    }
    y_test = {
        "hand_strength": y["hand_strength"][test_indices],
        "category_probs": y["category_probs"][test_indices],
        "specific_hand": y["specific_hand"][test_indices],
    }

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Создание датасетов
    train_dataset = HandRangeDataset(X_train_scaled, y_train)
    test_dataset = HandRangeDataset(X_test_scaled, y_test)

    batch_size = min(32, max(4, len(train_dataset) // 4))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_columns": available_features,
        "input_dim": len(available_features),
        "include_hole_cards": include_hole_cards,
    }


# ---------------------- 4. Обучение модели ----------------------
def train_hand_range_model(
    data_dict, hidden_dim=128, num_layers=3, epochs=20, lr=0.001
):
    """Обучение модели предсказания диапазонов рук"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    train_size = len(data_dict["train_loader"].dataset)
    test_size = len(data_dict["test_loader"].dataset)
    print(f"Размер обучающей выборки: {train_size}")
    print(f"Размер тестовой выборки: {test_size}")

    if train_size == 0:
        print("Ошибка: пустая обучающая выборка!")
        return None, None

    # Создание модели
    model = HandRangeRWKV(
        input_dim=data_dict["input_dim"], hidden_dim=hidden_dim, num_layers=num_layers
    ).to(device)

    # Оптимизатор
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Функции потерь
    strength_criterion = nn.CrossEntropyLoss()
    category_criterion = nn.BCEWithLogitsLoss()
    specific_criterion = nn.MSELoss()

    # История обучения
    history = {"train_loss": [], "val_loss": [], "strength_acc": []}
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Обучение
        model.train()
        train_losses = []
        strength_correct = 0
        strength_total = 0

        for batch_idx, (inputs, targets) in enumerate(data_dict["train_loader"]):
            inputs = inputs.to(device)
            target_strength = targets["hand_strength"].to(device)
            target_category = targets["category_probs"].to(device)
            target_specific = targets["specific_hand"].to(device)

            model.reset_states()
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(inputs)

            # Вычисление потерь
            strength_loss = strength_criterion(
                outputs["hand_strength"], target_strength
            )
            category_loss = category_criterion(
                outputs["category_probs"], target_category
            )
            specific_loss = specific_criterion(
                outputs["specific_hand"], target_specific
            )

            # Общая потеря
            total_loss = strength_loss + 0.5 * category_loss + 0.3 * specific_loss

            # Обратный проход
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(total_loss.item())

            # Точность силы руки
            _, predicted = torch.max(outputs["hand_strength"], 1)
            strength_total += target_strength.size(0)
            strength_correct += (predicted == target_strength).sum().item()

        # Валидация
        model.eval()
        val_losses = []
        val_strength_correct = 0
        val_strength_total = 0

        with torch.no_grad():
            for inputs, targets in data_dict["test_loader"]:
                inputs = inputs.to(device)
                target_strength = targets["hand_strength"].to(device)
                target_category = targets["category_probs"].to(device)
                target_specific = targets["specific_hand"].to(device)

                model.reset_states()
                outputs = model(inputs)

                # Потери
                strength_loss = strength_criterion(
                    outputs["hand_strength"], target_strength
                )
                category_loss = category_criterion(
                    outputs["category_probs"], target_category
                )
                specific_loss = specific_criterion(
                    outputs["specific_hand"], target_specific
                )
                total_loss = strength_loss + 0.5 * category_loss + 0.3 * specific_loss

                val_losses.append(total_loss.item())

                # Точность
                _, predicted = torch.max(outputs["hand_strength"], 1)
                val_strength_total += target_strength.size(0)
                val_strength_correct += (predicted == target_strength).sum().item()

        # Метрики эпохи
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        avg_val_loss = np.mean(val_losses) if val_losses else 0
        train_strength_acc = strength_correct / max(strength_total, 1)
        val_strength_acc = val_strength_correct / max(val_strength_total, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["strength_acc"].append(val_strength_acc)

        print(f"Эпоха {epoch+1}/{epochs}:")
        print(f"  Потери: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")
        print(
            f"  Точность силы руки: train={train_strength_acc:.3f}, val={val_strength_acc:.3f}"
        )

        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        scheduler.step(avg_val_loss)

    return model, history


# ---------------------- 5. Оценка и визуализация ----------------------
def evaluate_model_performance(model, data_dict, include_hole_cards=True):
    """Детальная оценка производительности модели"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_outputs = {
        "strength_pred": [],
        "strength_true": [],
        "category_pred": [],
        "category_true": [],
        "specific_pred": [],
        "specific_true": [],
    }

    with torch.no_grad():
        for inputs, targets in data_dict["test_loader"]:
            inputs = inputs.to(device)
            model.reset_states()
            outputs = model(inputs)

            # Собираем все предсказания
            _, strength_pred = torch.max(outputs["hand_strength"], 1)
            all_outputs["strength_pred"].extend(strength_pred.cpu().numpy())
            all_outputs["strength_true"].extend(targets["hand_strength"].numpy())

            category_pred = torch.sigmoid(outputs["category_probs"]).cpu().numpy()
            category_true = targets["category_probs"].numpy()
            all_outputs["category_pred"].extend(np.argmax(category_pred, axis=1))
            all_outputs["category_true"].extend(np.argmax(category_true, axis=1))

            all_outputs["specific_pred"].extend(outputs["specific_hand"].cpu().numpy())
            all_outputs["specific_true"].extend(targets["specific_hand"].numpy())

    # Вычисляем метрики
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        mean_squared_error,
    )

    strength_accuracy = accuracy_score(
        all_outputs["strength_true"], all_outputs["strength_pred"]
    )
    category_accuracy = accuracy_score(
        all_outputs["category_true"], all_outputs["category_pred"]
    )
    specific_mse = mean_squared_error(
        all_outputs["specific_true"], all_outputs["specific_pred"]
    )

    print(
        f"\n=== Оценка модели ({'с картами' if include_hole_cards else 'без карт'}) ==="
    )
    print(f"Точность предсказания силы руки: {strength_accuracy:.3f}")
    print(f"Точность предсказания категории: {category_accuracy:.3f}")
    print(f"MSE для конкретных рангов: {specific_mse:.4f}")

    # Детальный отчет по категориям
    category_names = PokerHandAnalyzer.get_all_categories()
    print(f"\nОтчет по категориям рук:")
    print(
        classification_report(
            all_outputs["category_true"],
            all_outputs["category_pred"],
            target_names=category_names,
            zero_division=0,
        )
    )

    return {
        "strength_accuracy": strength_accuracy,
        "category_accuracy": category_accuracy,
        "specific_mse": specific_mse,
        **all_outputs,
    }


def visualize_results(model, data_dict, history, include_hole_cards=True):
    """Визуализация результатов обучения и предсказаний"""

    # График обучения
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Потери
    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Validation")
    axes[0, 0].set_title("Потери обучения")
    axes[0, 0].set_xlabel("Эпоха")
    axes[0, 0].set_ylabel("Потери")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Точность
    axes[0, 1].plot(history["strength_acc"], label="Точность силы руки")
    axes[0, 1].set_title("Точность предсказания силы руки")
    axes[0, 1].set_xlabel("Эпоха")
    axes[0, 1].set_ylabel("Точность")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Предсказания модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_dict["test_loader"]:
            inputs = inputs.to(device)
            model.reset_states()
            outputs = model(inputs)

            _, strength_pred = torch.max(outputs["hand_strength"], 1)
            all_predictions.extend(strength_pred.cpu().numpy())
            all_targets.extend(targets["hand_strength"].numpy())

    # Матрица ошибок для силы руки
    from sklearn.metrics import confusion_matrix

    cm_strength = confusion_matrix(all_targets, all_predictions)

    im1 = axes[1, 0].imshow(cm_strength, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1, 0].set_title("Матрица ошибок: Сила руки")
    axes[1, 0].set_xlabel("Предсказанная сила")
    axes[1, 0].set_ylabel("Истинная сила")

    # Добавляем числа в матрицу
    for i in range(cm_strength.shape[0]):
        for j in range(cm_strength.shape[1]):
            axes[1, 0].text(
                j,
                i,
                str(cm_strength[i, j]),
                ha="center",
                va="center",
                color="white" if cm_strength[i, j] > cm_strength.max() / 2 else "black",
            )

    # Распределение предсказанных сил
    axes[1, 1].hist(
        all_predictions, bins=10, alpha=0.7, label="Предсказанные", density=True
    )
    axes[1, 1].hist(all_targets, bins=10, alpha=0.7, label="Истинные", density=True)
    axes[1, 1].set_title("Распределение силы рук")
    axes[1, 1].set_xlabel("Сила руки (0-9)")
    axes[1, 1].set_ylabel("Плотность")
    axes[1, 1].legend()

    model_type = "с картами игрока" if include_hole_cards else "без карт игрока"
    plt.suptitle(
        f"Результаты модели предсказания диапазонов рук ({model_type})", fontsize=16
    )
    plt.tight_layout()

    # Сохранение графика
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_name = f"training_results_{'with_cards' if include_hole_cards else 'without_cards'}_{timestamp}.png"

    # Создаем папку plots если её нет
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", plot_name)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"График сохранен: {plot_path}")

    return fig, plot_path


def predict_hand_ranges(model, data_dict, sample_hands=5):
    """Демонстрация предсказаний на примерах"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    category_names = PokerHandAnalyzer.get_all_categories()
    rank_names = PokerHandAnalyzer.get_rank_names()

    print(f"\n=== Примеры предсказаний ===")

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_dict["test_loader"]):
            if i >= sample_hands:
                break

            inputs = inputs.to(device)
            model.reset_states()
            outputs = model(inputs)

            # Берем первый пример из батча
            sample_idx = 0

            # Предсказания
            strength_probs = torch.softmax(outputs["hand_strength"], dim=1)[sample_idx]
            predicted_strength = torch.argmax(strength_probs).item()

            category_probs = torch.sigmoid(outputs["category_probs"])[sample_idx]
            predicted_category = torch.argmax(category_probs).item()

            specific_probs = outputs["specific_hand"][sample_idx]

            # Истинные значения
            true_strength = targets["hand_strength"][sample_idx].item()
            true_category = torch.argmax(targets["category_probs"][sample_idx]).item()

            print(f"\nПример {i+1}:")
            print(f"  Истинная сила руки: {true_strength}")
            print(
                f"  Предсказанная сила: {predicted_strength} (уверенность: {strength_probs[predicted_strength]:.3f})"
            )
            print(f"  Истинная категория: {category_names[true_category]}")
            print(
                f"  Предсказанная категория: {category_names[predicted_category]} (вероятность: {category_probs[predicted_category]:.3f})"
            )

            # Топ-3 наиболее вероятных ранга
            top_ranks = torch.topk(specific_probs, 3)
            print(f"  Топ-3 наиболее вероятных ранга:")
            for j, (prob, rank_idx) in enumerate(
                zip(top_ranks.values, top_ranks.indices)
            ):
                print(f"    {j+1}. {rank_names[rank_idx]}: {prob:.3f}")


# ---------------------- 6. Утилиты ----------------------
def setup_directories():
    """Создание необходимых папок"""
    directories = ["models", "plots", "data", "results"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def find_data_files(data_dir="data"):
    """Поиск CSV файлов с данными"""
    if not os.path.exists(data_dir):
        data_dir = "."  # Текущая папка если нет папки data

    # Ищем CSV файлы с покерными данными
    patterns = ["parsed_*.csv", "*poker*.csv", "*PDOM*.csv", "*.csv"]

    data_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(data_dir, pattern))
        data_files.extend(files)

    # Убираем дубликаты и сортируем
    data_files = sorted(list(set(data_files)))
    return data_files


def save_categories_json():
    """Сохранение JSON файла с категориями"""
    categories_info = {
        "hand_categories": PokerHandAnalyzer.HAND_CATEGORIES,
        "all_categories": PokerHandAnalyzer.get_all_categories(),
        "category_mapping": PokerHandAnalyzer.get_category_mapping(),
        "rank_names": PokerHandAnalyzer.get_rank_names(),
        "strength_levels": {
            "0": "Очень слабая рука",
            "1": "Слабая рука",
            "2": "Ниже средней",
            "3": "Средняя рука",
            "4": "Выше средней",
            "5": "Хорошая рука",
            "6": "Сильная рука",
            "7": "Очень сильная",
            "8": "Премиум рука",
            "9": "Лучшие руки",
        },
        "description": {
            "categories": "Категории покерных рук для классификации",
            "strength_levels": "Уровни силы рук от 0 (слабейшие) до 9 (сильнейшие)",
            "rank_names": "Названия рангов карт от 2 до Ace",
        },
    }

    os.makedirs("results", exist_ok=True)
    with open("results/poker_categories.json", "w", encoding="utf-8") as f:
        json.dump(categories_info, f, indent=2, ensure_ascii=False)

    print("Сохранен файл с категориями: results/poker_categories.json")
    return categories_info


def choose_data_file():
    """Выбор файла с данными"""
    data_files = find_data_files()

    if not data_files:
        print("❌ CSV файлы с данными не найдены!")
        print("Поместите CSV файл в папку 'data' или в текущую директорию")
        return None

    if len(data_files) == 1:
        print(f"✅ Найден файл данных: {data_files[0]}")
        return data_files[0]

    # Подсчитываем общий размер всех файлов
    total_size_mb = 0
    file_stats = []

    for file in data_files:
        size_mb = os.path.getsize(file) / 1024 / 1024
        total_size_mb += size_mb
        file_stats.append((file, size_mb))

    print(
        f"📁 Найдено {len(data_files)} файлов с данными (общий размер: {total_size_mb:.1f} MB):"
    )

    # Если файлов больше 10, показываем процентное соотношение
    if len(data_files) > 10:
        print(f"📊 Топ-10 крупнейших файлов (по размеру):")
        sorted_files = sorted(file_stats, key=lambda x: x[1], reverse=True)[:10]
        for i, (file, size_mb) in enumerate(sorted_files):
            percentage = (size_mb / total_size_mb) * 100
            print(
                f"  {i+1}. {os.path.basename(file)} ({size_mb:.1f} MB, {percentage:.1f}%)"
            )

        print(f"\n📋 Показаны только топ-10 файлов из {len(data_files)}")
        print(f"💡 Рекомендуется выбрать опцию 0 для объединения всех файлов")
    else:
        # Показываем все файлы с процентами
        for i, (file, size_mb) in enumerate(file_stats):
            percentage = (size_mb / total_size_mb) * 100
            print(
                f"  {i+1}. {os.path.basename(file)} ({size_mb:.1f} MB, {percentage:.1f}%)"
            )

    print(f"  0. ⭐ ОБЪЕДИНИТЬ ВСЕ {len(data_files)} ФАЙЛОВ И ОБУЧИТЬ МОЩНУЮ МОДЕЛЬ")

    while True:
        try:
            if len(data_files) > 10:
                choice = input(
                    f"\nВыберите файл (0=объединить все, 1-10=топ файлы): "
                ).strip()
            else:
                choice = input(
                    f"\nВыберите файл (0-{len(data_files)}, 0=объединить все): "
                ).strip()

            if choice.lower() in ["q", "quit", "exit"]:
                return None

            choice_idx = int(choice)

            if choice_idx == 0:
                print(
                    f"✅ Выбрано: объединить все {len(data_files)} файлов и обучить мощную модель"
                )
                return "COMBINE_ALL"
            elif len(data_files) > 10:
                # Для большого количества файлов - выбираем из топ-10
                if 1 <= choice_idx <= 10:
                    sorted_files = sorted(file_stats, key=lambda x: x[1], reverse=True)
                    selected_file = sorted_files[choice_idx - 1][0]
                    print(f"✅ Выбран файл: {selected_file}")
                    return selected_file
                else:
                    print(f"❌ Неверный выбор. Введите число от 0 до 10")
            else:
                # Для небольшого количества файлов - выбираем из всех
                if 1 <= choice_idx <= len(data_files):
                    selected_file = data_files[choice_idx - 1]
                    print(f"✅ Выбран файл: {selected_file}")
                    return selected_file
                else:
                    print(f"❌ Неверный выбор. Введите число от 0 до {len(data_files)}")
        except ValueError:
            print("❌ Введите число или 'q' для выхода")
        except KeyboardInterrupt:
            print("\n👋 Выход...")
            return None


def combine_all_data_files():
    """Объединение всех файлов в один большой датасет"""
    data_files = find_data_files()

    if not data_files:
        print("❌ Не найдено файлов для объединения")
        return None

    print(f"\n🔗 === ОБЪЕДИНЕНИЕ {len(data_files)} ФАЙЛОВ ===")

    all_dataframes = []
    total_records = 0
    total_showdowns = 0

    for i, data_path in enumerate(data_files, 1):
        print(f"📄 Загружаем файл {i}/{len(data_files)}: {data_path}")

        try:
            df = pd.read_csv(data_path)

            # Добавляем информацию об источнике данных
            df["source_file"] = os.path.basename(data_path)
            df["file_index"] = i

            # Подсчет статистики
            showdowns_in_file = (
                (df["Showdown_1"].notna()) & (df["Showdown_2"].notna())
            ).sum()

            print(f"   📊 Размер: {len(df):,} строк, шоудаунов: {showdowns_in_file:,}")

            all_dataframes.append(df)
            total_records += len(df)
            total_showdowns += showdowns_in_file

        except Exception as e:
            print(f"   ❌ Ошибка загрузки файла {data_path}: {e}")
            continue

    if not all_dataframes:
        print("❌ Не удалось загрузить ни одного файла")
        return None

    # Объединяем все данные
    print(f"\n🔄 Объединяем данные...")
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)

    print(f"✅ Объединение завершено:")
    print(f"   📈 Общий размер: {total_records:,} строк")
    print(f"   🃏 Общее количество шоудаунов: {total_showdowns:,}")
    print(
        f"   📊 Процент записей с шоудауном: {total_showdowns/total_records*100:.1f}%"
    )
    print(f"   📁 Источников файлов: {len(data_files)}")

    # Сохраняем объединенный файл
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_filename = f"data/combined_poker_data_{timestamp}.csv"

    os.makedirs("data", exist_ok=True)
    combined_df.to_csv(combined_filename, index=False)
    print(f"💾 Объединенные данные сохранены: {combined_filename}")

    # Создаем сводку по файлам
    file_summary = {}
    for file_path in data_files:
        file_df = combined_df[combined_df["source_file"] == os.path.basename(file_path)]
        file_showdowns = (
            (file_df["Showdown_1"].notna()) & (file_df["Showdown_2"].notna())
        ).sum()

        file_summary[file_path] = {
            "records": len(file_df),
            "showdowns": int(file_showdowns),
            "showdown_percentage": (
                float(file_showdowns / len(file_df) * 100) if len(file_df) > 0 else 0
            ),
        }

    # Сохраняем сводку
    summary_path = f"results/data_combination_summary_{timestamp}.json"
    combination_summary = {
        "timestamp": datetime.now().isoformat(),
        "total_files": len(data_files),
        "total_records": int(total_records),
        "total_showdowns": int(total_showdowns),
        "combined_file": combined_filename,
        "source_files": file_summary,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(combination_summary, f, indent=2, ensure_ascii=False)

    print(f"📋 Сводка сохранена: {summary_path}")

    return combined_filename, combination_summary


def process_all_files():
    """Обработка всех файлов - НОВАЯ ВЕРСИЯ с объединением данных"""

    print(f"\n🚀 === МАССОВАЯ ОБРАБОТКА С ОБЪЕДИНЕНИЕМ ДАННЫХ ===")

    # Сначала объединяем все файлы
    result = combine_all_data_files()
    if result is None:
        print("❌ Не удалось объединить файлы")
        return

    combined_filename, combination_summary = result

    # Теперь обучаем модель на объединенных данных
    print(f"\n🎯 === ОБУЧЕНИЕ НА ОБЪЕДИНЕННЫХ ДАННЫХ ===")
    print(f"📁 Используем файл: {combined_filename}")
    print(f"📊 Общее количество шоудаунов: {combination_summary['total_showdowns']:,}")

    if combination_summary["total_showdowns"] < 50:
        print(
            "⚠️  Даже после объединения мало данных с шоудауном для качественного обучения!"
        )
        return

    results = {}
    all_plots = []

    # Обучаем обе модели на объединенных данных
    for model_type, include_cards in [("БЕЗ карт", False), ("С картами", True)]:
        print(f"\n{'='*80}")
        print(f"🎲 Обучение модели {model_type} игрока на объединенных данных...")
        print(f"{'='*80}")

        try:
            data_dict = prepare_hand_range_data(
                combined_filename, include_hole_cards=include_cards
            )

            if data_dict is not None:
                # Полноценное обучение с хорошими параметрами
                model, history = train_hand_range_model(
                    data_dict,
                    hidden_dim=128,  # Больше нейронов
                    num_layers=3,  # Больше слоев
                    epochs=25,  # Больше эпох
                    lr=0.001,
                )

                # Детальная оценка
                performance = evaluate_model_performance(
                    model, data_dict, include_hole_cards=include_cards
                )

                # Визуализация
                fig, plot_path = visualize_results(
                    model, data_dict, history, include_cards
                )
                all_plots.append(plot_path)
                plt.show()

                # Примеры предсказаний
                predict_hand_ranges(model, data_dict, sample_hands=5)

                # Сохранение модели
                model_path, best_path = save_best_model(
                    model, data_dict, performance, include_cards
                )

                model_key = "with_cards" if include_cards else "without_cards"
                results[model_key] = performance

                print(f"✅ Модель {model_type} обучена успешно!")
                print(
                    f"   🎯 Точность силы руки: {performance['strength_accuracy']:.3f}"
                )
                print(
                    f"   🏷️  Точность категории: {performance['category_accuracy']:.3f}"
                )
                print(f"   📊 MSE рангов: {performance['specific_mse']:.4f}")

            else:
                print(f"❌ Ошибка подготовки данных для модели {model_type}")

        except Exception as e:
            print(f"❌ Ошибка обучения модели {model_type}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Сравнение моделей
    if "with_cards" in results and "without_cards" in results:
        print(f"\n" + "=" * 80)
        print(f"🏆 === СРАВНЕНИЕ МОДЕЛЕЙ (на объединенных данных) ===")

        with_cards_acc = results["with_cards"]["strength_accuracy"]
        without_cards_acc = results["without_cards"]["strength_accuracy"]

        print(f"   🎯 Точность модели С картами: {with_cards_acc:.3f}")
        print(f"   🎲 Точность модели БЕЗ карт: {without_cards_acc:.3f}")
        print(f"   📈 Разница: {with_cards_acc - without_cards_acc:.3f}")

        if with_cards_acc > without_cards_acc:
            print(f"   ✅ Модель с картами игрока показывает лучшую производительность")
        else:
            print(
                f"   🎯 Модель без карт игрока показывает сопоставимую производительность"
            )

        # Финальный отчет
        final_report = {
            "timestamp": datetime.now().isoformat(),
            "combined_data_file": combined_filename,
            "data_combination_summary": combination_summary,
            "models": {
                "with_cards": {
                    "strength_accuracy": float(with_cards_acc),
                    "category_accuracy": float(
                        results["with_cards"]["category_accuracy"]
                    ),
                    "specific_mse": float(results["with_cards"]["specific_mse"]),
                },
                "without_cards": {
                    "strength_accuracy": float(without_cards_acc),
                    "category_accuracy": float(
                        results["without_cards"]["category_accuracy"]
                    ),
                    "specific_mse": float(results["without_cards"]["specific_mse"]),
                },
            },
            "plots": all_plots,
            "training_summary": {
                "total_source_files": combination_summary["total_files"],
                "total_records_used": combination_summary["total_records"],
                "total_showdowns_used": combination_summary["total_showdowns"],
                "models_trained": len(results),
            },
        }

        report_path = f"results/combined_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        print(f"   📋 Финальный отчет сохранен: {report_path}")

    print(f"\n🎉 === ОБУЧЕНИЕ НА ОБЪЕДИНЕННЫХ ДАННЫХ ЗАВЕРШЕНО ===")
    print(f"📊 Использовано данных:")
    print(f"   📁 Файлов: {combination_summary['total_files']}")
    print(f"   📈 Записей: {combination_summary['total_records']:,}")
    print(f"   🃏 Шоудаунов: {combination_summary['total_showdowns']:,}")
    print(f"📁 Результаты сохранены в папках models/, plots/, results/")

    return results


def save_best_model(model, data_dict, performance, include_hole_cards):
    """Сохранение лучшей модели с полной информацией"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = "with_cards" if include_hole_cards else "without_cards"

    model_name = f"hand_range_model_{model_type}"
    model_file = f"{model_name}_{timestamp}.pth"

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", model_file)

    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": data_dict["scaler"],
            "feature_columns": data_dict["feature_columns"],
            "input_dim": data_dict["input_dim"],
            "include_hole_cards": include_hole_cards,
            "performance": performance,
            "timestamp": timestamp,
            "categories": PokerHandAnalyzer.get_all_categories(),
            "category_mapping": PokerHandAnalyzer.get_category_mapping(),
            "rank_names": PokerHandAnalyzer.get_rank_names(),
        },
        model_path,
    )

    # Создаем ссылку на лучшую модель
    best_model_path = os.path.join("models", f"{model_name}_best.pth")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    # Копируем файл вместо создания жесткой ссылки (совместимость с разными ОС)
    import shutil

    shutil.copy2(model_path, best_model_path)

    print(f"💾 Модель сохранена:")
    print(f"   📁 {model_path}")
    print(f"   🔗 {best_model_path} (лучшая)")

    return model_path, best_model_path


# ---------------------- 7. Главная функция ----------------------
def main():
    """Главная функция для обучения и оценки моделей"""

    print("🎰 === Обучение моделей предсказания диапазонов рук в покере ===\n")

    # Создаем необходимые папки
    setup_directories()

    # Сохраняем JSON с категориями
    save_categories_json()

    # Выбираем файл данных
    data_choice = choose_data_file()
    if not data_choice:
        print("Выход из программы.")
        return

    # Проверяем, выбрана ли массовая обработка
    if data_choice == "COMBINE_ALL":
        process_all_files()
        return

    # Обычная обработка одного файла
    data_path = data_choice

    # Анализ файла данных
    print(f"\n📊 Анализ файла данных...")
    df_check = pd.read_csv(data_path)
    showdown_count = (
        (df_check["Showdown_1"].notna()) & (df_check["Showdown_2"].notna())
    ).sum()
    total_rows = len(df_check)
    print(f"   📈 Общий размер файла: {total_rows:,} строк")
    print(f"   🃏 Найдено {showdown_count:,} записей с открытыми картами")
    print(f"   📊 Процент записей с шоудауном: {showdown_count/total_rows*100:.1f}%")

    if showdown_count < 50:
        print(
            "⚠️  Внимание: мало записей с открытыми картами для качественного обучения!"
        )
    else:
        print("✅ Достаточно данных для обучения.")

    results = {}
    all_plots = []

    # 1. Модель С картами игрока
    print(f"\n🎯 1. Обучение модели С картами игрока...")
    try:
        data_with_cards = prepare_hand_range_data(data_path, include_hole_cards=True)

        if data_with_cards is not None:
            model_with_cards, history_with_cards = train_hand_range_model(
                data_with_cards,
                hidden_dim=128,
                num_layers=3,
                epochs=20,
                lr=0.001,
            )

            # Оценка модели
            results["with_cards"] = evaluate_model_performance(
                model_with_cards, data_with_cards, include_hole_cards=True
            )

            # Визуализация
            fig1, plot_path1 = visualize_results(
                model_with_cards, data_with_cards, history_with_cards, True
            )
            all_plots.append(plot_path1)
            plt.show()

            # Примеры предсказаний
            predict_hand_ranges(model_with_cards, data_with_cards, sample_hands=3)

            # Сохранение модели
            model_path, best_path = save_best_model(
                model_with_cards, data_with_cards, results["with_cards"], True
            )

        else:
            print("❌ Не удалось подготовить данные для модели с картами")

    except Exception as e:
        print(f"❌ Ошибка при обучении модели с картами: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n" + "=" * 60 + "\n")

    # 2. Модель БЕЗ карт игрока
    print(f"🎲 2. Обучение модели БЕЗ карт игрока...")
    print(
        f"   (та же цель - угадать реальные карты, но без информации о них в признаках)"
    )
    try:
        data_without_cards = prepare_hand_range_data(
            data_path, include_hole_cards=False
        )

        if data_without_cards is not None:
            model_without_cards, history_without_cards = train_hand_range_model(
                data_without_cards,
                hidden_dim=128,
                num_layers=3,
                epochs=20,
                lr=0.001,
            )

            # Оценка модели
            results["without_cards"] = evaluate_model_performance(
                model_without_cards, data_without_cards, include_hole_cards=False
            )

            # Визуализация
            fig2, plot_path2 = visualize_results(
                model_without_cards, data_without_cards, history_without_cards, False
            )
            all_plots.append(plot_path2)
            plt.show()

            # Примеры предсказаний
            predict_hand_ranges(model_without_cards, data_without_cards, sample_hands=3)

            # Сохранение модели
            model_path, best_path = save_best_model(
                model_without_cards, data_without_cards, results["without_cards"], False
            )

        else:
            print("❌ Не удалось подготовить данные для модели без карт")

    except Exception as e:
        print(f"❌ Ошибка при обучении модели без карт: {e}")
        import traceback

        traceback.print_exc()

    # 3. Сравнение моделей
    print(f"\n" + "=" * 60)
    print(f"🏆 === СРАВНЕНИЕ МОДЕЛЕЙ ===")

    if "with_cards" in results and "without_cards" in results:
        with_cards_acc = results["with_cards"]["strength_accuracy"]
        without_cards_acc = results["without_cards"]["strength_accuracy"]

        print(f"   🎯 Точность модели С картами: {with_cards_acc:.3f}")
        print(f"   🎲 Точность модели БЕЗ карт: {without_cards_acc:.3f}")
        print(f"   📈 Разница: {with_cards_acc - without_cards_acc:.3f}")

        if with_cards_acc > without_cards_acc:
            print(f"   ✅ Модель с картами игрока показывает лучшую производительность")
        else:
            print(
                f"   🎯 Модель без карт игрока показывает сопоставимую производительность"
            )

        # Сохраняем сравнительный отчет
        comparison_report = {
            "timestamp": datetime.now().isoformat(),
            "data_file": data_path,
            "total_records": total_rows,
            "showdown_records": showdown_count,
            "models": {
                "with_cards": {
                    "strength_accuracy": float(with_cards_acc),
                    "category_accuracy": float(
                        results["with_cards"]["category_accuracy"]
                    ),
                    "specific_mse": float(results["with_cards"]["specific_mse"]),
                },
                "without_cards": {
                    "strength_accuracy": float(without_cards_acc),
                    "category_accuracy": float(
                        results["without_cards"]["category_accuracy"]
                    ),
                    "specific_mse": float(results["without_cards"]["specific_mse"]),
                },
            },
            "plots": all_plots,
        }

        report_path = (
            f"results/comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(comparison_report, f, indent=2, ensure_ascii=False)

        print(f"   📋 Отчет сохранен: {report_path}")

    print(f"\n🎉 Обучение завершено!")
    print(f"📁 Результаты сохранены в папках:")
    print(f"   🤖 models/     - обученные модели")
    print(f"   📊 plots/      - графики обучения")
    print(f"   📋 results/    - отчеты и конфигурации")
    print(f"   🃏 results/poker_categories.json - категории рук")


if __name__ == "__main__":
    main()
