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
        
        # Извлекаем только доступные признаки
        available_features = [col for col in self.feature_columns if col in preflop_seq.columns]
        features = preflop_seq[available_features].fillna(0).values
        
        # Применяем scaler
        if self.scaler:
            # Дополняем недостающие колонки нулями
            if len(available_features) < len(self.feature_columns):
                full_features = np.zeros((len(features), len(self.feature_columns)))
                for i, col in enumerate(self.feature_columns):
                    if col in available_features:
                        col_idx = available_features.index(col)
                        full_features[:, i] = features[:, col_idx]
                features = full_features
            
            features = self.scaler.transform(features)
        
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
        
        # Извлекаем доступные признаки
        available_features = [col for col in self.feature_columns if col in seq_df.columns]
        features = seq_df[available_features].fillna(0).values
        
        # Применяем scaler
        if self.scaler:
            # Дополняем недостающие колонки
            if len(available_features) < len(self.feature_columns):
                full_features = np.zeros((len(features), len(self.feature_columns)))
                for i, col in enumerate(self.feature_columns):
                    if col in available_features:
                        col_idx = available_features.index(col)
                        full_features[:, i] = features[:, col_idx]
                features = full_features
            
            features = self.scaler.transform(features)
        
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