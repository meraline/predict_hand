#!/usr/bin/env python3
"""
Отладочный скрипт для двухэтапной системы
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hand_range_prediction import *
from your_fixed_functions import *  # Здесь должны быть исправленные функции

def debug_data_preparation():
    """Отладка подготовки данных"""
    print("🔍 === ОТЛАДКА ПОДГОТОВКИ ДАННЫХ ===")
    
    # Выбираем файл
    data_files = find_data_files()
    if not data_files:
        print("❌ Нет файлов данных!")
        return
    
    data_file = data_files[0]  # Берем первый файл
    print(f"📁 Используем файл: {data_file}")
    
    # Проверяем маппинг
    hm3_mapping_path = 'hm3_10pct_simple_mapping_20250612_072153.json'
    if not os.path.exists(hm3_mapping_path):
        print(f"❌ Файл маппинга не найден: {hm3_mapping_path}")
        return
    
    # Загружаем данные
    df = pd.read_csv(data_file)
    print(f"\n📊 Общая статистика файла:")
    print(f"   Всего записей: {len(df)}")
    print(f"   Колонки: {list(df.columns)[:10]}...")
    
    # Проверяем шоудауны
    showdown_mask = (df['Showdown_1'].notna()) & (df['Showdown_2'].notna())
    print(f"   Записей с шоудауном: {showdown_mask.sum()} ({showdown_mask.sum()/len(df)*100:.1f}%)")
    
    # Проверяем уникальные руки
    if 'HandID' in df.columns:
        print(f"   Уникальных рук: {df['HandID'].nunique()}")
    elif 'Hand' in df.columns:
        print(f"   Уникальных рук: {df['Hand'].nunique()}")
    
    # Подготавливаем данные
    print(f"\n🔄 Подготовка данных...")
    data_dict = prepare_two_stage_data(data_file, hm3_mapping_path)
    
    if data_dict is None:
        print("❌ Ошибка подготовки данных!")
        return
    
    print(f"\n✅ Данные подготовлены:")
    print(f"   Train sequences: {len(data_dict['train_sequences'])}")
    print(f"   Val sequences: {len(data_dict['val_sequences'])}")
    print(f"   Test sequences: {len(data_dict['test_sequences'])}")
    print(f"   Feature columns: {len(data_dict['feature_columns'])}")
    
    # Проверяем распределение целевых переменных
    print(f"\n📊 Распределение целевых групп (train):")
    train_targets = data_dict['final_targets']['train']
    unique, counts = np.unique(train_targets, return_counts=True)
    for group, count in zip(unique, counts):
        print(f"   Группа {group}: {count} ({count/len(train_targets)*100:.1f}%)")
    
    return data_dict


def debug_preflop_model(data_dict):
    """Отладка префлоп модели"""
    print("\n🔍 === ОТЛАДКА ПРЕФЛОП МОДЕЛИ ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Устройство: {device}")
    
    # Создаем простую тестовую модель
    print(f"\n🧠 Создание упрощенной префлоп модели...")
    
    class SimplePreflopModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 169)  # 169 возможных стартовых рук
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            # Берем среднее по последовательности
            x = x.mean(dim=1)
            x = torch.relu(self.fc1(x))
            x = self.sigmoid(self.fc2(x))
            return x
    
    model = SimplePreflopModel(len(data_dict['feature_columns'])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Создаем простой датасет
    train_sequences = data_dict['train_sequences']
    
    print(f"📊 Обучение простой модели...")
    losses = []
    
    for epoch in range(10):
        epoch_losses = []
        
        for i, seq in enumerate(train_sequences[:50]):  # Берем только 50 примеров
            # Подготавливаем данные
            available_cols = [col for col in data_dict['feature_columns'] if col in seq.columns]
            if not available_cols:
                continue
                
            features = seq[available_cols].fillna(0).values
            if len(features) == 0:
                continue
                
            # Нормализация
            features = data_dict['scaler'].transform(features)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Создаем случайную цель для теста
            target = torch.rand(1, 169).to(device)
            target = target / target.sum(dim=1, keepdim=True)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(features_tensor)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        losses.append(avg_loss)
        print(f"   Эпоха {epoch+1}/10: Loss = {avg_loss:.4f}")
    
    # Визуализация
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title('Простая модель: Loss по эпохам')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    print(f"\n✅ Простая модель обучается корректно!")
    return model


def debug_final_model(data_dict):
    """Отладка финальной модели"""
    print("\n🔍 === ОТЛАДКА ФИНАЛЬНОЙ МОДЕЛИ ===")
    
    # Проверяем данные для финальной модели
    train_targets = data_dict['final_targets']['train']
    
    print(f"📊 Статистика целевых переменных:")
    print(f"   Количество: {len(train_targets)}")
    print(f"   Уникальные значения: {np.unique(train_targets)}")
    print(f"   Тип данных: {type(train_targets[0]) if train_targets else 'N/A'}")
    
    # Создаем простой датасет
    from torch.utils.data import TensorDataset
    
    # Берем первые 100 последовательностей
    sample_sequences = data_dict['train_sequences'][:100]
    sample_targets = train_targets[:100]
    
    # Создаем простые признаки
    X = []
    y = []
    
    for i, seq in enumerate(sample_sequences):
        if i >= len(sample_targets):
            break
            
        # Берем последнюю строку последовательности
        last_row_features = []
        for col in data_dict['feature_columns']:
            if col in seq.columns:
                val = seq[col].iloc[-1] if len(seq) > 0 else 0
                last_row_features.append(val if pd.notna(val) else 0)
            else:
                last_row_features.append(0)
        
        X.append(last_row_features)
        y.append(sample_targets[i])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    print(f"\n📊 Подготовленные данные:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   y уникальные: {np.unique(y)}")
    
    # Нормализация
    X = data_dict['scaler'].transform(X)
    
    # Создаем простую модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_classes)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    model = SimpleClassifier(X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Преобразуем в тензоры
    X_tensor = torch.tensor(X).to(device)
    y_tensor = torch.tensor(y).to(device)
    
    # Обучение
    print(f"\n🧠 Обучение простого классификатора...")
    losses = []
    accuracies = []
    
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        loss.backward()
        optimizer.step()
        
        # Точность
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).float().mean().item()
        
        losses.append(loss.item())
        accuracies.append(accuracy)
        
        if epoch % 5 == 0:
            print(f"   Эпоха {epoch+1}/20: Loss = {loss.item():.4f}, Accuracy = {accuracy:.3f}")
    
    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(losses)
    ax1.set_title('Loss по эпохам')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(accuracies)
    ax2.set_title('Точность по эпохам')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n✅ Простой классификатор работает!")
    print(f"   Финальная точность: {accuracies[-1]:.3f}")
    
    return model


def main_debug():
    """Основная функция отладки"""
    print("🔍 === ОТЛАДКА ДВУХЭТАПНОЙ СИСТЕМЫ ===\n")
    
    # Создаем папки
    setup_directories()
    
    # 1. Отладка подготовки данных
    data_dict = debug_data_preparation()
    if data_dict is None:
        return
    
    # 2. Отладка префлоп модели
    preflop_model = debug_preflop_model(data_dict)
    
    # 3. Отладка финальной модели
    final_model = debug_final_model(data_dict)
    
    print("\n✅ Отладка завершена!")
    print("📊 Рекомендации:")
    print("   1. Проверьте качество данных (мало шоудаунов)")
    print("   2. Увеличьте количество данных (объедините файлы)")
    print("   3. Упростите архитектуру моделей")
    print("   4. Используйте предобученные эмбеддинги для карт")


if __name__ == "__main__":
    main_debug()