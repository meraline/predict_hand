#!/usr/bin/env python3
"""
Тестовый скрипт для детального анализа всех 73 классов HM3
с учетом их упорядоченности по силе
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, OrderedDict
import json
import os
import sys
from datetime import datetime

# Импортируем необходимые классы
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class HM3OrdinalAnalyzer:
    """
    Анализатор распределения 73 классов HM3 с учетом упорядоченности
    """
    
    def __init__(self):
        # Полный упорядоченный список от сильнейших к слабейшим
        self.strength_mapping = {
            # Класс 4: МОНСТРЫ (индексы 0-13)
            "TwoCardStraightFlush": 4,
            "OneCardStraightFlush": 4, 
            "FourOfAKindWithPocketPair": 4,
            "FourOfAKindWithoutPocketPair": 4,
            "FourOfAKindOnBoard": 4,
            "OneCardFullHouseTopPair": 4,
            "OneCardFullHouseTripsOnBoard": 4,
            "FullHouseWithPocketPairNoTripsOnBoard": 4,
            "TwoCardFullHouseWithoutPocketPair": 4,
            "FullHouseWithPocketPairTripsOnBoard": 4,
            "ThreeFlushBoardNutFlush": 4,
            "FourFlushBoardNutFlush": 4,
            "TwoCardNutStraight": 4,
            "OneCardNutStraight": 4,
            
            # Класс 3: СИЛЬНЫЕ (индексы 14-23)
            "ThreeFlushBoardHighFlush": 3,
            "FourFlushBoardHighFlush": 3,
            "TwoCardStraight": 3,
            "OneCardStraight": 3,
            "HighSet": 3,
            "SecondSet": 3,
            "LowSet": 3,
            "HighTripsHighKicker": 3,
            "HighTripsLowKicker": 3,
            "TopTwoPair": 3,
            
            # Класс 2: СРЕДНИЕ (индексы 24-42)
            "ThreeFlushBoardLowFlush": 2,
            "FourFlushBoardLowFlush": 2,
            "SecondTripsHighKicker": 2,
            "SecondTripsLowKicker": 2,
            "LowTripsHighKicker": 2,
            "LowTripsLowKicker": 2,
            "TripsOnBoard": 2,
            "TopPairPlusPair": 2,
            "NonTopTwoPair": 2,
            "PocketPairOverPairPlusLowerPairedBoard": 2,
            "PocketPairPlusHigherPairedBoard": 2,
            "PocketPairPlusLowerPairedBoard": 2,
            "TopPairPlusPairedBoard": 2,
            "SecondPairPlusPairedBoard": 2,
            "LowPairPlusPairedBoard": 2,
            "TwoPairsOnBoard": 2,
            "OverPair": 2,
            "TopPairTopKicker": 2,
            "TopPairGoodKicker": 2,
            
            # Класс 1: СЛАБЫЕ (индексы 43-56)
            "TopPairWeakKicker": 1,
            "SecondPocketPair": 1,
            "SecondPairAceKicker": 1,
            "SecondPairNonAceKicker": 1,
            "LowPocketPair": 1,
            "BottomPairAceKicker": 1,
            "BottomPairNonAceKicker": 1,
            "PairedBoardNoOvercards": 1,
            "PairedBoardOneOvercard": 1,
            "PairedBoardTwoOvercards": 1,
            "TwoCardNutFlushDraw": 1,
            "TwoCardHighFlushDraw": 1,
            "OneCardNutFlushDraw": 1,
            "OneCardHighFlushDraw": 1,
            
            # Класс 0: МУСОР/ДРО (индексы 57-72)
            "TwoCardLowFlushDraw": 0,
            "OneCardLowFlushDraw": 0,
            "TwoCardBackdoorNutFlushDraw": 0,
            "TwoCardBackdoorFlushDraw": 0,
            "OneCardBackDoorNutFlushDraw": 0,
            "OneCardBackDoorFlushDraw": 0,
            "TwoCardDoubleGutShotStraightDraw": 0,
            "TwoCardOpenEndedStraightDraw": 0,
            "OneCardOpenEndedStraightDraw": 0,
            "TwoCardGutshotStraightDraw": 0,
            "OneCardGutshotStraightDraw": 0,
            "TwoCardBackdoorStraightDraw": 0,
            "OneCardBackdoorStraightDraw": 0,
            "AceOnBoard": 0,
            "KingOnBoard": 0,
            "Other": 0
        }
        
        # Создаем упорядоченный список
        self.ordered_categories = list(self.strength_mapping.keys())
        self.category_to_index = {cat: i for i, cat in enumerate(self.ordered_categories)}
        
        # Названия классов силы
        self.strength_names = {
            0: "Мусор/Дро",
            1: "Слабые",
            2: "Средние",
            3: "Сильные",
            4: "Монстры"
        }
    
    def analyze_data(self, file_path):
        """
        Полный анализ данных с учетом упорядоченности
        """
        print(f"🔍 === АНАЛИЗ HM3 РАСПРЕДЕЛЕНИЯ С УЧЕТОМ УПОРЯДОЧЕННОСТИ ===\n")
        print(f"📁 Файл: {file_path}")
        
        # Загрузка данных
        print(f"⏳ Загрузка данных...")
        df = pd.read_csv(file_path)
        
        # Фильтрация записей с шоудауном
        mask = (df['Showdown_1'].notna()) & (df['Showdown_2'].notna())
        df_showdown = df[mask].copy()
        
        print(f"✅ Загружено {len(df)} записей")
        print(f"🃏 Записей с шоудауном: {len(df_showdown)} ({len(df_showdown)/len(df)*100:.1f}%)\n")
        
        if len(df_showdown) == 0:
            print("❌ Нет записей с шоудауном!")
            return None
        
        # Добавляем HM3 классификацию
        print(f"🎯 Применение HM3 классификации...")
        from hand_range_prediction import add_hand_evaluation_to_dataframe
        df_analyzed = add_hand_evaluation_to_dataframe(df_showdown)
        
        # Подсчет распределения
        hand_type_counts = df_analyzed['hand_type_hm3'].value_counts()
        total_hands = len(df_analyzed)
        
        # Детальный анализ
        analysis_results = self._detailed_analysis(hand_type_counts, total_hands)
        
        # Визуализация основных результатов
        self._create_visualizations(analysis_results, hand_type_counts)
        
        # Предложение группировок
        grouping_proposals = self._propose_groupings(analysis_results, hand_type_counts)
        
        # Используем новый метод визуализации
        if '10percent_quantiles' in grouping_proposals:
            print(f"\n📊 Создание улучшенной визуализации 10% группировки...")
            self.visualize_10percent_grouping(
                grouping_proposals['10percent_quantiles'], 
                analysis_results['ordered_stats']
            )
        
        # Сохранение результатов
        self._save_results(analysis_results, grouping_proposals)
        
        return analysis_results, grouping_proposals
    
    def _detailed_analysis(self, hand_type_counts, total_hands):
        """
        Детальный анализ распределения
        """
        print(f"\n📊 === ДЕТАЛЬНЫЙ АНАЛИЗ РАСПРЕДЕЛЕНИЯ ===")
        print(f"Всего уникальных типов рук: {len(hand_type_counts)}")
        print(f"Всего рук проанализировано: {total_hands:,}\n")
        
        # Создаем упорядоченную статистику
        ordered_stats = []
        missing_categories = []
        
        for i, category in enumerate(self.ordered_categories):
            if category in hand_type_counts.index:
                count = hand_type_counts[category]
                percentage = count / total_hands * 100
                strength_class = self.strength_mapping[category]
                
                ordered_stats.append({
                    'index': i,
                    'category': category,
                    'count': count,
                    'percentage': percentage,
                    'strength_class': strength_class,
                    'strength_name': self.strength_names[strength_class],
                    'cumulative_percentage': 0  # Заполним позже
                })
            else:
                missing_categories.append(category)
        
        # Вычисляем кумулятивные проценты
        cumsum = 0
        for stat in ordered_stats:
            cumsum += stat['percentage']
            stat['cumulative_percentage'] = cumsum
        
        # Анализ по классам силы
        strength_distribution = defaultdict(lambda: {'count': 0, 'types': 0, 'categories': []})
        
        for stat in ordered_stats:
            sc = stat['strength_class']
            strength_distribution[sc]['count'] += stat['count']
            strength_distribution[sc]['types'] += 1
            strength_distribution[sc]['categories'].append(stat['category'])
        
        print("📊 Распределение по классам силы:")
        for strength in sorted(strength_distribution.keys(), reverse=True):
            info = strength_distribution[strength]
            percentage = info['count'] / total_hands * 100
            print(f"\n💪 Класс {strength} - {self.strength_names[strength]}:")
            print(f"   Всего рук: {info['count']:,} ({percentage:.1f}%)")
            print(f"   Типов рук: {info['types']}")
            
            # Топ-5 в классе
            class_categories = sorted(
                [(cat, hand_type_counts[cat]) for cat in info['categories'] if cat in hand_type_counts.index],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            print(f"   Топ-5 типов:")
            for cat, cnt in class_categories:
                print(f"      {cat}: {cnt} ({cnt/total_hands*100:.2f}%)")
        
        # Анализ проблемных зон
        problems = self._analyze_problems(ordered_stats, missing_categories)
        
        # Анализ баланса
        balance_metrics = self._analyze_balance(ordered_stats)
        
        return {
            'ordered_stats': ordered_stats,
            'strength_distribution': dict(strength_distribution),
            'missing_categories': missing_categories,
            'problems': problems,
            'balance_metrics': balance_metrics,
            'total_hands': total_hands
        }
    
    def _analyze_problems(self, ordered_stats, missing_categories):
        """
        Анализ проблемных зон в распределении
        """
        print(f"\n🚨 === АНАЛИЗ ПРОБЛЕМНЫХ ЗОН ===")
        
        problems = {
            'missing': missing_categories,
            'rare': [],      # < 10 примеров
            'scarce': [],    # < 50 примеров
            'sparse': [],    # < 0.1%
            'gaps': [],      # Промежутки в упорядоченности
            'imbalanced_regions': []
        }
        
        # Анализ редких категорий
        prev_index = -1
        for stat in ordered_stats:
            # Редкие по количеству
            if stat['count'] < 10:
                problems['rare'].append((stat['category'], stat['count']))
            elif stat['count'] < 50:
                problems['scarce'].append((stat['category'], stat['count']))
            
            # Редкие по проценту
            if stat['percentage'] < 0.1:
                problems['sparse'].append((stat['category'], stat['percentage']))
            
            # Проверяем промежутки
            if stat['index'] - prev_index > 1:
                gap_size = stat['index'] - prev_index - 1
                problems['gaps'].append({
                    'start': prev_index,
                    'end': stat['index'],
                    'size': gap_size,
                    'missing': self.ordered_categories[prev_index+1:stat['index']]
                })
            
            prev_index = stat['index']
        
        # Анализ дисбаланса по регионам
        # Делим на квинтили по индексу
        quintile_size = len(self.ordered_categories) // 5
        for q in range(5):
            start_idx = q * quintile_size
            end_idx = (q + 1) * quintile_size if q < 4 else len(self.ordered_categories)
            
            quintile_stats = [s for s in ordered_stats if start_idx <= s['index'] < end_idx]
            if quintile_stats:
                quintile_count = sum(s['count'] for s in quintile_stats)
                quintile_pct = sum(s['percentage'] for s in quintile_stats)
                
                problems['imbalanced_regions'].append({
                    'quintile': q + 1,
                    'range': f"{self.ordered_categories[start_idx]} - {self.ordered_categories[end_idx-1]}",
                    'count': quintile_count,
                    'percentage': quintile_pct,
                    'expected_pct': 20.0
                })
        
        # Вывод проблем
        print(f"❌ Отсутствующие категории: {len(problems['missing'])}")
        if problems['missing']:
            print(f"   {problems['missing'][:5]}..." if len(problems['missing']) > 5 else f"   {problems['missing']}")
        
        print(f"\n⚠️ Редкие категории:")
        print(f"   < 10 примеров: {len(problems['rare'])} категорий")
        print(f"   < 50 примеров: {len(problems['scarce'])} категорий")
        print(f"   < 0.1%: {len(problems['sparse'])} категорий")
        
        print(f"\n🕳️ Промежутки в данных: {len(problems['gaps'])}")
        for gap in problems['gaps'][:3]:  # Показываем первые 3
            print(f"   Промежуток {gap['start']}-{gap['end']}: {gap['size']} категорий отсутствует")
        
        print(f"\n📊 Дисбаланс по квинтилям силы:")
        for region in problems['imbalanced_regions']:
            deviation = region['percentage'] - region['expected_pct']
            print(f"   Квинтиль {region['quintile']}: {region['percentage']:.1f}% (отклонение: {deviation:+.1f}%)")
        
        return problems
    
    def _analyze_balance(self, ordered_stats):
        """
        Анализ метрик баланса
        """
        counts = [s['count'] for s in ordered_stats]
        
        # Основные метрики
        metrics = {
            'total_categories': len(self.ordered_categories),
            'present_categories': len(ordered_stats),
            'coverage': len(ordered_stats) / len(self.ordered_categories) * 100,
            'min_count': min(counts),
            'max_count': max(counts),
            'mean_count': np.mean(counts),
            'median_count': np.median(counts),
            'std_count': np.std(counts),
            'cv': np.std(counts) / np.mean(counts),  # Коэффициент вариации
            'imbalance_ratio': max(counts) / min(counts),
            'gini_coefficient': self._calculate_gini(counts),
            'entropy': self._calculate_entropy([s['percentage']/100 for s in ordered_stats])
        }
        
        # Анализ монотонности (в идеале сильные руки должны быть реже)
        strength_counts = defaultdict(list)
        for stat in ordered_stats:
            strength_counts[stat['strength_class']].append(stat['count'])
        
        avg_by_strength = {
            strength: np.mean(counts) 
            for strength, counts in strength_counts.items()
        }
        
        # Проверяем, убывает ли частота с увеличением силы
        is_monotonic = all(
            avg_by_strength.get(i, 0) >= avg_by_strength.get(i+1, 0) 
            for i in range(4)
        )
        
        metrics['natural_monotonicity'] = is_monotonic
        metrics['avg_by_strength'] = avg_by_strength
        
        print(f"\n📈 === МЕТРИКИ БАЛАНСА ===")
        print(f"Покрытие категорий: {metrics['coverage']:.1f}%")
        print(f"Соотношение макс/мин: {metrics['imbalance_ratio']:.1f}:1")
        print(f"Коэффициент вариации: {metrics['cv']:.2f}")
        print(f"Коэффициент Джини: {metrics['gini_coefficient']:.3f}")
        print(f"Энтропия: {metrics['entropy']:.3f}")
        print(f"Естественная монотонность: {'✅ Да' if is_monotonic else '❌ Нет'}")
        
        return metrics
    
    def _calculate_gini(self, values):
        """Расчет коэффициента Джини"""
        sorted_values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((i + 1) * sorted_values[i] for i in range(n))) / (n * cumsum[-1]) - (n + 1) / n
    
    def _calculate_entropy(self, probabilities):
        """Расчет энтропии"""
        return -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
    
    def _propose_groupings(self, analysis_results, hand_type_counts):
        """
        Предлагает различные стратегии группировки
        """
        print(f"\n🎯 === ПРЕДЛОЖЕНИЯ ПО ГРУППИРОВКЕ ===")
        
        proposals = {}
        ordered_stats = analysis_results['ordered_stats']
        total_hands = analysis_results['total_hands']
        
        # Стратегия 1: Равномерная по количеству
        proposals['uniform_count'] = self._create_uniform_count_groups(ordered_stats, target_groups=15)
        
        # Стратегия 2: Равномерная по квантилям силы
        proposals['uniform_strength'] = self._create_uniform_strength_groups(ordered_stats, target_groups=15)
        
        # Стратегия 3: Адаптивная (учитывает редкие классы)
        proposals['adaptive'] = self._create_adaptive_groups(ordered_stats, analysis_results['problems'])
        
        # Стратегия 4: Семантическая (по типам комбинаций)
        proposals['semantic'] = self._create_semantic_groups()
        
        # НОВАЯ Стратегия 5: По 10% квантилям
        proposals['10percent_quantiles'] = self._create_10percent_groups(ordered_stats)
        
        # Оценка качества каждой стратегии
        print(f"\n📊 Сравнение стратегий группировки:")
        
        for strategy_name, groups in proposals.items():
            print(f"\n📌 Стратегия: {strategy_name}")
            quality = self._evaluate_grouping_quality(groups, ordered_stats)
            
            print(f"   Групп: {quality['num_groups']}")
            print(f"   Баланс размеров (CV): {quality['size_cv']:.2f}")
            print(f"   Сохранение порядка: {'✅' if quality['preserves_order'] else '❌'}")
            print(f"   Покрытие данных: {quality['coverage']:.1f}%")
            print(f"   Мин/Макс размер группы: {quality['min_size']}/{quality['max_size']}")
        
        # Рекомендация
        best_strategy = self._recommend_best_strategy(proposals, ordered_stats)
        print(f"\n🏆 Рекомендуемая стратегия: {best_strategy}")
        
        return proposals
    
    
    # 2. Новая стратегия группировки по 10% квантилям
    def _create_10percent_groups(self, ordered_stats):
        """
        Создает группы по 10% квантилям данных с сохранением упорядоченности
        Каждая группа содержит примерно 10% от общего количества рук
        """
        total_count = sum(s['count'] for s in ordered_stats)
        target_per_group = total_count * 0.1  # 10% от общего количества
        
        groups = []
        current_group = {
            'categories': [], 
            'total_count': 0, 
            'indices': [],
            'percentage': 0.0,
            'strength_classes': set()
        }
        
        cumulative_count = 0
        group_number = 1
        
        for stat in ordered_stats:
            # Проверяем, нужно ли начать новую группу
            if current_group['total_count'] > 0 and cumulative_count >= target_per_group * group_number:
                # Финализируем текущую группу
                current_group['percentage'] = current_group['total_count'] / total_count * 100
                groups.append(current_group)
                
                # Начинаем новую группу
                group_number += 1
                current_group = {
                    'categories': [], 
                    'total_count': 0, 
                    'indices': [],
                    'percentage': 0.0,
                    'strength_classes': set()
                }
            
            # Добавляем категорию в текущую группу
            current_group['categories'].append(stat['category'])
            current_group['total_count'] += stat['count']
            current_group['indices'].append(stat['index'])
            current_group['strength_classes'].add(stat['strength_class'])
            
            cumulative_count += stat['count']
        
        # Добавляем последнюю группу
        if current_group['categories']:
            current_group['percentage'] = current_group['total_count'] / total_count * 100
            groups.append(current_group)
        
        # Анализ качества группировки
        print(f"\n📊 10% группировка создана:")
        print(f"   Количество групп: {len(groups)}")
        
        for i, group in enumerate(groups):
            strength_desc = ", ".join([f"Класс {s}" for s in sorted(group['strength_classes'])])
            print(f"\n   Группа {i+1} ({group['percentage']:.1f}%):")
            print(f"      Категорий: {len(group['categories'])}")
            print(f"      Примеров: {group['total_count']:,}")
            print(f"      Индексы: {min(group['indices'])}-{max(group['indices'])}")
            print(f"      Классы силы: {strength_desc}")
            
            # Показываем первые 3 категории
            sample_cats = group['categories'][:3]
            if len(group['categories']) > 3:
                print(f"      Категории: {', '.join(sample_cats)}...")
            else:
                print(f"      Категории: {', '.join(sample_cats)}")
        
        return groups

    
    def _create_uniform_count_groups(self, ordered_stats, target_groups=15):
        """Создает группы с примерно одинаковым количеством примеров"""
        total_count = sum(s['count'] for s in ordered_stats)
        target_per_group = total_count / target_groups
        
        groups = []
        current_group = {'categories': [], 'total_count': 0, 'indices': []}
        
        for stat in ordered_stats:
            if current_group['total_count'] > 0 and \
               current_group['total_count'] + stat['count'] > target_per_group * 1.2:
                groups.append(current_group)
                current_group = {'categories': [], 'total_count': 0, 'indices': []}
            
            current_group['categories'].append(stat['category'])
            current_group['total_count'] += stat['count']
            current_group['indices'].append(stat['index'])
        
        if current_group['categories']:
            groups.append(current_group)
        
        return groups
    
    def _create_uniform_strength_groups(self, ordered_stats, target_groups=15):
        """Создает группы равномерно по диапазону силы"""
        indices_per_group = len(self.ordered_categories) / target_groups
        
        groups = []
        current_group_id = 0
        current_group = {'categories': [], 'total_count': 0, 'indices': []}
        
        for stat in ordered_stats:
            expected_group = int(stat['index'] / indices_per_group)
            
            if expected_group > current_group_id and current_group['categories']:
                groups.append(current_group)
                current_group = {'categories': [], 'total_count': 0, 'indices': []}
                current_group_id = expected_group
            
            current_group['categories'].append(stat['category'])
            current_group['total_count'] += stat['count']
            current_group['indices'].append(stat['index'])
        
        if current_group['categories']:
            groups.append(current_group)
        
        return groups
    
    def _create_adaptive_groups(self, ordered_stats, problems):
        """Адаптивная группировка с учетом проблемных зон"""
        groups = []
        current_group = {'categories': [], 'total_count': 0, 'indices': []}
        
        # Определяем пороги для объединения
        min_group_size = 100  # Минимальный размер группы
        rare_threshold = 50   # Порог для редких категорий
        
        for stat in ordered_stats:
            # Если текущая категория редкая, всегда добавляем в текущую группу
            if stat['count'] < rare_threshold:
                current_group['categories'].append(stat['category'])
                current_group['total_count'] += stat['count']
                current_group['indices'].append(stat['index'])
            else:
                # Если текущая группа достаточно большая, начинаем новую
                if current_group['total_count'] >= min_group_size:
                    if current_group['categories']:
                        groups.append(current_group)
                    current_group = {'categories': [], 'total_count': 0, 'indices': []}
                
                current_group['categories'].append(stat['category'])
                current_group['total_count'] += stat['count']
                current_group['indices'].append(stat['index'])
        
        if current_group['categories']:
            groups.append(current_group)
        
        return groups
    
    def _create_semantic_groups(self):
        """Семантическая группировка по типам комбинаций"""
        semantic_groups = [
            {
                'name': 'StraightFlushes',
                'categories': [c for c in self.ordered_categories if 'StraightFlush' in c],
                'indices': []
            },
            {
                'name': 'Quads',
                'categories': [c for c in self.ordered_categories if 'FourOfAKind' in c],
                'indices': []
            },
            {
                'name': 'FullHouses',
                'categories': [c for c in self.ordered_categories if 'FullHouse' in c],
                'indices': []
            },
            {
                'name': 'Flushes',
                'categories': [c for c in self.ordered_categories if 'Flush' in c and 'Draw' not in c and 'Straight' not in c],
                'indices': []
            },
            {
                'name': 'Straights',
                'categories': [c for c in self.ordered_categories if 'Straight' in c and 'Flush' not in c and 'Draw' not in c],
                'indices': []
            },
            {
                'name': 'Sets',
                'categories': [c for c in self.ordered_categories if 'Set' in c],
                'indices': []
            },
            {
                'name': 'Trips',
                'categories': [c for c in self.ordered_categories if 'Trips' in c],
                'indices': []
            },
            {
                'name': 'TwoPairs',
                'categories': [c for c in self.ordered_categories if 'TwoPair' in c or 'PairPlus' in c],
                'indices': []
            },
            {
                'name': 'TopPairs',
                'categories': [c for c in self.ordered_categories if 'TopPair' in c and 'Plus' not in c],
                'indices': []
            },
            {
                'name': 'MiddlePairs',
                'categories': [c for c in self.ordered_categories if any(x in c for x in ['SecondPair', 'OverPair', 'PocketPair'])],
                'indices': []
            },
            {
                'name': 'WeakPairs',
                'categories': [c for c in self.ordered_categories if any(x in c for x in ['BottomPair', 'LowPair', 'WeakKicker'])],
                'indices': []
            },
            {
                'name': 'StrongDraws',
                'categories': [c for c in self.ordered_categories if 'Draw' in c and any(x in c for x in ['Nut', 'High', 'OpenEnded'])],
                'indices': []
            },
            {
                'name': 'WeakDraws',
                'categories': [c for c in self.ordered_categories if 'Draw' in c and any(x in c for x in ['Low', 'Gutshot', 'Backdoor'])],
                'indices': []
            },
            {
                'name': 'HighCards',
                'categories': [c for c in self.ordered_categories if any(x in c for x in ['AceOnBoard', 'KingOnBoard', 'Other'])],
                'indices': []
            }
        ]
        
        # Заполняем индексы
        for group in semantic_groups:
            group['indices'] = [self.category_to_index[cat] for cat in group['categories'] if cat in self.category_to_index]
            group['total_count'] = 0  # Будет заполнено позже
        
        # Удаляем пустые группы
        semantic_groups = [g for g in semantic_groups if g['categories']]
        
        return semantic_groups
    
    def _evaluate_grouping_quality(self, groups, ordered_stats):
        """Оценка качества группировки"""
        # Создаем маппинг категория -> count
        cat_counts = {s['category']: s['count'] for s in ordered_stats}
        
        # Заполняем total_count для групп
        for group in groups:
            if 'total_count' not in group or group['total_count'] == 0:
                group['total_count'] = sum(cat_counts.get(cat, 0) for cat in group['categories'])
        
        # Метрики качества
        group_sizes = [g['total_count'] for g in groups if g['total_count'] > 0]
        
        # Проверка сохранения порядка
        preserves_order = True
        for i in range(1, len(groups)):
            prev_indices = groups[i-1].get('indices', [])
            curr_indices = groups[i].get('indices', [])
            if prev_indices and curr_indices:
                if max(prev_indices) >= min(curr_indices):
                    preserves_order = False
                    break
        
        # Покрытие данных
        covered_categories = set()
        for group in groups:
            covered_categories.update(group['categories'])
        
        coverage = len(covered_categories) / len([s['category'] for s in ordered_stats]) * 100
        
        return {
            'num_groups': len(groups),
            'size_cv': np.std(group_sizes) / np.mean(group_sizes) if group_sizes else 0,
            'preserves_order': preserves_order,
            'coverage': coverage,
            'min_size': min(group_sizes) if group_sizes else 0,
            'max_size': max(group_sizes) if group_sizes else 0
        }
    
    def _recommend_best_strategy(self, proposals, ordered_stats):
        """Рекомендует лучшую стратегию"""
        scores = {}
        
        for strategy_name, groups in proposals.items():
            quality = self._evaluate_grouping_quality(groups, ordered_stats)
            
            # Скоринг (веса можно настроить)
            score = 0
            score += (1 - quality['size_cv']) * 30  # Баланс размеров (30%)
            score += (1 if quality['preserves_order'] else 0) * 40  # Сохранение порядка (40%)
            score += quality['coverage'] / 100 * 20  # Покрытие (20%)
            score += (15 - abs(quality['num_groups'] - 15)) / 15 * 10  # Близость к целевому количеству (10%)
            
            scores[strategy_name] = score
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _create_visualizations(self, analysis_results, hand_type_counts):
        """Создает визуализации"""
        print(f"\n📊 Создание визуализаций...")
        
        ordered_stats = analysis_results['ordered_stats']
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Упорядоченное распределение
        ax1 = plt.subplot(3, 3, 1)
        indices = [s['index'] for s in ordered_stats]
        counts = [s['count'] for s in ordered_stats]
        colors = [plt.cm.RdYlGn_r(s['strength_class']/4) for s in ordered_stats]
        
        ax1.bar(indices, counts, color=colors, alpha=0.8)
        ax1.set_xlabel('Индекс категории (0=сильнейшая)')
        ax1.set_ylabel('Количество')
        ax1.set_title('Упорядоченное распределение по силе')
        ax1.set_yscale('log')
        
        # 2. Распределение по классам силы
        ax2 = plt.subplot(3, 3, 2)
        strength_counts = defaultdict(int)
        for stat in ordered_stats:
            strength_counts[stat['strength_class']] += stat['count']
        
        strengths = sorted(strength_counts.keys(), reverse=True)
        counts = [strength_counts[s] for s in strengths]
        labels = [f"{s}: {self.strength_names[s]}" for s in strengths]
        colors = ['green', 'lightgreen', 'yellow', 'orange', 'red'][:len(strengths)]
        
        wedges, texts, autotexts = ax2.pie(counts, labels=labels, colors=colors, 
                                           autopct='%1.1f%%', startangle=90)
        ax2.set_title('Распределение по классам силы')
        
        # 3. Кумулятивное распределение
        ax3 = plt.subplot(3, 3, 3)
        cumsum = np.cumsum([s['count'] for s in ordered_stats])
        cumsum_pct = cumsum / cumsum[-1] * 100
        
        ax3.plot(indices, cumsum_pct, 'b-', linewidth=3)
        ax3.fill_between(indices, 0, cumsum_pct, alpha=0.3)
        ax3.set_xlabel('Индекс категории')
        ax3.set_ylabel('Кумулятивный %')
        ax3.set_title('Кумулятивное покрытие данных')
        ax3.grid(True, alpha=0.3)
        
        # Добавляем квантили
        for q in [25, 50, 75, 90, 95]:
            idx = np.argmax(cumsum_pct >= q)
            ax3.axhline(y=q, color='r', linestyle='--', alpha=0.5)
            ax3.axvline(x=indices[idx], color='r', linestyle='--', alpha=0.5)
            ax3.text(indices[idx]+1, q-2, f'{q}%', fontsize=8)
        
        # 4. Heatmap распределения по силе
        ax4 = plt.subplot(3, 3, 4)
        
        # Создаем матрицу для heatmap
        matrix_size = 10
        heatmap_data = np.zeros((5, matrix_size))
        
        for stat in ordered_stats:
            strength = stat['strength_class']
            position = int(stat['index'] / len(self.ordered_categories) * matrix_size)
            position = min(position, matrix_size - 1)
            heatmap_data[strength, position] += stat['count']
        
        im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax4.set_yticks(range(5))
        ax4.set_yticklabels([self.strength_names[i] for i in range(5)])
        ax4.set_xlabel('Позиция в упорядоченном списке')
        ax4.set_ylabel('Класс силы')
        ax4.set_title('Тепловая карта распределения')
        plt.colorbar(im, ax=ax4)
        
        # 5. Топ-20 категорий
        ax5 = plt.subplot(3, 3, 5)
        top20 = hand_type_counts.head(20)
        y_pos = np.arange(len(top20))
        
        bars = ax5.barh(y_pos, top20.values)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(top20.index, fontsize=8)
        ax5.set_xlabel('Количество')
        ax5.set_title('Топ-20 наиболее частых типов')
        ax5.invert_yaxis()
        
        # Раскрашиваем по силе
        for i, (cat, bar) in enumerate(zip(top20.index, bars)):
            if cat in self.category_to_index:
                strength = self.strength_mapping[cat]
                bar.set_color(plt.cm.RdYlGn_r(strength/4))
        
        # 6. Редкие категории
        ax6 = plt.subplot(3, 3, 6)
        rare_stats = [s for s in ordered_stats if s['count'] < 50]
        
        if rare_stats:
            rare_indices = [s['index'] for s in rare_stats]
            rare_counts = [s['count'] for s in rare_stats]
            
            ax6.scatter(rare_indices, rare_counts, c='red', alpha=0.6, s=50)
            ax6.set_xlabel('Индекс категории')
            ax6.set_ylabel('Количество')
            ax6.set_title(f'Редкие категории (<50 примеров): {len(rare_stats)}')
            ax6.set_ylim(0, 55)
        
        # 7. Баланс по квинтилям
        ax7 = plt.subplot(3, 3, 7)
        quintile_data = analysis_results['problems']['imbalanced_regions']
        
        quintiles = [f"Q{d['quintile']}" for d in quintile_data]
        percentages = [d['percentage'] for d in quintile_data]
        
        bars = ax7.bar(quintiles, percentages)
        ax7.axhline(y=20, color='r', linestyle='--', label='Ожидаемое (20%)')
        ax7.set_ylabel('Процент данных')
        ax7.set_title('Распределение по квинтилям силы')
        ax7.legend()
        
        # Раскрашиваем отклонения
        for bar, pct in zip(bars, percentages):
            if abs(pct - 20) > 5:
                bar.set_color('red')
            elif abs(pct - 20) > 2:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # 8. Промежутки в данных
        ax8 = plt.subplot(3, 3, 8)
        gaps = analysis_results['problems']['gaps']
        
        if gaps:
            gap_positions = [(g['start'] + g['end']) / 2 for g in gaps]
            gap_sizes = [g['size'] for g in gaps]
            
            ax8.bar(range(len(gaps)), gap_sizes)
            ax8.set_xlabel('Промежуток #')
            ax8.set_ylabel('Размер промежутка')
            ax8.set_title(f'Промежутки в данных: {len(gaps)}')
        
        # 9. Метрики баланса
        ax9 = plt.subplot(3, 3, 9)
        metrics = analysis_results['balance_metrics']
        
        metric_names = ['CV', 'Gini', 'Entropy', 'Imbalance\nRatio/100']
        metric_values = [
            metrics['cv'],
            metrics['gini_coefficient'],
            metrics['entropy'] / np.log2(len(ordered_stats)),  # Нормализованная энтропия
            min(metrics['imbalance_ratio'] / 100, 1)  # Ограничиваем для визуализации
        ]
        
        bars = ax9.bar(metric_names, metric_values)
        ax9.set_ylim(0, 1)
        ax9.set_ylabel('Значение (0-1)')
        ax9.set_title('Метрики баланса')
        
        # Раскрашиваем по качеству
        for bar, val in zip(bars, metric_values):
            if val < 0.3:
                bar.set_color('green')
            elif val < 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig('hm3_ordinal_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Визуализации сохранены в hm3_ordinal_analysis.png")
        
        
    ######
    def save_10percent_grouping_config(self, groups, ordered_stats, timestamp):
        """
        Сохраняет конфигурацию 10% группировки с улучшенной структурой
        """
        # Определяем цвета для групп
        group_colors = [
            '#006400',  # G1 - Темно-зеленый (сильнейшие)
            '#228B22',  # G2 - Лесной зеленый
            '#32CD32',  # G3 - Лаймовый
            '#7CFC00',  # G4 - Зеленый газон
            '#ADFF2F',  # G5 - Желто-зеленый
            '#FFD700',  # G6 - Золотой
            '#FFA500',  # G7 - Оранжевый
            '#FF8C00',  # G8 - Темно-оранжевый
            '#FF6347',  # G9 - Томатный
            '#DC143C'   # G10 - Малиновый (слабейшие)
        ]
        
        config_10pct = {
            'strategy': '10percent_quantiles',
            'description': '10% группировка - каждая группа содержит примерно 10% от общего количества данных',
            'num_groups': len(groups),
            'groups': [],
            'category_to_group_mapping': {},
            'visualization': {
                'color_scheme': 'gradient_green_to_red',
                'group_colors': {}
            },
            'summary_statistics': {
                'total_examples': int(sum(g['total_count'] for g in groups)),
                'total_categories': int(sum(len(g['categories']) for g in groups)),
                'average_group_size': float(np.mean([g['total_count'] for g in groups])),
                'std_group_size': float(np.std([g['total_count'] for g in groups]))
            }
        }
        
        # Детальная информация по каждой группе
        for i, group in enumerate(groups):
            group_id = f'G{i+1}'
            
            # Определяем доминирующий класс силы
            strength_composition = {}
            for cat in group['categories']:
                for stat in ordered_stats:
                    if stat['category'] == cat:
                        strength_class = stat['strength_class']
                        strength_name = self.strength_names[strength_class]
                        if strength_name not in strength_composition:
                            strength_composition[strength_name] = 0
                        strength_composition[strength_name] += int(stat['count'])  # Конвертируем в int
                        break
            
            # Сортируем по количеству
            sorted_strengths = sorted(strength_composition.items(), key=lambda x: x[1], reverse=True)
            dominant_strength = sorted_strengths[0][0] if sorted_strengths else 'Unknown'
            
            # Процентный состав по классам силы
            total_in_group = sum(strength_composition.values())
            strength_percentages = {
                name: count / total_in_group * 100 
                for name, count in strength_composition.items()
            } if total_in_group > 0 else {}
            
            # Конвертируем все индексы в int
            indices_list = [int(idx) for idx in group['indices']]
            
            group_info = {
                'group_id': group_id,
                'group_index': i,
                'color': group_colors[i] if i < len(group_colors) else '#808080',
                'total_examples': int(group['total_count']),
                'percentage_of_data': float(group['percentage']),
                'num_categories': len(group['categories']),
                'categories': group['categories'],
                'index_range': {
                    'min': int(min(indices_list)),
                    'max': int(max(indices_list)),
                    'span': int(max(indices_list) - min(indices_list) + 1)
                },
                'strength_composition': {
                    'counts': {k: int(v) for k, v in strength_composition.items()},
                    'percentages': {k: round(float(v), 1) for k, v in strength_percentages.items()},
                    'dominant': dominant_strength
                },
                'average_strength_index': float(np.mean(indices_list)),
                'description': self._generate_group_description(i, group, dominant_strength)
            }
            
            config_10pct['groups'].append(group_info)
            
            # Маппинг категорий к группам
            for cat in group['categories']:
                config_10pct['category_to_group_mapping'][cat] = {
                    'group_id': group_id,
                    'group_index': i,
                    'group_color': group_colors[i] if i < len(group_colors) else '#808080'
                }
            
            # Цвета для визуализации
            config_10pct['visualization']['group_colors'][group_id] = group_colors[i] if i < len(group_colors) else '#808080'
        
        # Добавляем рекомендации по использованию
        config_10pct['usage_recommendations'] = {
            'ordinal_regression': {
                'suitable': True,
                'reason': 'Группы упорядочены по силе рук от сильнейших к слабейшим',
                'target_encoding': 'Используйте group_index как целевую переменную (0-9)'
            },
            'classification': {
                'suitable': True,
                'multiclass': True,
                'num_classes': len(groups),
                'class_balance': 'Примерно сбалансированные классы (~10% каждый)'
            },
            'feature_importance': 'Группы отражают естественную иерархию силы покерных рук'
        }
        
        # Сохраняем
        config_10pct_filename = f'hm3_10percent_grouping_{timestamp}.json'
        with open(config_10pct_filename, 'w', encoding='utf-8') as f:
            json.dump(config_10pct, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Конфигурация 10% группировки сохранена: {config_10pct_filename}")
        
        # Создаем упрощенный файл для быстрого использования
        simple_mapping = {
            'description': '10% группировка HM3 категорий для модели',
            'mapping': {}
        }
        
        for cat, info in config_10pct['category_to_group_mapping'].items():
            simple_mapping['mapping'][cat] = info['group_index']
        
        simple_filename = f'hm3_10pct_simple_mapping_{timestamp}.json'
        with open(simple_filename, 'w', encoding='utf-8') as f:
            json.dump(simple_mapping, f, indent=2, ensure_ascii=False)
        
        print(f"🎯 Упрощенный маппинг сохранен: {simple_filename}")
        
        return config_10pct_filename, simple_filename

    def _generate_group_description(self, group_index, group, dominant_strength):
        """
        Генерирует описание для группы
        """
        descriptions = [
            "Сильнейшие руки - премиум комбинации",
            "Очень сильные руки - натсы и околонатсовые",
            "Сильные руки - хорошие готовые комбинации",
            "Сильные/средние руки - микс сильных и средних",
            "Средние руки - стандартные комбинации",
            "Средние/слабые руки - микс средних и слабых",
            "Слабые руки - маргинальные комбинации",
            "Слабые руки и сильные дро",
            "Очень слабые руки и дро",
            "Мусор - слабейшие руки и слабые дро"
        ]
        
        if group_index < len(descriptions):
            return f"{descriptions[group_index]} (доминирует: {dominant_strength})"
        else:
            return f"Группа {group_index + 1} (доминирует: {dominant_strength})"    
    
    
    # 4. Визуализация 10% группировки
    def visualize_10percent_grouping(self, groups, ordered_stats):
        """
        Создает улучшенную визуализацию для 10% группировки с согласованными цветами
        """
        # Определяем цветовую схему для 10 групп (от сильных к слабым)
        # Используем градиент от зеленого (сильные) к красному (слабые)
        group_colors = [
            '#006400',  # G1 - Темно-зеленый (сильнейшие)
            '#228B22',  # G2 - Лесной зеленый
            '#32CD32',  # G3 - Лаймовый
            '#7CFC00',  # G4 - Зеленый газон
            '#ADFF2F',  # G5 - Желто-зеленый
            '#FFD700',  # G6 - Золотой
            '#FFA500',  # G7 - Оранжевый
            '#FF8C00',  # G8 - Темно-оранжевый
            '#FF6347',  # G9 - Томатный
            '#DC143C'   # G10 - Малиновый (слабейшие)
        ]
        
        plt.figure(figsize=(20, 12))
        
        # График 1: Основная информация о группах
        ax1 = plt.subplot(2, 3, 1)
        
        group_sizes = [g['total_count'] for g in groups]
        group_labels = [f"G{i+1}" for i in range(len(groups))]
        
        bars = ax1.bar(range(len(groups)), group_sizes, color=group_colors[:len(groups)])
        
        # Добавляем процент на каждый столбец
        for i, (bar, group) in enumerate(zip(bars, groups)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{group["percentage"]:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax1.set_xticks(range(len(groups)))
        ax1.set_xticklabels(group_labels, fontsize=12, fontweight='bold')
        ax1.set_ylabel('Количество примеров', fontsize=12)
        ax1.set_title('10% группировка: размеры групп', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Добавляем горизонтальную линию для идеального 10%
        ideal_size = sum(group_sizes) * 0.1
        ax1.axhline(y=ideal_size, color='red', linestyle='--', alpha=0.5, label='Идеальный размер (10%)')
        ax1.legend()
        
        # График 2: Состав групп по классам силы (улучшенный)
        ax2 = plt.subplot(2, 3, 2)
        
        # Подсчитываем состав каждой группы по классам силы
        group_compositions = []
        for group in groups:
            composition = [0] * 5  # 5 классов силы
            
            for cat in group['categories']:
                for stat in ordered_stats:
                    if stat['category'] == cat:
                        composition[stat['strength_class']] += stat['count']
                        break
            
            group_compositions.append(composition)
        
        # Нормализуем для процентного отображения
        group_compositions_pct = []
        for comp in group_compositions:
            total = sum(comp)
            if total > 0:
                comp_pct = [c/total * 100 for c in comp]
            else:
                comp_pct = [0] * 5
            group_compositions_pct.append(comp_pct)
        
        # Стековая диаграмма
        bottom = np.zeros(len(groups))
        strength_colors = {
            4: '#006400',  # Монстры - темно-зеленый
            3: '#32CD32',  # Сильные - ярко-зеленый
            2: '#FFD700',  # Средние - золотой
            1: '#FFA500',  # Слабые - оранжевый
            0: '#DC143C'   # Мусор - красный
        }
        
        for strength_class in range(5):
            values = [comp[strength_class] for comp in group_compositions_pct]
            ax2.bar(range(len(groups)), values, bottom=bottom, 
                    color=strength_colors[strength_class], 
                    label=f'Класс {strength_class}: {self.strength_names[strength_class]}',
                    edgecolor='white', linewidth=0.5)
            
            # Добавляем проценты для значимых компонентов
            for i, val in enumerate(values):
                if val > 5:  # Показываем только если больше 5%
                    ax2.text(i, bottom[i] + val/2, f'{val:.0f}%', 
                            ha='center', va='center', fontsize=9, fontweight='bold')
            
            bottom += values
        
        ax2.set_xticks(range(len(groups)))
        ax2.set_xticklabels([f"G{i+1}" for i in range(len(groups))], fontsize=12, fontweight='bold')
        ax2.set_ylabel('Процент состава', fontsize=12)
        ax2.set_title('Состав групп по классам силы', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax2.set_ylim(0, 100)
        
        # График 3: Детальная информация о каждой группе
        ax3 = plt.subplot(2, 3, 3)
        ax3.axis('off')
        
        # Создаем таблицу с информацией
        table_data = []
        for i, group in enumerate(groups):
            strength_classes = sorted(list(group['strength_classes']))
            dominant_class = max(set(strength_classes), key=strength_classes.count) if strength_classes else 0
            
            table_data.append([
                f'G{i+1}',
                f'{group["percentage"]:.1f}%',
                f'{len(group["categories"])}',
                f'{group["total_count"]:,}',
                self.strength_names[dominant_class]
            ])
        
        table = ax3.table(cellText=table_data,
                        colLabels=['Группа', '% данных', 'Категорий', 'Примеров', 'Доминирующий класс'],
                        cellLoc='center',
                        loc='center',
                        colColours=['lightgray']*5)
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # Раскрашиваем ячейки групп
        for i in range(len(groups)):
            table[(i+1, 0)].set_facecolor(group_colors[i])
            table[(i+1, 0)].set_text_props(weight='bold', color='white')
        
        ax3.set_title('Сводная информация по группам', fontsize=14, fontweight='bold', pad=20)
        
        # График 4: Примеры категорий в каждой группе
        ax4 = plt.subplot(2, 3, 4)
        
        y_positions = []
        for i, group in enumerate(groups):
            y_pos = len(groups) - i - 1
            y_positions.append(y_pos)
            
            # Показываем первые 3 категории
            sample_cats = group['categories'][:3]
            if len(group['categories']) > 3:
                text = f"G{i+1}: {', '.join(sample_cats)}..."
            else:
                text = f"G{i+1}: {', '.join(sample_cats)}"
            
            ax4.text(0.05, y_pos, text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=group_colors[i], alpha=0.7),
                    color='white' if i < 3 else 'black', fontweight='bold')
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(-0.5, len(groups) - 0.5)
        ax4.axis('off')
        ax4.set_title('Примеры категорий в группах', fontsize=14, fontweight='bold')
        
        # График 5: Распределение силы по группам
        ax5 = plt.subplot(2, 3, 5)
        
        # Для каждой группы вычисляем средний индекс силы
        avg_indices = []
        for group in groups:
            if group['indices']:
                avg_idx = np.mean(group['indices'])
                avg_indices.append(avg_idx)
            else:
                avg_indices.append(0)
        
        ax5.plot(range(len(groups)), avg_indices, 'ko-', linewidth=2, markersize=10)
        
        # Раскрашиваем точки
        for i, (x, y) in enumerate(zip(range(len(groups)), avg_indices)):
            ax5.plot(x, y, 'o', color=group_colors[i], markersize=12)
        
        ax5.set_xticks(range(len(groups)))
        ax5.set_xticklabels([f"G{i+1}" for i in range(len(groups))], fontsize=12, fontweight='bold')
        ax5.set_ylabel('Средний индекс силы', fontsize=12)
        ax5.set_title('Тренд силы по группам', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.invert_yaxis()  # Инвертируем, чтобы сильные были вверху
        
        # График 6: Круговая диаграмма с улучшенным дизайном
        ax6 = plt.subplot(2, 3, 6)
        
        percentages = [g['percentage'] for g in groups]
        
        # Создаем эксплодированную диаграмму
        explode = [0.05 if p > 12 else 0.02 for p in percentages]  # Выделяем большие группы
        
        wedges, texts, autotexts = ax6.pie(percentages, 
                                        labels=[f"G{i+1}" for i in range(len(groups))],
                                        colors=group_colors[:len(groups)],
                                        autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
                                        startangle=90,
                                        explode=explode,
                                        shadow=True)
        
        # Улучшаем текст
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax6.set_title('Процентное распределение групп', fontsize=14, fontweight='bold')
        
        # Общий заголовок
        plt.suptitle('Анализ 10% группировки HM3 категорий', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('hm3_10percent_grouping_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Улучшенная визуализация 10% группировки сохранена: hm3_10percent_grouping_improved.png")
        
        # Дополнительно выводим сводку
        print("\n📋 СВОДКА ПО 10% ГРУППИРОВКЕ:")
        print("=" * 80)
        for i, group in enumerate(groups):
            strength_desc = ", ".join([f"{self.strength_names[s]}" for s in sorted(group['strength_classes'])])
            print(f"\n🎯 Группа {i+1} (G{i+1}) - {group['percentage']:.1f}% данных:")
            print(f"   Цвет: {group_colors[i]}")
            print(f"   Категорий: {len(group['categories'])}")
            print(f"   Примеров: {group['total_count']:,}")
            print(f"   Индексы: {min(group['indices'])}-{max(group['indices'])}")
            print(f"   Классы силы: {strength_desc}")
        print("=" * 80)
    
    # Исправленный метод _save_results
    def _save_results(self, analysis_results, grouping_proposals):
        """Сохраняет результаты анализа с правильной конвертацией типов"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Функция для безопасной конвертации numpy/pandas типов
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            else:
                return obj
        
        # Подготовка данных для JSON с конвертацией
        json_results = {
            'timestamp': timestamp,
            'summary': {
                'total_hands': int(analysis_results['total_hands']),
                'categories_found': len(analysis_results['ordered_stats']),
                'categories_missing': len(analysis_results['missing_categories']),
                'balance_metrics': convert_to_serializable(analysis_results['balance_metrics']),
                'problems_summary': {
                    'rare_categories': len(analysis_results['problems']['rare']),
                    'scarce_categories': len(analysis_results['problems']['scarce']),
                    'gaps_count': len(analysis_results['problems']['gaps'])
                }
            },
            'strength_distribution': convert_to_serializable(analysis_results['strength_distribution']),
            'ordered_statistics': [
                convert_to_serializable({
                    'index': s['index'],
                    'category': s['category'],
                    'count': s['count'],
                    'percentage': s['percentage'],
                    'strength_class': s['strength_class'],
                    'cumulative_percentage': s['cumulative_percentage']
                })
                for s in analysis_results['ordered_stats']
            ],
            'grouping_proposals': convert_to_serializable(grouping_proposals)
        }
        
        # Сохранение JSON
        json_filename = f'hm3_ordinal_analysis_{timestamp}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результаты сохранены:")
        print(f"   📊 Графики: hm3_ordinal_analysis.png")
        print(f"   📋 Данные: {json_filename}")
        
        # Сохранение рекомендованной группировки отдельно
        best_strategy = self._recommend_best_strategy(grouping_proposals, analysis_results['ordered_stats'])
        best_groups = grouping_proposals[best_strategy]
        
        grouping_config = {
            'strategy': best_strategy,
            'num_groups': len(best_groups),
            'category_to_group': {},
            'group_info': {}
        }
        
        for i, group in enumerate(best_groups):
            for cat in group['categories']:
                grouping_config['category_to_group'][cat] = i
            
            grouping_config['group_info'][str(i)] = convert_to_serializable({
                'categories': group['categories'],
                'size': len(group['categories']),
                'total_count': group['total_count']
            })
        
        config_filename = f'hm3_grouping_config_{timestamp}.json'
        with open(config_filename, 'w', encoding='utf-8') as f:
            json.dump(grouping_config, f, indent=2, ensure_ascii=False)
        
        print(f"   🎯 Конфигурация группировки: {config_filename}")
        
        # Сохраняем конфигурацию 10% группировки используя новый метод
        if '10percent_quantiles' in grouping_proposals:
            groups_10pct = grouping_proposals['10percent_quantiles']
            
            # Используем новый метод
            config_10pct_filename, simple_filename = self.save_10percent_grouping_config(
                groups_10pct, 
                analysis_results['ordered_stats'],
                timestamp
            )

def main():
    """Главная функция с поддержкой выбора файла из папки data"""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description='Анализ распределения 73 классов HM3')
    parser.add_argument('data_file', nargs='?', help='Путь к CSV файлу с данными (опционально)')
    parser.add_argument('--output-dir', default='.', help='Директория для сохранения результатов')
    parser.add_argument('--data-dir', default='data', help='Директория с данными (по умолчанию: data)')
    
    args = parser.parse_args()
    
    # Если файл не указан, ищем в папке data
    if args.data_file and os.path.exists(args.data_file):
        selected_file = args.data_file
    else:
        # Поиск CSV файлов в папке data
        data_dir = args.data_dir
        if not os.path.exists(data_dir):
            data_dir = "."
        
        # Паттерны для поиска
        patterns = [
            os.path.join(data_dir, "parsed_*.csv"),
            os.path.join(data_dir, "*poker*.csv"),
            os.path.join(data_dir, "*PDOM*.csv"),
            os.path.join(data_dir, "*.csv")
        ]
        
        data_files = []
        for pattern in patterns:
            files = glob.glob(pattern)
            # Исключаем файлы из подпапки combined
            files = [f for f in files if 'combined' not in os.path.dirname(f)]
            data_files.extend(files)
        
        # Убираем дубликаты
        data_files = sorted(list(set(data_files)))
        
        if not data_files:
            print(f"❌ CSV файлы не найдены в папке {data_dir}!")
            return 1
        
        if len(data_files) == 1:
            selected_file = data_files[0]
            print(f"✅ Найден единственный файл: {selected_file}")
        else:
            # Показываем список файлов с размерами
            print(f"\n📁 Найдено {len(data_files)} файлов в папке {data_dir}:")
            print("=" * 80)
            
            file_info = []
            total_size = 0
            
            for i, file_path in enumerate(data_files):
                size_mb = os.path.getsize(file_path) / 1024 / 1024
                total_size += size_mb
                file_info.append((i, file_path, size_mb))
                
            # Сортируем по размеру (большие первыми)
            file_info.sort(key=lambda x: x[2], reverse=True)
            
            # Показываем файлы
            for idx, (orig_idx, file_path, size_mb) in enumerate(file_info):
                if idx < 20:  # Показываем только топ-20
                    print(f"{orig_idx+1:3d}. {os.path.basename(file_path):50s} {size_mb:8.1f} MB")
            
            if len(data_files) > 20:
                print(f"\n... и еще {len(data_files) - 20} файлов")
            
            print("=" * 80)
            print(f"📊 Общий размер: {total_size:.1f} MB")
            print(f"\n💡 Совет: выберите файл с наибольшим количеством данных")
            print(f"   или введите 'all' для анализа всех файлов")
            
            # Выбор файла
            while True:
                choice = input("\nВыберите номер файла (1-{}) или 'all' для всех: ".format(len(data_files))).strip()
                
                if choice.lower() == 'all':
                    print("\n🔄 Объединение всех файлов для анализа...")
                    
                    # Объединяем все файлы
                    all_dfs = []
                    total_showdowns = 0
                    
                    for file_path in data_files:
                        try:
                            print(f"   Загрузка {os.path.basename(file_path)}...", end='', flush=True)
                            df = pd.read_csv(file_path)
                            
                            # Добавляем источник
                            df['source_file'] = os.path.basename(file_path)
                            
                            # Считаем шоудауны
                            showdowns = ((df['Showdown_1'].notna()) & (df['Showdown_2'].notna())).sum()
                            total_showdowns += showdowns
                            
                            all_dfs.append(df)
                            print(f" ✓ ({len(df)} строк, {showdowns} шоудаунов)")
                            
                        except Exception as e:
                            print(f" ✗ Ошибка: {e}")
                    
                    if not all_dfs:
                        print("❌ Не удалось загрузить ни одного файла!")
                        return 1
                    
                    # Объединяем
                    print(f"\n🔗 Объединение {len(all_dfs)} файлов...")
                    combined_df = pd.concat(all_dfs, ignore_index=True)
                    
                    # Сохраняем временный файл
                    temp_file = f"temp_combined_for_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    combined_df.to_csv(temp_file, index=False)
                    
                    print(f"✅ Объединено: {len(combined_df)} строк, {total_showdowns} шоудаунов")
                    selected_file = temp_file
                    break
                    
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(data_files):
                        selected_file = data_files[idx]
                        print(f"✅ Выбран файл: {selected_file}")
                        break
                    else:
                        print(f"❌ Неверный номер. Введите число от 1 до {len(data_files)}")
                except ValueError:
                    print(f"❌ Введите число или 'all'")
    
    # Создание анализатора
    analyzer = HM3OrdinalAnalyzer()
    
    # Создаем директорию для результатов если нужно
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Меняем рабочую директорию для сохранения результатов
    original_dir = os.getcwd()
    os.chdir(args.output_dir)
    
    try:
        # Запуск анализа
        print(f"\n🚀 Запуск анализа файла: {os.path.basename(selected_file)}")
        analysis_results, grouping_proposals = analyzer.analyze_data(selected_file)
        
        if analysis_results:
            print(f"\n✅ Анализ завершен успешно!")
            print(f"🎯 Рекомендуется использовать сохраненную конфигурацию группировки")
            print(f"   для обучения модели с ординальной регрессией")
            
            # Удаляем временный файл если создавали
            if 'temp_combined_for_analysis' in selected_file:
                os.remove(selected_file)
                print(f"🗑️  Временный файл удален")
            
            return 0
        else:
            print(f"\n❌ Анализ не удался")
            return 1
            
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Возвращаемся в исходную директорию
        os.chdir(original_dir)


if __name__ == "__main__":
    exit(main())