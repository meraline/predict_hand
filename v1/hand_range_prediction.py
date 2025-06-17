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
from collections import defaultdict
import itertools
from collections import Counter



def safe_json_serialize(obj):
    """Безопасная конвертация всех типов для JSON"""
    import numpy as np

    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj




class PokerHandEvaluator:
    """Класс для определения типа руки из 73 комбинаций HM3"""

    RANK_VALUES = {
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "T": 10,
        "J": 11,
        "Q": 12,
        "K": 13,
        "A": 14,
    }

    SUIT_MAP = {"s": 0, "h": 1, "d": 2, "c": 3}

    def __init__(self):
        self.hand_type_to_strength = self._create_strength_mapping()

    def _create_strength_mapping(self):
        """Создает маппинг типов рук в 5 классов силы"""
        return {
            # Класс 4: МОНСТРЫ
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
            # Класс 3: СИЛЬНЫЕ
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
            # Класс 2: СРЕДНИЕ
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
            # Класс 1: СЛАБЫЕ
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
            # Класс 0: МУСОР
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
            "Other": 0,
        }

    def parse_card(self, card_str):
        """Парсит карту из строки"""
        if not card_str or len(card_str) < 2:
            return None, None
        rank = self.RANK_VALUES.get(card_str[0].upper())
        suit = self.SUIT_MAP.get(card_str[1].lower())
        return rank, suit

    def evaluate_hand(self, hole_cards, board_cards):
        """
        Определяет тип руки из 73 комбинаций

        Args:
            hole_cards: список из 2 карт игрока ['As', 'Kh']
            board_cards: список карт на борде ['Qd', 'Js', 'Tc', '9h', '8s']

        Returns:
            tuple: (hand_type_name, strength_class)
        """
        # Парсим карты
        hole = [self.parse_card(c) for c in hole_cards if c]
        board = [self.parse_card(c) for c in board_cards if c]

        # Убираем None значения
        hole = [(r, s) for r, s in hole if r is not None]
        board = [(r, s) for r, s in board if r is not None]

        if len(hole) != 2:
            return "Other", 0

        # Все карты вместе
        all_cards = hole + board

        # Проверяем комбинации в порядке убывания силы

        # 1. Стрит-флеш
        sf_type = self._check_straight_flush(hole, board, all_cards)
        if sf_type:
            return sf_type, self.hand_type_to_strength[sf_type]

        # 2. Каре
        quads_type = self._check_four_of_kind(hole, board, all_cards)
        if quads_type:
            return quads_type, self.hand_type_to_strength[quads_type]

        # 3. Фулл-хаус
        fh_type = self._check_full_house(hole, board, all_cards)
        if fh_type:
            return fh_type, self.hand_type_to_strength[fh_type]

        # 4. Флеш
        flush_type = self._check_flush(hole, board, all_cards)
        if flush_type:
            return flush_type, self.hand_type_to_strength[flush_type]

        # 5. Стрит
        straight_type = self._check_straight(hole, board, all_cards)
        if straight_type:
            return straight_type, self.hand_type_to_strength[straight_type]

        # 6. Сет/Трипс
        trips_type = self._check_three_of_kind(hole, board, all_cards)
        if trips_type:
            return trips_type, self.hand_type_to_strength[trips_type]

        # 7. Две пары
        two_pair_type = self._check_two_pair(hole, board, all_cards)
        if two_pair_type:
            return two_pair_type, self.hand_type_to_strength[two_pair_type]

        # 8. Одна пара
        pair_type = self._check_one_pair(hole, board, all_cards)
        if pair_type:
            return pair_type, self.hand_type_to_strength[pair_type]

        # 9. Дро
        draw_type = self._check_draws(hole, board)
        if draw_type:
            return draw_type, self.hand_type_to_strength[draw_type]

        # 10. Высокие карты
        high_card_type = self._check_high_cards(board)
        if high_card_type:
            return high_card_type, self.hand_type_to_strength[high_card_type]

        return "Other", 0

    def _check_straight_flush(self, hole, board, all_cards):
        """Проверяет стрит-флеш"""
        # Группируем по мастям
        suits = {}
        for rank, suit in all_cards:
            if suit not in suits:
                suits[suit] = []
            suits[suit].append(rank)

        # Проверяем каждую масть
        for suit, ranks in suits.items():
            if len(ranks) >= 5:
                # Проверяем стрит в этой масти
                if self._is_straight_in_ranks(sorted(ranks, reverse=True)):
                    # Считаем сколько карт из hole cards участвуют
                    hole_in_sf = sum(1 for r, s in hole if s == suit and r in ranks[:5])
                    if hole_in_sf == 2:
                        return "TwoCardStraightFlush"
                    elif hole_in_sf == 1:
                        return "OneCardStraightFlush"
        return None

    def _check_four_of_kind(self, hole, board, all_cards):
        """Проверяет каре"""
        rank_counts = Counter(r for r, s in all_cards)

        for rank, count in rank_counts.items():
            if count == 4:
                hole_ranks = [r for r, s in hole]
                board_ranks = [r for r, s in board]

                # Каре на борде
                if board_ranks.count(rank) == 4:
                    return "FourOfAKindOnBoard"
                # Карманная пара
                elif hole_ranks[0] == hole_ranks[1] == rank:
                    return "FourOfAKindWithPocketPair"
                else:
                    return "FourOfAKindWithoutPocketPair"
        return None

    def _check_full_house(self, hole, board, all_cards):
        """Проверяет фулл-хаус"""
        rank_counts = Counter(r for r, s in all_cards)
        trips = [r for r, c in rank_counts.items() if c >= 3]
        pairs = [r for r, c in rank_counts.items() if c >= 2]

        if trips and len(pairs) >= 2:
            hole_ranks = [r for r, s in hole]
            board_ranks = [r for r, s in board]

            # Проверяем разные типы фулл-хаусов
            if hole_ranks[0] == hole_ranks[1]:  # Карманная пара
                if board_ranks.count(trips[0]) == 3:  # Трипс на борде
                    return "FullHouseWithPocketPairTripsOnBoard"
                else:
                    return "FullHouseWithPocketPairNoTripsOnBoard"
            elif board_ranks.count(trips[0]) == 3:  # Трипс на борде
                return "OneCardFullHouseTripsOnBoard"
            elif max(hole_ranks) in trips and max(board_ranks) == max(hole_ranks):
                return "OneCardFullHouseTopPair"
            else:
                return "TwoCardFullHouseWithoutPocketPair"
        return None

    def _check_flush(self, hole, board, all_cards):
        """Проверяет флеш"""
        suit_counts = Counter(s for r, s in all_cards)

        for suit, count in suit_counts.items():
            if count >= 5:
                # Карты этой масти
                suited_cards = sorted(
                    [r for r, s in all_cards if s == suit], reverse=True
                )
                hole_suited = [r for r, s in hole if s == suit]
                board_suited = [r for r, s in board if s == suit]

                # ИСПРАВЛЕНИЕ: Проверяем, что hole_suited не пустой
                if not hole_suited:
                    # Флеш полностью на борде
                    continue

                # Определяем силу флеша
                if 14 in hole_suited:  # Туз у игрока
                    if len(board_suited) == 3:
                        return "ThreeFlushBoardNutFlush"
                    else:
                        return "FourFlushBoardNutFlush"
                elif max(hole_suited) >= 11:  # K, Q, J
                    if len(board_suited) == 3:
                        return "ThreeFlushBoardHighFlush"
                    else:
                        return "FourFlushBoardHighFlush"
                else:
                    if len(board_suited) == 3:
                        return "ThreeFlushBoardLowFlush"
                    else:
                        return "FourFlushBoardLowFlush"
        return None

    def _check_straight(self, hole, board, all_cards):
        """Проверяет стрит"""
        ranks = sorted(set(r for r, s in all_cards), reverse=True)

        # Проверяем колесо (A-2-3-4-5)
        if set([14, 2, 3, 4, 5]).issubset(ranks):
            ranks.append(1)  # Туз как единица

        # Ищем стрит
        for i in range(len(ranks) - 4):
            if ranks[i] - ranks[i + 4] == 4:
                straight_ranks = ranks[i : i + 5]
                hole_ranks = [r for r, s in hole]

                # Считаем карты игрока в стрите
                hole_in_straight = sum(1 for r in hole_ranks if r in straight_ranks)

                # Проверяем натсовость
                is_nut = straight_ranks[0] == max(ranks) or (
                    14 in straight_ranks and 13 in straight_ranks
                )

                if hole_in_straight == 2:
                    return "TwoCardNutStraight" if is_nut else "TwoCardStraight"
                elif hole_in_straight == 1:
                    return "OneCardNutStraight" if is_nut else "OneCardStraight"
        return None

    def _check_three_of_kind(self, hole, board, all_cards):
        """Проверяет сет/трипс"""
        rank_counts = Counter(r for r, s in all_cards)

        for rank, count in rank_counts.items():
            if count == 3:
                hole_ranks = [r for r, s in hole]
                board_ranks = sorted([r for r, s in board], reverse=True)

                # Трипс на борде
                if board_ranks.count(rank) == 3:
                    return "TripsOnBoard"

                # Сет (карманная пара)
                if hole_ranks[0] == hole_ranks[1] == rank:
                    board_higher = [r for r in board_ranks if r > rank]
                    if len(board_higher) == 0:
                        return "HighSet"
                    elif len(board_higher) == 1:
                        return "SecondSet"
                    else:
                        return "LowSet"

                # Трипс (одна карта в руке)
                else:
                    # ИСПРАВЛЕНИЕ: Проверяем, что есть карты для кикера
                    kicker_candidates = [r for r in hole_ranks if r != rank]
                    if not kicker_candidates:
                        return "LowTripsLowKicker"

                    kicker = max(kicker_candidates)
                    board_higher = [r for r in board_ranks if r > rank]

                    if len(board_higher) == 0:  # Топ трипс
                        return (
                            "HighTripsHighKicker"
                            if kicker >= 12
                            else "HighTripsLowKicker"
                        )
                    elif len(board_higher) == 1:  # Средний трипс
                        return (
                            "SecondTripsHighKicker"
                            if kicker >= 12
                            else "SecondTripsLowKicker"
                        )
                    else:  # Младший трипс
                        return (
                            "LowTripsHighKicker"
                            if kicker >= 12
                            else "LowTripsLowKicker"
                        )
        return None

    def _check_two_pair(self, hole, board, all_cards):
        """Проверяет две пары"""
        rank_counts = Counter(r for r, s in all_cards)
        pairs = sorted([r for r, c in rank_counts.items() if c >= 2], reverse=True)

        if len(pairs) >= 2:
            hole_ranks = [r for r, s in hole]
            board_ranks = sorted([r for r, s in board], reverse=True)

            top_board = board_ranks[0] if board_ranks else 0

            # Обе карты игрока спарены
            if (
                hole_ranks[0] in pairs
                and hole_ranks[1] in pairs
                and hole_ranks[0] != hole_ranks[1]
            ):
                if hole_ranks[0] == top_board and hole_ranks[1] == board_ranks[1]:
                    return "TopTwoPair"
                elif max(hole_ranks) == top_board:
                    return "TopPairPlusPair"
                else:
                    return "NonTopTwoPair"

            # Карманная пара
            elif hole_ranks[0] == hole_ranks[1] and hole_ranks[0] in pairs:
                board_pairs = [r for r in board_ranks if rank_counts[r] >= 2]
                if board_pairs:
                    if hole_ranks[0] > board_pairs[0]:
                        return "PocketPairOverPairPlusLowerPairedBoard"
                    elif hole_ranks[0] > min(board_pairs):
                        return "PocketPairPlusHigherPairedBoard"
                    else:
                        return "PocketPairPlusLowerPairedBoard"

            # Одна пара с игроком + пара на борде
            elif any(r in pairs for r in hole_ranks):
                player_pair = next(r for r in hole_ranks if r in pairs)
                if player_pair == top_board:
                    return "TopPairPlusPairedBoard"
                elif board_ranks.index(player_pair) == 1:
                    return "SecondPairPlusPairedBoard"
                else:
                    return "LowPairPlusPairedBoard"

            # Две пары на борде
            else:
                return "TwoPairsOnBoard"
        return None

    def _check_one_pair(self, hole, board, all_cards):
        """Проверяет одну пару"""
        rank_counts = Counter(r for r, s in all_cards)
        pairs = [r for r, c in rank_counts.items() if c == 2]

        if pairs:
            hole_ranks = sorted([r for r, s in hole], reverse=True)
            board_ranks = sorted([r for r, s in board], reverse=True)
            pair_rank = max(pairs)

            # Карманная пара
            if hole_ranks[0] == hole_ranks[1] == pair_rank:
                if not board_ranks or pair_rank > max(board_ranks):
                    return "OverPair"
                elif pair_rank in board_ranks:
                    board_position = board_ranks.index(pair_rank)
                    if board_position == 1:
                        return "SecondPocketPair"
                    else:
                        return "LowPocketPair"
                else:
                    return "LowPocketPair"

            # Пара с бордом
            elif pair_rank in hole_ranks:
                # ИСПРАВЛЕНИЕ: Проверяем наличие кикера
                kicker_candidates = [r for r in hole_ranks if r != pair_rank]
                if not kicker_candidates:
                    kicker = 2  # Минимальный ранг
                else:
                    kicker = max(kicker_candidates)

                # Проверяем, есть ли pair_rank в board_ranks перед использованием index()
                if pair_rank in board_ranks:
                    board_position = board_ranks.index(pair_rank)

                    if board_position == 0:  # Топ пара
                        if kicker == 14:
                            return "TopPairTopKicker"
                        elif kicker >= 11:
                            return "TopPairGoodKicker"
                        else:
                            return "TopPairWeakKicker"
                    elif board_position == 1:  # Средняя пара
                        return (
                            "SecondPairAceKicker"
                            if kicker == 14
                            else "SecondPairNonAceKicker"
                        )
                    else:  # Младшая пара
                        return (
                            "BottomPairAceKicker"
                            if kicker == 14
                            else "BottomPairNonAceKicker"
                        )
                else:
                    # Если pair_rank нет в board_ranks, это может быть карманная пара
                    return "LowPocketPair"

            # Пара на борде
            else:
                if board_ranks:
                    # ИСПРАВЛЕНИЕ: Проверяем, что board_ranks не пустой
                    overcards = sum(1 for r in hole_ranks if r > max(board_ranks))
                    if overcards == 0:
                        return "PairedBoardNoOvercards"
                    elif overcards == 1:
                        return "PairedBoardOneOvercard"
                    else:
                        return "PairedBoardTwoOvercards"
                else:
                    return "PairedBoardTwoOvercards"

        return None

    def _check_draws(self, hole, board):
        """Проверяет дро"""
        if len(board) < 5:  # Только на флопе и терне есть дро
            # Флеш-дро
            flush_draw = self._check_flush_draw(hole, board)
            if flush_draw:
                return flush_draw

            # Стрит-дро
            straight_draw = self._check_straight_draw(hole, board)
            if straight_draw:
                return straight_draw
        return None

    def _check_flush_draw(self, hole, board):
        """Проверяет флеш-дро"""
        all_cards = hole + board
        suit_counts = Counter(s for r, s in all_cards)

        for suit, count in suit_counts.items():
            if count == 4:  # Флеш-дро
                hole_suited = [r for r, s in hole if s == suit]
                if len(hole_suited) == 2:
                    if 14 in hole_suited:
                        return "TwoCardNutFlushDraw"
                    elif max(hole_suited) >= 11:
                        return "TwoCardHighFlushDraw"
                    else:
                        return "TwoCardLowFlushDraw"
                elif len(hole_suited) == 1:
                    if hole_suited[0] == 14:
                        return "OneCardNutFlushDraw"
                    elif hole_suited[0] >= 11:
                        return "OneCardHighFlushDraw"
                    else:
                        return "OneCardLowFlushDraw"
            elif count == 3 and len(board) == 3:  # Бэкдорное флеш-дро
                hole_suited = [r for r, s in hole if s == suit]
                # ИСПРАВЛЕНИЕ: Проверяем, что hole_suited не пустой
                if not hole_suited:
                    continue

                if len(hole_suited) == 2:
                    if 14 in hole_suited:
                        return "TwoCardBackdoorNutFlushDraw"
                    else:
                        return "TwoCardBackdoorFlushDraw"
                elif len(hole_suited) == 1:
                    if hole_suited[0] == 14:
                        return "OneCardBackDoorNutFlushDraw"
                    else:
                        return "OneCardBackDoorFlushDraw"
        return None

    def _check_straight_draw(self, hole, board):
        """Проверяет стрит-дро"""
        all_ranks = sorted(set(r for r, s in hole + board), reverse=True)
        hole_ranks = [r for r, s in hole]

        # OESD и гатшоты
        outs = 0
        draw_type = None

        for target in range(14, 4, -1):
            straight_cards = list(range(target - 4, target + 1))
            have = sum(1 for r in straight_cards if r in all_ranks)
            hole_in = sum(1 for r in straight_cards if r in hole_ranks)

            if have == 4 and hole_in >= 1:
                missing = [r for r in straight_cards if r not in all_ranks][0]
                if missing == straight_cards[0] or missing == straight_cards[4]:
                    # OESD
                    if hole_in == 2:
                        return "TwoCardOpenEndedStraightDraw"
                    else:
                        return "OneCardOpenEndedStraightDraw"
                else:
                    # Гатшот
                    if hole_in == 2:
                        draw_type = "TwoCardGutshotStraightDraw"
                    else:
                        draw_type = "OneCardGutshotStraightDraw"
                    outs += 4

        if outs >= 8:
            return "TwoCardDoubleGutShotStraightDraw"
        elif draw_type:
            return draw_type

        # Бэкдорное стрит-дро (только на флопе)
        if len(board) == 3:
            for target in range(14, 4, -1):
                straight_cards = list(range(target - 4, target + 1))
                have = sum(1 for r in straight_cards if r in all_ranks)
                hole_in = sum(1 for r in straight_cards if r in hole_ranks)

                if have == 3 and hole_in >= 1:
                    if hole_in == 2:
                        return "TwoCardBackdoorStraightDraw"
                    else:
                        return "OneCardBackdoorStraightDraw"

        return None

    def _check_high_cards(self, board):
        """Проверяет высокие карты на борде"""
        board_ranks = [r for r, s in board]
        if 14 in board_ranks:
            return "AceOnBoard"
        elif 13 in board_ranks:
            return "KingOnBoard"
        return None

    def _is_straight_in_ranks(self, ranks):
        """Проверяет есть ли стрит в списке рангов"""
        ranks = sorted(set(ranks), reverse=True)

        # Проверяем обычные стриты
        for i in range(len(ranks) - 4):
            if ranks[i] - ranks[i + 4] == 4:
                return True

        # Проверяем колесо (A-5)
        if set([14, 2, 3, 4, 5]).issubset(ranks):
            return True

        return False


# Функция для добавления в препроцессинг данных
def add_hand_evaluation_to_dataframe(df):
    """Добавляет оценку типа руки и силы к DataFrame"""
    evaluator = PokerHandEvaluator()

    hand_types = []
    strength_classes = []
    errors_count = 0

    for idx, row in df.iterrows():
        try:
            # Карты игрока
            hole_cards = [row.get("Showdown_1"), row.get("Showdown_2")]

            # Карты борда
            board_cards = []
            for i in range(1, 6):
                card = row.get(f"Card{i}")
                if pd.notna(card) and card:
                    board_cards.append(card)

            # Оцениваем руку
            if all(pd.notna(c) for c in hole_cards):
                hand_type, strength_class = evaluator.evaluate_hand(
                    hole_cards, board_cards
                )
            else:
                hand_type, strength_class = "Other", 0

        except Exception as e:
            errors_count += 1
            if errors_count < 10:  # Показываем только первые 10 ошибок
                print(f"⚠️ Ошибка в строке {idx}: {e}")
            hand_type, strength_class = "Other", 0

        hand_types.append(hand_type)
        strength_classes.append(strength_class)

        # Прогресс для больших данных
        if idx > 0 and idx % 10000 == 0:
            print(f"   Обработано {idx:,} / {len(df):,} строк...")

    if errors_count > 0:
        print(f"⚠️ Всего ошибок при оценке рук: {errors_count}")

    df["hand_type_hm3"] = hand_types
    df["hand_strength_class"] = strength_classes

    # Статистика
    print(f"\n📊 Распределение типов рук:")
    type_counts = df["hand_type_hm3"].value_counts()
    for hand_type, count in type_counts.head(20).items():
        strength = evaluator.hand_type_to_strength.get(hand_type, 0)
        print(
            f"   {hand_type:40s}: {count:5d} ({count/len(df)*100:5.1f}%) - Сила: {strength}"
        )

    print(f"\n📊 Распределение классов силы:")
    strength_dist = df["hand_strength_class"].value_counts().sort_index()
    class_names = ["Мусор/Дро", "Слабые", "Средние", "Сильные", "Монстры"]
    for strength, count in strength_dist.items():
        print(
            f"   Класс {strength} ({class_names[strength]}): {count:5d} ({count/len(df)*100:5.1f}%)"
        )

    return df


# ---------------------- 1. Модель RWKV ----------------------


class SequenceHandRangeRWKV(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=3,
        max_sequence_length=20,
        num_strength_classes=5,
        num_categories=73,
    ):
        super(
            SequenceHandRangeRWKV, self
        ).__init__()  # Правильный вызов родительского конструктора

        # Сохраняем параметры
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        self.num_strength_classes = num_strength_classes
        self.num_categories = num_categories

        # Создаем слои
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.rwkv_layers = nn.ModuleList(
            [RWKV_Block(hidden_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.3)

        # Выходные слои для разных предсказаний
        self.hand_strength_head = nn.Linear(hidden_dim, num_strength_classes)
        self.category_head = nn.Linear(hidden_dim, num_categories)
        self.specific_hand_head = nn.Linear(hidden_dim, 13)  # 13 рангов карт

        # Sigmoid только для specific_hand
        self.sigmoid = nn.Sigmoid()

    def reset_states(self):
        """Сброс состояния всех RWKV слоев"""
        for layer in self.rwkv_layers:
            layer.reset_state()

    def forward(self, x, return_all_timesteps=False):
        """
        x: [batch_size, sequence_length, input_dim]
        return_all_timesteps: если True, возвращает предсказания для всех временных шагов
        """
        batch_size, seq_len, _ = x.size()

        # Применяем embedding
        x = self.embedding(x)  # [batch_size, seq_len, hidden_dim]

        # Проходим через RWKV слои
        for layer in self.rwkv_layers:
            residual = x
            x = layer(x)
            x = residual + x
            x = self.norm(x)
            x = self.dropout(x)

        # Если нужны все временные шаги
        if return_all_timesteps:
            # Предсказания для каждого временного шага
            all_predictions = []
            for t in range(seq_len):
                xt = x[:, t, :]  # [batch_size, hidden_dim]

                hand_strength = self.hand_strength_head(xt)
                category_logits = self.category_head(xt)
                specific_hand = self.sigmoid(self.specific_hand_head(xt))

                all_predictions.append(
                    {
                        "hand_strength": hand_strength,
                        "category_probs": category_logits,
                        "specific_hand": specific_hand,
                    }
                )
            return all_predictions
        else:
            # Берем только последний временной шаг для предсказания
            x = x[:, -1, :]  # [batch_size, hidden_dim]

            # Многозадачный выход
            hand_strength = self.hand_strength_head(x)
            category_logits = self.category_head(x)
            specific_hand = self.sigmoid(self.specific_hand_head(x))

            return {
                "hand_strength": hand_strength,
                "category_probs": category_logits,
                "specific_hand": specific_hand,
            }


class SequenceHandRangeDataset(Dataset):
    def __init__(self, sequences, feature_columns, targets=None, max_length=20):
        """
        sequences: список DataFrame'ов, каждый - последовательность действий игрока
        feature_columns: список названий колонок с признаками
        targets: словарь с целевыми переменными для каждой последовательности
        max_length: максимальная длина последовательности (для padding)
        """
        self.sequences = sequences
        self.feature_columns = feature_columns
        self.max_length = max_length
        self.has_targets = targets is not None

        if self.has_targets:
            self.targets = targets

        # Предварительная обработка всех последовательностей
        self.processed_sequences = []
        self.valid_indices = []

        for idx, seq_df in enumerate(sequences):
            try:
                # Проверяем наличие всех необходимых колонок
                # Пробуем сначала со _scaled, затем без
                available_columns = []
                for col in feature_columns:
                    if f"{col}_scaled" in seq_df.columns:
                        available_columns.append(f"{col}_scaled")
                    elif col in seq_df.columns:
                        available_columns.append(col)
                    else:
                        # Колонка отсутствует - заполняем нулями
                        print(f"⚠️ Колонка {col} отсутствует в последовательности {idx}")

                if len(available_columns) == 0:
                    print(f"❌ Нет доступных признаков в последовательности {idx}")
                    continue

                # Извлекаем признаки
                features = seq_df[available_columns].values.astype(np.float32)

                # Дополняем недостающие колонки нулями
                if len(available_columns) < len(feature_columns):
                    # Создаем полную матрицу признаков
                    full_features = np.zeros(
                        (len(features), len(feature_columns)), dtype=np.float32
                    )

                    # Заполняем доступные признаки
                    for i, col in enumerate(feature_columns):
                        if f"{col}_scaled" in seq_df.columns:
                            full_features[:, i] = seq_df[f"{col}_scaled"].values
                        elif col in seq_df.columns:
                            full_features[:, i] = seq_df[col].values

                    features = full_features

                # Проверяем на валидность
                if not np.any(np.isnan(features)) and len(features) > 0:
                    self.processed_sequences.append(features)
                    self.valid_indices.append(idx)

            except Exception as e:
                print(f"⚠️ Ошибка обработки последовательности {idx}: {e}")
                continue

        print(
            f"📊 Dataset создан: {len(self.valid_indices)} валидных последовательностей из {len(sequences)}"
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        features = self.processed_sequences[idx]

        # Padding/truncation до max_length
        seq_len = min(len(features), self.max_length)

        # Создаем тензор с padding
        padded_features = np.zeros(
            (self.max_length, len(self.feature_columns)), dtype=np.float32
        )
        padded_features[:seq_len] = features[:seq_len]

        features_tensor = torch.tensor(padded_features, dtype=torch.float32)

        if self.has_targets:
            # Целевые переменные только для последнего элемента последовательности
            target_dict = {
                "hand_strength": torch.tensor(
                    self.targets["hand_strength"][actual_idx], dtype=torch.long
                ),
                "category_probs": torch.tensor(
                    self.targets["category_probs"][actual_idx], dtype=torch.float32
                ),
                "specific_hand": torch.tensor(
                    self.targets["specific_hand"][actual_idx], dtype=torch.float32
                ),
                "sequence_length": torch.tensor(seq_len, dtype=torch.long),
            }
            return features_tensor, target_dict
        else:
            return features_tensor, torch.tensor(seq_len, dtype=torch.long)


def create_sequences_with_targets(df, feature_columns, max_sequence_length=20):
    """
    Создает последовательности с соответствующими целевыми переменными
    """
    print(f"🎯 Создание последовательностей с целевыми переменными...")

    # Создаем последовательности
    sequences, sequence_info = create_player_sequences(
        df, max_sequence_length=max_sequence_length
    )

    # Создаем целевые переменные для каждой последовательности
    targets = {"hand_strength": [], "category_probs": [], "specific_hand": []}

    valid_sequences = []

    for i, (seq_df, seq_info) in enumerate(zip(sequences, sequence_info)):
        try:
            # Берем целевую переменную с последней записи в последовательности
            last_row = seq_df.iloc[-1]

            if pd.notna(last_row.get("hand_strength")):
                targets["hand_strength"].append(int(last_row["hand_strength"]))

                # Находим категорию
                category_idx = 8  # default "other"
                if pd.notna(last_row.get("hand_category")):
                    category_mapping = PokerHandAnalyzer.get_category_mapping()
                    category_idx = category_mapping.get(last_row["hand_category"], 8)

                category_probs = np.zeros(9)
                category_probs[category_idx] = 1.0
                targets["category_probs"].append(category_probs)

                # Создаем specific_hand на основе реальных карт
                specific_hand = np.zeros(13)
                analyzer = PokerHandAnalyzer()

                card1 = last_row.get("Showdown_1")
                card2 = last_row.get("Showdown_2")

                if pd.notna(card1) and pd.notna(card2):
                    rank1, _ = analyzer.parse_card(card1)
                    rank2, _ = analyzer.parse_card(card2)

                    if rank1 is not None and rank2 is not None:
                        rank1_idx = max(0, min(12, rank1 - 2))
                        rank2_idx = max(0, min(12, rank2 - 2))
                        specific_hand[rank1_idx] = 0.6
                        specific_hand[rank2_idx] = 0.6

                        # Нормализуем
                        if specific_hand.sum() > 0:
                            specific_hand = specific_hand / specific_hand.sum()
                        else:
                            specific_hand = np.ones(13) / 13
                    else:
                        specific_hand = np.ones(13) / 13
                else:
                    specific_hand = np.ones(13) / 13

                targets["specific_hand"].append(specific_hand)
                valid_sequences.append(seq_df)

        except Exception as e:
            # Пропускаем проблемные последовательности
            continue

    print(
        f"   ✅ Создано {len(valid_sequences)} последовательностей с целевыми переменными"
    )

    return valid_sequences, targets


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
# Добавьте эти методы в класс PokerHandAnalyzer


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
    def get_all_categories():
        """Возвращает список категорий для старой системы (9 категорий)"""
        return [
            "high_card",
            "pair",
            "two_pair",
            "three_of_kind",
            "straight",
            "flush",
            "full_house",
            "four_of_kind",
            "straight_flush",
        ]

    @staticmethod
    def get_category_mapping():
        """Возвращает маппинг категорий в индексы"""
        categories = PokerHandAnalyzer.get_all_categories()
        return {cat: i for i, cat in enumerate(categories)}

    def analyze_hand_strength(self, card1, card2):
        """
        Анализирует силу руки (для старой системы)
        Возвращает: (strength 0-9, category)
        """
        # Это заглушка для старой системы
        # В вашем случае используется HM3, так что этот метод не критичен

        rank1, suit1 = self.parse_card(card1)
        rank2, suit2 = self.parse_card(card2)

        if rank1 is None or rank2 is None:
            return 0, "other"

        # Простая логика для карманных карт
        if rank1 == rank2:  # Пара
            strength = min(9, 2 + (rank1 - 2) // 2)  # 2-9 в зависимости от ранга
            return strength, "pair"
        elif suit1 == suit2:  # Одномастные
            strength = min(9, 3 + max(rank1, rank2) // 3)
            return strength, "high_card"
        else:  # Разномастные
            strength = min(9, max(rank1, rank2) // 2)
            return strength, "high_card"


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


def create_target_variables(df, use_hm3_classification=True):
    """Создание целевых переменных с поддержкой HM3 классификации"""
    n_samples = len(df)

    if use_hm3_classification:
        # Для HM3: 5 классов силы
        num_strength_classes = 5

        # Получаем уникальные типы рук для категорий
        unique_hand_types = df["hand_type_hm3"].unique()
        category_mapping = {ht: i for i, ht in enumerate(sorted(unique_hand_types))}
        num_categories = len(unique_hand_types)

        print(f"📊 HM3 классификация:")
        print(f"   Классов силы: {num_strength_classes}")
        print(f"   Типов рук: {num_categories}")

    else:
        # Старая система
        num_strength_classes = 10
        category_mapping = PokerHandAnalyzer.get_category_mapping()
        num_categories = 9

    # Целевые переменные
    targets = {
        "hand_strength": df["hand_strength"].values.astype(int),
        "category_probs": np.zeros((n_samples, num_categories)),
        "specific_hand": np.zeros((n_samples, 13)),  # Ранги карт
    }

    # Кодирование категорий
    for i, row in df.iterrows():
        if use_hm3_classification:
            cat_idx = category_mapping.get(row["hand_type_hm3"], 0)
        else:
            cat_idx = category_mapping.get(row["hand_category"], 8)

        targets["category_probs"][i, cat_idx] = 1.0

        # Specific hand (ранги карт игрока)
        analyzer = PokerHandAnalyzer()
        try:
            card1 = row["Showdown_1"]
            card2 = row["Showdown_2"]

            if pd.notna(card1) and pd.notna(card2):
                rank1, _ = analyzer.parse_card(card1)
                rank2, _ = analyzer.parse_card(card2)

                if rank1 is not None and rank2 is not None:
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
        except Exception:
            targets["specific_hand"][i, :] = 1 / 13

    # Нормализуем specific_hand
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
    category_names = get_all_hm3_categories()
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


# 4. Дополнительная утилитная функция для безопасной конвертации в JSON
def convert_to_json_serializable(obj):
    """Конвертирует numpy/torch типы в стандартные Python типы для JSON"""
    import numpy as np

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


# 5. Обновленная функция setup_directories
def setup_directories():
    """Создание необходимых папок"""
    directories = [
        "models",
        "plots",
        "data",
        "data/combined",  # Подпапка для объединенных файлов
        "results",
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("📁 Создана структура папок:")
    for directory in directories:
        print(f"   ✅ {directory}/")


# 2. Функция find_data_files для игнорирования объединенных файлов
def find_data_files(data_dir="data"):
    """Поиск CSV файлов с данными (игнорируя объединенные)"""
    if not os.path.exists(data_dir):
        data_dir = "."  # Текущая папка если нет папки data

    # Ищем CSV файлы с покерными данными
    patterns = ["parsed_*.csv", "*poker*.csv", "*PDOM*.csv", "*.csv"]

    data_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(data_dir, pattern))
        # Фильтруем объединенные файлы
        files = [f for f in files if "combined" not in os.path.dirname(f)]
        data_files.extend(files)

    # Убираем дубликаты и сортируем
    data_files = sorted(list(set(data_files)))
    return data_files


def save_categories_json():
    """Сохранение JSON файла с HM3 категориями"""
    evaluator = PokerHandEvaluator()
    all_categories = get_all_hm3_categories()

    categories_info = {
        "classification_system": "HoldemManager3",
        "total_categories": len(all_categories),
        "categories": all_categories,
        "categories_by_strength": {
            str(strength): [
                cat
                for cat in all_categories
                if evaluator.hand_type_to_strength.get(cat, -1) == strength
            ]
            for strength in range(5)
        },
        "strength_levels": {
            "0": "Мусор/Дро (Trash/Draws)",
            "1": "Слабые руки (Weak hands)",
            "2": "Средние руки (Medium hands)",
            "3": "Сильные руки (Strong hands)",
            "4": "Монстры (Monster hands)",
        },
        "rank_names": PokerHandAnalyzer.get_rank_names(),
        "description": "HoldemManager3 система классификации с 73 типами рук",
    }

    os.makedirs("results", exist_ok=True)
    with open("results/hm3_poker_categories.json", "w", encoding="utf-8") as f:
        json.dump(categories_info, f, indent=2, ensure_ascii=False)

    print("💾 Сохранен файл с HM3 категориями: results/hm3_poker_categories.json")
    return categories_info


# 6. Обновленная функция choose_data_file для показа правильной структуры
def choose_data_file():
    """Выбор файла с данными"""
    data_files = find_data_files()

    if not data_files:
        print("❌ CSV файлы с данными не найдены!")
        print("Поместите CSV файл в папку 'data' или в текущую директорию")
        print("(Объединенные файлы должны быть в data/combined/)")
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
    print(f"\n💡 Объединенные файлы будут сохранены в data/combined/")

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

    # Создаем подпапку для объединенных файлов
    combined_dir = os.path.join("data", "combined")
    os.makedirs(combined_dir, exist_ok=True)

    # Сохраняем объединенный файл в подпапку
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_filename = os.path.join(
        combined_dir, f"combined_poker_data_{timestamp}.csv"
    )

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


def smart_train_val_test_split(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Комбинированное разделение: по игрокам + временной компонент
    """
    print(f"🎯 === УМНОЕ РАЗДЕЛЕНИЕ ДАННЫХ ===")

    # Проверяем наличие нужных колонок
    required_cols = ["PlayerID", "Timestamp", "HandID"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"⚠️  Отсутствуют колонки: {missing_cols}")
        print(f"📝 Создаем псевдо-колонки для разделения...")

        # Создаем псевдо PlayerID если его нет
        if "PlayerID" not in df.columns:
            # Используем комбинацию файла и места для создания игрока
            if "source_file" in df.columns and "Seat" in df.columns:
                df["PlayerID"] = (
                    df["source_file"].astype(str) + "_seat_" + df["Seat"].astype(str)
                )
            else:
                # В крайнем случае - просто группы строк
                df["PlayerID"] = (df.index // 50).astype(str)  # группы по 50 записей

        # Создаем псевдо Timestamp если его нет
        if "Timestamp" not in df.columns:
            if "Round" in df.columns:
                df["Timestamp"] = df["Round"]
            else:
                df["Timestamp"] = df.index

        # Создаем псевдо HandID если его нет
        if "HandID" not in df.columns:
            if "Round" in df.columns:
                df["HandID"] = df["Round"].astype(str) + "_" + df.index.astype(str)
            else:
                df["HandID"] = (df.index // 10).astype(str)  # группы по 10 записей

    # Анализ данных
    unique_players = df["PlayerID"].unique()
    total_players = len(unique_players)
    total_records = len(df)

    print(f"📊 Статистика данных:")
    print(f"   👥 Уникальных игроков: {total_players:,}")
    print(f"   📈 Общее количество записей: {total_records:,}")
    print(f"   📊 Записей на игрока: {total_records/total_players:.1f}")

    # Анализируем распределение записей по игрокам
    player_counts = df["PlayerID"].value_counts()
    print(f"   📋 Мин записей у игрока: {player_counts.min()}")
    print(f"   📋 Макс записей у игрока: {player_counts.max()}")
    print(f"   📋 Медиана записей: {player_counts.median():.0f}")

    # Фильтруем игроков с минимальным количеством записей
    min_records_per_player = 5  # минимум для нормального разделения
    good_players = player_counts[player_counts >= min_records_per_player].index

    if len(good_players) < total_players:
        filtered_out = total_players - len(good_players)
        print(
            f"   🚨 Отфильтровано {filtered_out} игроков с < {min_records_per_player} записей"
        )
        df = df[df["PlayerID"].isin(good_players)]
        unique_players = good_players
        total_players = len(unique_players)

    print(f"   ✅ Итого игроков для разделения: {total_players}")
    print(f"   ✅ Итого записей после фильтрации: {len(df):,}")

    # Разделяем игроков на группы
    np.random.seed(random_state)
    shuffled_players = np.random.permutation(unique_players)

    # Рассчитываем размеры групп
    test_players_count = max(1, int(total_players * test_size))
    val_players_count = max(1, int(total_players * val_size))
    train_players_count = total_players - test_players_count - val_players_count

    print(f"\n🎯 Разделение игроков:")
    print(
        f"   🎓 Train: {train_players_count} игроков ({train_players_count/total_players*100:.1f}%)"
    )
    print(
        f"   🔍 Validation: {val_players_count} игроков ({val_players_count/total_players*100:.1f}%)"
    )
    print(
        f"   🧪 Test: {test_players_count} игроков ({test_players_count/total_players*100:.1f}%)"
    )

    # Назначаем игроков в группы
    train_players = shuffled_players[:train_players_count]
    val_players = shuffled_players[
        train_players_count : train_players_count + val_players_count
    ]
    test_players = shuffled_players[train_players_count + val_players_count :]

    # Собираем данные
    train_data_list = []
    val_data_list = []
    test_data_list = []

    # Для train игроков: ранние записи в train, поздние в validation
    print(f"\n📅 Временное разделение внутри train игроков...")
    for player_id in train_players:
        player_data = df[df["PlayerID"] == player_id].copy()
        player_data = player_data.sort_values("Timestamp")

        # 80% ранних записей игрока в train, 20% поздних в validation
        split_idx = max(1, int(len(player_data) * 0.8))
        train_data_list.append(player_data.iloc[:split_idx])

        if len(player_data) > split_idx:
            val_data_list.append(player_data.iloc[split_idx:])

    # Для validation игроков: все записи в validation
    print(f"📝 Добавляем данные validation игроков...")
    for player_id in val_players:
        player_data = df[df["PlayerID"] == player_id].copy()
        val_data_list.append(player_data)

    # Для test игроков: все записи в test
    print(f"🧪 Добавляем данные test игроков...")
    for player_id in test_players:
        player_data = df[df["PlayerID"] == player_id].copy()
        test_data_list.append(player_data)

    # Объединяем данные
    train_df = (
        pd.concat(train_data_list, ignore_index=True)
        if train_data_list
        else pd.DataFrame()
    )
    val_df = (
        pd.concat(val_data_list, ignore_index=True) if val_data_list else pd.DataFrame()
    )
    test_df = (
        pd.concat(test_data_list, ignore_index=True)
        if test_data_list
        else pd.DataFrame()
    )

    # Проверяем результат
    print(f"\n✅ Результаты разделения:")
    print(f"   🎓 Train: {len(train_df):,} записей ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   🔍 Validation: {len(val_df):,} записей ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   🧪 Test: {len(test_df):,} записей ({len(test_df)/len(df)*100:.1f}%)")

    # Проверяем пересечения игроков
    train_player_set = (
        set(train_df["PlayerID"].unique()) if len(train_df) > 0 else set()
    )
    val_player_set = set(val_df["PlayerID"].unique()) if len(val_df) > 0 else set()
    test_player_set = set(test_df["PlayerID"].unique()) if len(test_df) > 0 else set()

    train_val_overlap = train_player_set & val_player_set
    train_test_overlap = train_player_set & test_player_set
    val_test_overlap = val_player_set & test_player_set

    print(f"\n🔍 Проверка пересечений игроков:")
    print(f"   Train ∩ Val: {len(train_val_overlap)} игроков (ожидается > 0)")
    print(f"   Train ∩ Test: {len(train_test_overlap)} игроков (должно быть 0)")
    print(f"   Val ∩ Test: {len(val_test_overlap)} игроков (должно быть 0)")

    if train_test_overlap or val_test_overlap:
        print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Есть пересечения игроков между train/val и test!")
    else:
        print(f"✅ Отлично! Нет утечек между train/val и test выборками")

    return train_df, val_df, test_df


def create_player_sequences(df, max_sequence_length=20, min_sequence_length=3):
    """
    Создание последовательностей действий игроков для RWKV
    """
    print(f"🔄 === СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ===")
    print(f"   📏 Максимальная длина последовательности: {max_sequence_length}")
    print(f"   📏 Минимальная длина последовательности: {min_sequence_length}")

    sequences = []
    sequence_info = []

    for player_id in df["PlayerID"].unique():
        player_data = df[df["PlayerID"] == player_id].copy()
        player_data = player_data.sort_values("Timestamp")

        # Создаем скользящие окна для каждого игрока
        for start_idx in range(len(player_data)):
            for end_idx in range(
                start_idx + min_sequence_length,
                min(start_idx + max_sequence_length + 1, len(player_data) + 1),
            ):

                sequence = player_data.iloc[start_idx:end_idx].copy()

                # Добавляем информацию о последовательности
                sequence_info.append(
                    {
                        "player_id": player_id,
                        "sequence_length": len(sequence),
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "start_time": sequence["Timestamp"].iloc[0],
                        "end_time": sequence["Timestamp"].iloc[-1],
                    }
                )

                sequences.append(sequence)

    print(f"   ✅ Создано {len(sequences):,} последовательностей")
    print(f"   📊 Средняя длина: {np.mean([len(seq) for seq in sequences]):.1f}")
    print(f"   📊 Игроков: {df['PlayerID'].nunique()}")

    return sequences, sequence_info


def prepare_sequence_hand_range_data_with_hm3(
    file_path,
    include_hole_cards=True,
    max_sequence_length="auto",
    balance_strategy="adaptive",
    use_hm3_classification=True,
):
    """
    Подготовка данных с HM3 классификацией рук (ИСПРАВЛЕННАЯ ВЕРСИЯ)
    """
    print(f"🎯 === ПОДГОТОВКА ДАННЫХ С ПОСЛЕДОВАТЕЛЬНОСТЯМИ ===")

    # Загрузка данных
    if isinstance(file_path, str):
        df = pd.read_csv(file_path)
    else:
        df = file_path

    print(f"📊 Загружено {len(df)} строк данных")

    # Фильтрация записей с известными картами
    mask = (df["Showdown_1"].notna()) & (df["Showdown_2"].notna())
    df_filtered = df[mask].copy().reset_index(drop=True)

    print(f"🎲 Найдено {len(df_filtered)} записей с открытыми картами")

    if len(df_filtered) == 0:
        print("❌ Нет записей с открытыми картами!")
        return None

    # ИСПРАВЛЕНИЕ: Выполняем анализ рук ТОЛЬКО ОДИН РАЗ
    if use_hm3_classification:
        print("🃏 Анализ рук по системе HM3 (73 типа)...")
        df_filtered = add_hand_evaluation_to_dataframe(df_filtered)

        # Используем HM3 классификацию
        df_filtered["hand_strength"] = df_filtered["hand_strength_class"].clip(0, 4)
        df_filtered["hand_category"] = df_filtered["hand_type_hm3"]

        # Проверяем диапазон значений
        print(
            f"   ✅ Диапазон силы руки: {df_filtered['hand_strength'].min()}-{df_filtered['hand_strength'].max()}"
        )
        print(f"   ✅ Количество классов: 5 (0-4)")
    else:
        # Старая система - преобразуем 0-9 в 0-4
        print("🔍 Анализ силы рук (старая система)...")
        analyzer = PokerHandAnalyzer()
        hand_analysis = df_filtered.apply(
            lambda row: analyzer.analyze_hand_strength(
                row["Showdown_1"], row["Showdown_2"]
            ),
            axis=1,
        )

        # Преобразуем 10 классов (0-9) в 5 классов (0-4)
        old_strength = [x[0] for x in hand_analysis]
        df_filtered["hand_strength"] = [
            min(4, s // 2) for s in old_strength
        ]  # 0-1 -> 0, 2-3 -> 1, etc.
        df_filtered["hand_category"] = [x[1] for x in hand_analysis]

        print(f"   ✅ Преобразование: 10 классов (0-9) → 5 классов (0-4)")

    # Стандартизация названий колонок
    column_mapping = {
        "PlayerId": "PlayerID",
        "Hand": "HandID",
        "StartDateUtc": "Timestamp",
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df_filtered.columns and new_col not in df_filtered.columns:
            df_filtered[new_col] = df_filtered[old_col]
            print(f"🔄 Переименована колонка: {old_col} → {new_col}")

    # Создание недостающих колонок
    if "PlayerID" not in df_filtered.columns:
        df_filtered["PlayerID"] = (
            df_filtered["Seat"].astype(str)
            + "_"
            + df_filtered.get("TournamentNumber", df_filtered.index // 100).astype(str)
        )

    if "Timestamp" not in df_filtered.columns:
        df_filtered["Timestamp"] = df_filtered.get("Round", df_filtered.index)

    if "HandID" not in df_filtered.columns:
        df_filtered["HandID"] = df_filtered.get("Hand", df_filtered.index // 10)

    # Вывод статистики ТОЛЬКО ОДИН РАЗ
    print(f"✅ Анализ завершен. Распределение силы рук:")
    strength_dist = df_filtered["hand_strength"].value_counts().sort_index()

    class_names = ["Мусор/Дро", "Слабые", "Средние", "Сильные", "Монстры"]
    for strength, count in strength_dist.items():
        if 0 <= strength < len(class_names):
            print(
                f"   {class_names[strength]}: {count} рук ({count/len(df_filtered)*100:.1f}%)"
            )

    # УДАЛЯЕМ дублирующий анализ рук!
    # Больше НЕ вызываем analyzer.analyze_hand_strength здесь

    # Умное разделение данных
    print(f"\n🎯 Разделение данных...")
    train_df, val_df, test_df = smart_train_val_test_split(df_filtered)

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

    # Добавляем карты стола
    analyzer = PokerHandAnalyzer()  # Создаем только для парсинга карт
    board_columns = ["Card1", "Card2", "Card3", "Card4", "Card5"]
    for col in board_columns:
        if col in df_filtered.columns:
            df_filtered[f"{col}_rank"], df_filtered[f"{col}_suit"] = zip(
                *df_filtered[col].apply(
                    lambda x: analyzer.parse_card(x) if pd.notna(x) else (0, 0)
                )
            )
            feature_columns.extend([f"{col}_rank", f"{col}_suit"])

    # Добавляем карты игрока ТОЛЬКО если нужно
    if include_hole_cards:
        print("🃏 Добавляем карты игрока в признаки...")
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

    # Кодирование категориальных переменных
    encoders = {}

    if "Position" in df_filtered.columns:
        encoders["position"] = LabelEncoder()
        df_filtered["Position_encoded"] = encoders["position"].fit_transform(
            df_filtered["Position"].fillna("Unknown")
        )
        feature_columns.append("Position_encoded")

    if "Action" in df_filtered.columns:
        encoders["action"] = LabelEncoder()
        df_filtered["Action_encoded"] = encoders["action"].fit_transform(
            df_filtered["Action"].fillna("Unknown")
        )
        feature_columns.append("Action_encoded")

    if "TypeBuyIn" in df_filtered.columns:
        encoders["buyin"] = LabelEncoder()
        df_filtered["TypeBuyIn_encoded"] = encoders["buyin"].fit_transform(
            df_filtered["TypeBuyIn"].fillna("Unknown")
        )
        feature_columns.append("TypeBuyIn_encoded")

    # Копируем все необходимые колонки в разделенные датафреймы
    for split_df in [train_df, val_df, test_df]:
        for col in df_filtered.columns:
            if col in feature_columns or col in [
                "hand_strength",
                "hand_category",
                "Showdown_1",
                "Showdown_2",
            ]:
                if col not in split_df.columns:
                    split_df[col] = df_filtered.loc[split_df.index, col]

    # Обработка проблемных значений
    print("🧹 Очистка данных...")
    for split_df in [train_df, val_df, test_df]:
        # SPR
        if "SPR" in split_df.columns:
            pot_zero_mask = (split_df["Pot"] == 0) | (split_df["Pot"].isnull())
            split_df.loc[pot_zero_mask, "SPR"] = 100.0

            spr_95th = split_df["SPR"].quantile(0.95)
            extreme_spr_mask = split_df["SPR"] > spr_95th * 2
            split_df.loc[extreme_spr_mask, "SPR"] = spr_95th

            spr_median = split_df["SPR"].median()
            split_df["SPR"] = split_df["SPR"].fillna(spr_median)

        # Остальные колонки
        for col in feature_columns:
            if col in split_df.columns:
                if split_df[col].dtype in ["int64", "float64"]:
                    split_df[col] = split_df[col].replace([np.inf, -np.inf], np.nan)
                    median_val = split_df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    split_df[col] = split_df[col].fillna(median_val)
                else:
                    split_df[col] = split_df[col].fillna(0)

    # Отбор доступных признаков
    available_features = [col for col in feature_columns if col in train_df.columns]
    print(f"📋 Используется {len(available_features)} признаков")

    # Нормализация признаков
    print("📊 Нормализация данных...")
    scaler = StandardScaler()

    # Обучаем scaler только на train данных
    scaler.fit(train_df[available_features])

    # Применяем ко всем выборкам
    train_scaled = pd.DataFrame(
        scaler.transform(train_df[available_features]),
        columns=available_features,
        index=train_df.index,
    )
    val_scaled = pd.DataFrame(
        scaler.transform(val_df[available_features]),
        columns=available_features,
        index=val_df.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df[available_features]),
        columns=available_features,
        index=test_df.index,
    )

    # Добавляем масштабированные данные обратно
    for col in available_features:
        train_df[f"{col}_scaled"] = train_scaled[col]
        val_df[f"{col}_scaled"] = val_scaled[col]
        test_df[f"{col}_scaled"] = test_scaled[col]

    # Создание последовательностей для каждой выборки
    print(f"\n🔄 Создание последовательностей...")

    # Автоматическое определение длины
    if max_sequence_length == "auto":
        sequence_params = analyze_optimal_sequence_length(df_filtered)
        max_sequence_length = sequence_params["max_length"]
        print(
            f"\n✅ Автоматически определена максимальная длина: {max_sequence_length}"
        )
    else:
        max_sequence_length = int(max_sequence_length)

    # Создаем последовательности с правильными признаками
    scaled_features = [f"{col}_scaled" for col in available_features]

    # Train последовательности
    train_sequences, _ = create_player_sequences(
        train_df, max_sequence_length=max_sequence_length
    )
    train_sequences_final, train_targets, num_categories = (
        create_sequences_with_targets_fixed(
            train_sequences, scaled_features, use_hm3=use_hm3_classification
        )
    )

    # Validation последовательности
    val_sequences, _ = create_player_sequences(
        val_df, max_sequence_length=max_sequence_length
    )
    val_sequences_final, val_targets, num_categories_val = (
        create_sequences_with_targets_fixed(
            val_sequences, scaled_features, use_hm3=use_hm3_classification
        )
    )

    # Test последовательности
    test_sequences, _ = create_player_sequences(
        test_df, max_sequence_length=max_sequence_length
    )
    test_sequences_final, test_targets, num_categories_test = (
        create_sequences_with_targets_fixed(
            test_sequences, scaled_features, use_hm3=use_hm3_classification
        )
    )

    # Проверка консистентности
    assert (
        num_categories == num_categories_val == num_categories_test
    ), f"Несоответствие категорий: train={num_categories}, val={num_categories_val}, test={num_categories_test}"

    print(f"\n✅ Все выборки имеют одинаковое количество категорий: {num_categories}")

    # Создание датасетов
    print("📦 Создание датасетов...")

    train_dataset = SequenceHandRangeDataset(
        train_sequences_final,
        available_features,
        train_targets,
        max_sequence_length,
    )
    val_dataset = SequenceHandRangeDataset(
        val_sequences_final, available_features, val_targets, max_sequence_length
    )
    test_dataset = SequenceHandRangeDataset(
        test_sequences_final,
        available_features,
        test_targets,
        max_sequence_length,
    )

    # DataLoaders
    batch_size = min(16, max(2, len(train_dataset) // 8))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"✅ Подготовка данных завершена:")
    print(f"   🎓 Train: {len(train_dataset)} последовательностей")
    print(f"   🔍 Validation: {len(val_dataset)} последовательностей")
    print(f"   🧪 Test: {len(test_dataset)} последовательностей")
    print(f"   📦 Размер батча: {batch_size}")
    print(f"   🏷️  Количество категорий HM3: {num_categories}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "scaler": scaler,
        "encoders": encoders,
        "feature_columns": available_features,
        "input_dim": len(available_features),
        "max_sequence_length": max_sequence_length,
        "include_hole_cards": include_hole_cards,
        "use_hm3": use_hm3_classification,
        "num_strength_classes": 5,  # Всегда 5 для HM3
        "num_categories": num_categories,  # 73 для HM3, 9 для старой системы
        "category_mapping": (
            category_mapping if "category_mapping" in locals() else None
        ),
    }


# 5. Дополнительная функция для сохранения маппинга категорий
def save_hm3_categories_mapping():
    """
    Сохраняет полный маппинг HM3 категорий для последующего использования
    """
    all_categories = get_all_hm3_categories()
    category_mapping = {cat: i for i, cat in enumerate(all_categories)}

    # Группируем по силе
    evaluator = PokerHandEvaluator()
    categories_by_strength = defaultdict(list)

    for cat in all_categories:
        strength = evaluator.hand_type_to_strength[cat]
        categories_by_strength[strength].append(cat)

    mapping_info = {
        "total_categories": len(all_categories),
        "category_to_index": category_mapping,
        "index_to_category": {i: cat for cat, i in category_mapping.items()},
        "categories_by_strength": dict(categories_by_strength),
        "strength_names": {
            0: "Мусор/Дро",
            1: "Слабые",
            2: "Средние",
            3: "Сильные",
            4: "Монстры",
        },
        "all_categories": all_categories,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/hm3_categories_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping_info, f, indent=2, ensure_ascii=False)

    print(
        "💾 Сохранен полный маппинг HM3 категорий: results/hm3_categories_mapping.json"
    )
    return mapping_info


# 1. Добавьте эту функцию для получения всех HM3 категорий
def get_all_hm3_categories():
    """
    Возвращает полный список всех возможных HM3 категорий (73 типа)
    """
    evaluator = PokerHandEvaluator()
    # Все категории из маппинга силы рук
    all_categories = list(evaluator.hand_type_to_strength.keys())
    # Сортируем для консистентности
    return sorted(all_categories)


# 2. Дополнительная проверка в create_sequences_with_targets_fixed
def create_sequences_with_targets_fixed(sequences, feature_columns, use_hm3=True):
    """
    Создает последовательности с фиксированным набором категорий (v2)
    """
    print(f"🎯 Создание {len(sequences)} последовательностей с целевыми переменными...")

    if use_hm3:
        # Используем ВСЕ возможные HM3 категории (73 типа)
        all_possible_categories = get_all_hm3_categories()
        category_mapping = {cat: i for i, cat in enumerate(all_possible_categories)}
        num_categories = len(all_possible_categories)  # Всегда 73

        print(f"   🏷️  Используется полный набор HM3 категорий: {num_categories}")
        print(
            f"   📋 Категории: {all_possible_categories[:10]}... (показаны первые 10)"
        )

        # Проверяем какие категории есть в данных
        found_categories = set()
        for seq_df in sequences:
            if "hand_category" in seq_df.columns:
                found_categories.update(seq_df["hand_category"].dropna().unique())

        print(
            f"   📊 Найдено в данных: {len(found_categories)} из {num_categories} категорий"
        )

        # Предупреждаем о неизвестных категориях
        unknown_categories = found_categories - set(all_possible_categories)
        if unknown_categories:
            print(f"   ⚠️  Неизвестные категории в данных: {unknown_categories}")
            print(f"      Они будут отображены на категорию 'Other'")
    else:
        category_mapping = PokerHandAnalyzer.get_category_mapping()
        num_categories = 9
        print(f"   🏷️  Используются стандартные 9 категорий")

    # Создаем целевые переменные
    targets = {"hand_strength": [], "category_probs": [], "specific_hand": []}
    valid_sequences = []
    analyzer = PokerHandAnalyzer()

    # Статистика по категориям
    category_counts = defaultdict(int)

    for seq_idx, seq_df in enumerate(sequences):
        try:
            if len(seq_df) == 0:
                continue

            last_row = seq_df.iloc[-1]

            # Проверяем наличие необходимых данных
            if not (
                pd.notna(last_row.get("hand_strength"))
                and pd.notna(last_row.get("Showdown_1"))
                and pd.notna(last_row.get("Showdown_2"))
            ):
                continue

            # Сила руки (0-4)
            strength = int(last_row["hand_strength"])
            if strength < 0 or strength > 4:
                print(f"   ⚠️  Seq {seq_idx}: неверная сила руки {strength}, пропускаем")
                continue

            targets["hand_strength"].append(strength)

            # Категория руки
            category_probs = np.zeros(num_categories)

            if (
                use_hm3
                and "hand_category" in last_row
                and pd.notna(last_row["hand_category"])
            ):
                category = last_row["hand_category"]
                # Если категория неизвестна, используем 'Other'
                if category not in category_mapping:
                    category = "Other"
                    if seq_idx < 5:  # Логируем только первые несколько
                        print(
                            f"   📝 Seq {seq_idx}: неизвестная категория '{last_row['hand_category']}' -> 'Other'"
                        )

                category_idx = category_mapping[category]
                category_counts[category] += 1
            else:
                # Для не-HM3 или отсутствующих категорий
                category = last_row.get("hand_category", "other")
                category_idx = category_mapping.get(
                    category, 8
                )  # 8 = 'other' в старой системе
                category_counts[category] += 1

            category_probs[category_idx] = 1.0
            targets["category_probs"].append(category_probs)

            # Конкретные ранги карт
            specific_hand = create_specific_hand_vector(
                last_row["Showdown_1"], last_row["Showdown_2"], analyzer
            )
            targets["specific_hand"].append(specific_hand)

            valid_sequences.append(seq_df)

        except Exception as e:
            print(f"   ⚠️  Ошибка в последовательности {seq_idx}: {e}")
            continue

    print(f"   ✅ Создано {len(valid_sequences)} валидных последовательностей")
    print(f"   🏷️  Фиксированная размерность category_probs: {num_categories}")

    # Показываем топ категорий
    if category_counts:
        top_categories = sorted(
            category_counts.items(), key=lambda x: x[1], reverse=True
        )[:10]
        print(f"   📊 Топ-10 категорий в данных:")
        for cat, count in top_categories:
            print(f"      {cat}: {count} ({count/len(valid_sequences)*100:.1f}%)")

    # Проверка диапазона значений силы
    if targets["hand_strength"]:
        min_strength = min(targets["hand_strength"])
        max_strength = max(targets["hand_strength"])
        print(f"   💪 Диапазон силы руки: {min_strength}-{max_strength}")

        # Распределение силы
        strength_dist = defaultdict(int)
        for s in targets["hand_strength"]:
            strength_dist[s] += 1
        print(f"   📊 Распределение силы:")
        for s in sorted(strength_dist.keys()):
            print(
                f"      {s}: {strength_dist[s]} ({strength_dist[s]/len(targets['hand_strength'])*100:.1f}%)"
            )

    return valid_sequences, targets, num_categories


# 3. Вспомогательная функция для создания вектора specific_hand
def create_specific_hand_vector(card1, card2, analyzer):
    """
    Создает вектор вероятностей для конкретных рангов карт
    """
    specific_hand = np.zeros(13)

    rank1, _ = analyzer.parse_card(card1)
    rank2, _ = analyzer.parse_card(card2)

    if rank1 is not None and rank2 is not None:
        # Индексы рангов (0-12 для 2-A)
        rank1_idx = max(0, min(12, rank1 - 2))
        rank2_idx = max(0, min(12, rank2 - 2))

        # Основные ранги
        specific_hand[rank1_idx] = 0.6
        specific_hand[rank2_idx] = 0.6

        # Добавляем вероятность на соседние ранги (блеф/полублеф)
        for offset in [-1, 1]:
            for rank_idx in [rank1_idx, rank2_idx]:
                neighbor_idx = rank_idx + offset
                if 0 <= neighbor_idx < 13:
                    specific_hand[neighbor_idx] = 0.1

        # Нормализация
        total = specific_hand.sum()
        if total > 0:
            specific_hand = specific_hand / total
        else:
            specific_hand = np.ones(13) / 13
    else:
        # Равномерное распределение если карты не распознаны
        specific_hand = np.ones(13) / 13

    return specific_hand


def create_player_sequences(df, max_sequence_length=20, min_sequence_length=3):
    """
    Создание последовательностей действий игроков с улучшенной логикой
    """
    print(f"🔄 Создание последовательностей игроков...")
    print(f"   📏 Длина: {min_sequence_length}-{max_sequence_length}")

    sequences = []
    sequence_info = []

    # Группируем по игрокам
    for player_id in df["PlayerID"].unique():
        player_data = df[df["PlayerID"] == player_id].copy()

        # Сортируем по времени
        if "Timestamp" in player_data.columns:
            player_data = player_data.sort_values("Timestamp")
        elif "Round" in player_data.columns:
            player_data = player_data.sort_values("Round")

        player_records = len(player_data)

        # Если у игрока слишком мало записей, создаем одну последовательность
        if player_records < min_sequence_length:
            if player_records > 0:
                sequences.append(player_data)
                sequence_info.append(
                    {
                        "player_id": player_id,
                        "sequence_length": player_records,
                        "type": "short_player_sequence",
                    }
                )
            continue

        # Для игроков с достаточным количеством записей создаем скользящие окна
        sequences_created = 0

        # Стратегия 1: Скользящие окна с шагом
        step_size = max(1, max_sequence_length // 4)  # Шаг в 1/4 от макс длины

        for start_idx in range(0, player_records - min_sequence_length + 1, step_size):
            end_idx = min(start_idx + max_sequence_length, player_records)

            if end_idx - start_idx >= min_sequence_length:
                sequence = player_data.iloc[start_idx:end_idx].copy()
                sequences.append(sequence)

                sequence_info.append(
                    {
                        "player_id": player_id,
                        "sequence_length": len(sequence),
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "type": "sliding_window",
                    }
                )

                sequences_created += 1

        # Стратегия 2: Если создали мало последовательностей, добавляем случайные сегменты
        if sequences_created < 3 and player_records >= max_sequence_length:
            for _ in range(2):  # Создаем еще 2 случайные последовательности
                start_idx = np.random.randint(
                    0, player_records - min_sequence_length + 1
                )
                seq_length = np.random.randint(
                    min_sequence_length,
                    min(max_sequence_length, player_records - start_idx) + 1,
                )
                end_idx = start_idx + seq_length

                sequence = player_data.iloc[start_idx:end_idx].copy()
                sequences.append(sequence)

                sequence_info.append(
                    {
                        "player_id": player_id,
                        "sequence_length": len(sequence),
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "type": "random_segment",
                    }
                )

    print(f"   ✅ Создано {len(sequences)} последовательностей")
    print(f"   📊 Игроков: {df['PlayerID'].nunique()}")
    print(f"   📊 Средняя длина: {np.mean([len(seq) for seq in sequences]):.1f}")

    # Статистика по типам последовательностей
    type_counts = {}
    for info in sequence_info:
        seq_type = info.get("type", "unknown")
        type_counts[seq_type] = type_counts.get(seq_type, 0) + 1

    print(f"   📋 По типам:")
    for seq_type, count in type_counts.items():
        print(f"      {seq_type}: {count}")

    return sequences, sequence_info


# 1. Обновленная функция train_sequence_hand_range_model
def train_sequence_hand_range_model(
    data_dict, hidden_dim=128, num_layers=3, epochs=25, lr=0.001
):
    """
    Обучение RWKV модели с поддержкой последовательностей (ИСПРАВЛЕННАЯ)
    """
    print(f"🚀 === ОБУЧЕНИЕ RWKV МОДЕЛИ С ПОСЛЕДОВАТЕЛЬНОСТЯМИ ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Устройство: {device}")

    # Размеры выборок
    train_size = len(data_dict["train_loader"].dataset)
    val_size = len(data_dict["val_loader"].dataset)
    test_size = len(data_dict["test_loader"].dataset)

    print(f"📊 Размеры выборок:")
    print(f"   🎓 Train: {train_size} последовательностей")
    print(f"   🔍 Validation: {val_size} последовательностей")
    print(f"   🧪 Test: {test_size} последовательностей")

    if train_size == 0:
        print("❌ Пустая обучающая выборка!")
        return None, None

    # ИСПРАВЛЕНИЕ: Получаем правильное количество категорий из data_dict
    num_categories = data_dict.get("num_categories", 73)
    num_strength_classes = data_dict.get("num_strength_classes", 5)

    print(f"📊 Параметры классификации:")
    print(f"   🏷️  Количество категорий: {num_categories}")
    print(f"   💪 Количество классов силы: {num_strength_classes}")

    # Создание модели с правильными параметрами
    model = SequenceHandRangeRWKV(
        input_dim=data_dict["input_dim"],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_sequence_length=data_dict["max_sequence_length"],
        num_strength_classes=num_strength_classes,  # Передаем из data_dict
        num_categories=num_categories,  # Передаем реальное количество категорий
    ).to(device)

    print(f"🧠 Модель создана:")
    print(f"   📥 Входных признаков: {data_dict['input_dim']}")
    print(f"   🧮 Скрытых нейронов: {hidden_dim}")
    print(f"   🏗️  Слоев RWKV: {num_layers}")
    print(f"   📏 Макс. длина последовательности: {data_dict['max_sequence_length']}")
    print(f"   🏷️  Выходных категорий: {num_categories}")
    print(f"   💪 Выходных классов силы: {num_strength_classes}")

    # Оптимизатор и планировщик
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=True
    )

    # Функции потерь
    strength_criterion = nn.CrossEntropyLoss()
    category_criterion = nn.BCEWithLogitsLoss()
    specific_criterion = nn.MSELoss()

    # История обучения
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_strength_acc": [],
        "val_strength_acc": [],
        "train_category_acc": [],
        "val_category_acc": [],
        "train_specific_mse": [],
        "val_specific_mse": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 8

    print(f"\n🎯 Начинаем обучение на {epochs} эпох...")

    for epoch in range(epochs):
        epoch_start_time = datetime.now()

        # ===================== ОБУЧЕНИЕ =====================
        model.train()
        train_losses = []
        train_strength_correct = 0
        train_strength_total = 0
        train_category_correct = 0
        train_category_total = 0
        train_specific_mse_sum = 0
        train_batches = 0

        print(f"\n📈 Эпоха {epoch+1}/{epochs} - Обучение...")

        for batch_idx, (inputs, targets) in enumerate(data_dict["train_loader"]):
            # Перемещаем данные на устройство
            inputs = inputs.to(device)  # [batch_size, seq_len, features]
            target_strength = targets["hand_strength"].to(device)
            target_category = targets["category_probs"].to(device)
            target_specific = targets["specific_hand"].to(device)
            seq_lengths = targets["sequence_length"].to(device)

            # ВАЖНО: Сброс состояния модели для каждого батча
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

            # Взвешенная общая потеря
            total_loss = strength_loss + 0.5 * category_loss + 0.3 * specific_loss

            # Обратный проход
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Метрики
            train_losses.append(total_loss.item())

            # Точность силы руки
            _, predicted_strength = torch.max(outputs["hand_strength"], 1)
            train_strength_total += target_strength.size(0)
            train_strength_correct += (
                (predicted_strength == target_strength).sum().item()
            )

            # Точность категории
            predicted_category = (
                torch.sigmoid(outputs["category_probs"]) > 0.5
            ).float()
            train_category_total += target_category.numel()
            train_category_correct += (
                (predicted_category == target_category).sum().item()
            )

            # MSE для конкретных рангов
            train_specific_mse_sum += specific_loss.item()
            train_batches += 1

            # Прогресс
            if (batch_idx + 1) % max(1, len(data_dict["train_loader"]) // 10) == 0:
                progress = (batch_idx + 1) / len(data_dict["train_loader"]) * 100
                current_loss = total_loss.item()
                print(f"   📊 {progress:5.1f}% | Loss: {current_loss:.4f}")

        # ===================== ВАЛИДАЦИЯ =====================
        model.eval()
        val_losses = []
        val_strength_correct = 0
        val_strength_total = 0
        val_category_correct = 0
        val_category_total = 0
        val_specific_mse_sum = 0
        val_batches = 0

        print(f"🔍 Валидация...")

        with torch.no_grad():
            for inputs, targets in data_dict["val_loader"]:
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

                # Метрики
                _, predicted_strength = torch.max(outputs["hand_strength"], 1)
                val_strength_total += target_strength.size(0)
                val_strength_correct += (
                    (predicted_strength == target_strength).sum().item()
                )

                predicted_category = (
                    torch.sigmoid(outputs["category_probs"]) > 0.5
                ).float()
                val_category_total += target_category.numel()
                val_category_correct += (
                    (predicted_category == target_category).sum().item()
                )

                val_specific_mse_sum += specific_loss.item()
                val_batches += 1

        # ===================== МЕТРИКИ ЭПОХИ =====================
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        avg_val_loss = np.mean(val_losses) if val_losses else 0

        train_strength_acc = train_strength_correct / max(train_strength_total, 1)
        val_strength_acc = val_strength_correct / max(val_strength_total, 1)

        train_category_acc = train_category_correct / max(train_category_total, 1)
        val_category_acc = val_category_correct / max(val_category_total, 1)

        train_specific_mse = train_specific_mse_sum / max(train_batches, 1)
        val_specific_mse = val_specific_mse_sum / max(val_batches, 1)

        current_lr = optimizer.param_groups[0]["lr"]

        # Сохранение в историю
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_strength_acc"].append(train_strength_acc)
        history["val_strength_acc"].append(val_strength_acc)
        history["train_category_acc"].append(train_category_acc)
        history["val_category_acc"].append(val_category_acc)
        history["train_specific_mse"].append(train_specific_mse)
        history["val_specific_mse"].append(val_specific_mse)
        history["learning_rates"].append(current_lr)

        # Время эпохи
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()

        # Вывод результатов эпохи
        print(f"\n📊 Эпоха {epoch+1}/{epochs} завершена за {epoch_time:.1f}с:")
        print(f"   📉 Потери: train={avg_train_loss:.4f}, val={avg_val_loss:.4f}")
        print(
            f"   🎯 Точность силы: train={train_strength_acc:.3f}, val={val_strength_acc:.3f}"
        )
        print(
            f"   🏷️  Точность категории: train={train_category_acc:.3f}, val={val_category_acc:.3f}"
        )
        print(
            f"   📊 MSE рангов: train={train_specific_mse:.4f}, val={val_specific_mse:.4f}"
        )
        print(f"   📈 Learning rate: {current_lr:.6f}")

        # Обновление планировщика
        scheduler.step(avg_val_loss)

        # Ранняя остановка и сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"   ⭐ Новая лучшая модель! Val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{early_stopping_patience}")

        # Проверка ранней остановки
        if patience_counter >= early_stopping_patience:
            print(f"\n🛑 Ранняя остановка на эпохе {epoch+1}")
            print(f"   📈 Лучший val loss: {best_val_loss:.4f}")
            break

    # Загружаем лучшие веса
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Загружены веса лучшей модели (val loss: {best_val_loss:.4f})")

    print(f"\n🎉 Обучение завершено!")
    return model, history


# 2. Исправленная функция evaluate_sequence_model_performance
def evaluate_sequence_model_performance(model, data_dict, include_hole_cards=True):
    """
    Детальная оценка производительности последовательной модели
    """
    print(f"\n🔍 === ОЦЕНКА МОДЕЛИ ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_outputs = {
        "strength_pred": [],
        "strength_true": [],
        "category_pred": [],
        "category_true": [],
        "specific_pred": [],
        "specific_true": [],
        "sequence_lengths": [],
    }

    total_sequences = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, targets in data_dict["test_loader"]:
            inputs = inputs.to(device)
            seq_lengths = targets["sequence_length"]

            model.reset_states()
            outputs = model(inputs)

            # Собираем предсказания
            _, strength_pred = torch.max(outputs["hand_strength"], 1)
            all_outputs["strength_pred"].extend(strength_pred.cpu().numpy())
            all_outputs["strength_true"].extend(targets["hand_strength"].numpy())

            # Категории
            category_pred = torch.sigmoid(outputs["category_probs"]).cpu().numpy()
            category_true = targets["category_probs"].numpy()
            all_outputs["category_pred"].extend(np.argmax(category_pred, axis=1))
            all_outputs["category_true"].extend(np.argmax(category_true, axis=1))

            # Конкретные ранги
            all_outputs["specific_pred"].extend(outputs["specific_hand"].cpu().numpy())
            all_outputs["specific_true"].extend(targets["specific_hand"].numpy())

            # Длины последовательностей
            all_outputs["sequence_lengths"].extend(seq_lengths.numpy())

            total_sequences += len(strength_pred)
            correct_predictions += (
                (strength_pred.cpu() == targets["hand_strength"]).sum().item()
            )

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

    model_type = "с картами игрока" if include_hole_cards else "без карт игрока"
    print(f"📊 Результаты модели {model_type}:")
    print(f"   🎯 Точность силы руки: {strength_accuracy:.3f}")
    print(f"   🏷️  Точность категории: {category_accuracy:.3f}")
    print(f"   📊 MSE рангов карт: {specific_mse:.4f}")
    print(f"   📈 Всего последовательностей: {total_sequences}")

    # Анализ по длине последовательностей
    seq_lengths = np.array(all_outputs["sequence_lengths"])
    strength_preds = np.array(all_outputs["strength_pred"])
    strength_true = np.array(all_outputs["strength_true"])

    print(f"\n📏 Анализ по длине последовательностей:")
    for seq_len in sorted(set(seq_lengths)):
        mask = seq_lengths == seq_len
        if mask.sum() > 0:
            acc = accuracy_score(strength_true[mask], strength_preds[mask])
            count = mask.sum()
            print(f"   Длина {seq_len:2d}: {acc:.3f} точность ({count:3d} примеров)")

    # ИСПРАВЛЕНИЕ: Не создаем детальный отчет по категориям для HM3
    # так как sklearn не может обработать 73 категории корректно
    if data_dict.get("use_hm3", True):
        print(f"\n🏷️  Статистика по категориям:")
        unique_true = np.unique(all_outputs["category_true"])
        unique_pred = np.unique(all_outputs["category_pred"])
        print(f"   Уникальных истинных категорий: {len(unique_true)}")
        print(f"   Уникальных предсказанных категорий: {len(unique_pred)}")
    else:
        # Старая система с 9 категориями
        category_names = PokerHandAnalyzer.get_all_categories()
        print(f"\n🏷️  Отчет по категориям рук:")
        try:
            print(
                classification_report(
                    all_outputs["category_true"],
                    all_outputs["category_pred"],
                    target_names=category_names,
                    zero_division=0,
                )
            )
        except Exception as e:
            print(f"   ⚠️  Не удалось создать детальный отчет: {e}")

    return {
        "strength_accuracy": strength_accuracy,
        "category_accuracy": category_accuracy,
        "specific_mse": specific_mse,
        "total_sequences": total_sequences,
        **all_outputs,
    }


def visualize_sequence_results(model, data_dict, history, include_hole_cards=True):
    """
    Визуализация результатов обучения последовательной модели
    """
    print(f"📊 Создание графиков...")

    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    # 1. Потери обучения
    axes[0, 0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0, 0].plot(history["val_loss"], label="Validation", linewidth=2)
    axes[0, 0].set_title("Потери обучения", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Эпоха")
    axes[0, 0].set_ylabel("Потери")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Точность силы руки
    axes[0, 1].plot(history["train_strength_acc"], label="Train", linewidth=2)
    axes[0, 1].plot(history["val_strength_acc"], label="Validation", linewidth=2)
    axes[0, 1].set_title(
        "Точность предсказания силы руки", fontsize=14, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Эпоха")
    axes[0, 1].set_ylabel("Точность")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Точность категорий
    axes[1, 0].plot(history["train_category_acc"], label="Train", linewidth=2)
    axes[1, 0].plot(history["val_category_acc"], label="Validation", linewidth=2)
    axes[1, 0].set_title(
        "Точность предсказания категорий", fontsize=14, fontweight="bold"
    )
    axes[1, 0].set_xlabel("Эпоха")
    axes[1, 0].set_ylabel("Точность")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. MSE конкретных рангов
    axes[1, 1].plot(history["train_specific_mse"], label="Train", linewidth=2)
    axes[1, 1].plot(history["val_specific_mse"], label="Validation", linewidth=2)
    axes[1, 1].set_title("MSE предсказания рангов карт", fontsize=14, fontweight="bold")
    axes[1, 1].set_xlabel("Эпоха")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Learning Rate
    axes[2, 0].plot(history["learning_rates"], linewidth=2, color="red")
    axes[2, 0].set_title("Learning Rate", fontsize=14, fontweight="bold")
    axes[2, 0].set_xlabel("Эпоха")
    axes[2, 0].set_ylabel("Learning Rate")
    axes[2, 0].set_yscale("log")
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Матрица ошибок для силы руки
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

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(all_targets, all_predictions)

    im = axes[2, 1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[2, 1].set_title("Матрица ошибок: Сила руки", fontsize=14, fontweight="bold")
    axes[2, 1].set_xlabel("Предсказанная сила")
    axes[2, 1].set_ylabel("Истинная сила")

    # Добавляем числа в матрицу
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2, 1].text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    # Общий заголовок
    model_type = "с картами игрока" if include_hole_cards else "без карт игрока"
    fig.suptitle(
        f"Результаты обучения RWKV модели ({model_type})",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    # Сохранение
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_name = f"sequence_training_results_{'with_cards' if include_hole_cards else 'without_cards'}_{timestamp}.png"

    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", plot_name)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"📊 График сохранен: {plot_path}")

    return fig, plot_path


# 1. Исправленная функция predict_sequence_hand_ranges
def predict_sequence_hand_ranges(model, data_dict, sample_hands=5):
    """
    Демонстрация предсказаний последовательной модели
    """
    print(f"\n🎲 === ПРИМЕРЫ ПРЕДСКАЗАНИЙ ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # ИСПРАВЛЕНИЕ: Получаем правильный список категорий
    if data_dict.get("use_hm3", True):
        # Для HM3 используем полный список 73 категорий
        category_names = get_all_hm3_categories()
    else:
        # Для старой системы
        category_names = PokerHandAnalyzer.get_all_categories()

    rank_names = PokerHandAnalyzer.get_rank_names()

    shown_examples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_dict["test_loader"]):
            if shown_examples >= sample_hands:
                break

            inputs = inputs.to(device)
            seq_lengths = targets["sequence_length"]

            model.reset_states()
            outputs = model(inputs)

            batch_size = inputs.size(0)

            for sample_idx in range(min(batch_size, sample_hands - shown_examples)):
                # Длина последовательности
                seq_len = seq_lengths[sample_idx].item()

                # Предсказания
                strength_probs = torch.softmax(outputs["hand_strength"], dim=1)[
                    sample_idx
                ]
                predicted_strength = torch.argmax(strength_probs).item()
                strength_confidence = strength_probs[predicted_strength].item()

                category_probs = torch.sigmoid(outputs["category_probs"])[sample_idx]
                predicted_category = torch.argmax(category_probs).item()
                category_confidence = category_probs[predicted_category].item()

                specific_probs = outputs["specific_hand"][sample_idx]

                # Истинные значения
                true_strength = targets["hand_strength"][sample_idx].item()
                true_category = torch.argmax(
                    targets["category_probs"][sample_idx]
                ).item()

                print(f"\n🎯 Пример {shown_examples + 1}:")
                print(f"   📏 Длина последовательности: {seq_len}")
                print(f"   💪 Истинная сила: {true_strength}")
                print(
                    f"   🎯 Предсказанная сила: {predicted_strength} (уверенность: {strength_confidence:.3f})"
                )

                # ИСПРАВЛЕНИЕ: Проверяем границы индексов
                if 0 <= true_category < len(category_names):
                    print(f"   🏷️  Истинная категория: {category_names[true_category]}")
                else:
                    print(
                        f"   🏷️  Истинная категория: индекс {true_category} (вне диапазона)"
                    )

                if 0 <= predicted_category < len(category_names):
                    print(
                        f"   🏷️  Предсказанная категория: {category_names[predicted_category]} (вероятность: {category_confidence:.3f})"
                    )
                else:
                    print(
                        f"   🏷️  Предсказанная категория: индекс {predicted_category} (вне диапазона)"
                    )

                # Топ-3 наиболее вероятных ранга
                top_ranks = torch.topk(specific_probs, 3)
                print(f"   🃏 Топ-3 вероятных ранга:")
                for j, (prob, rank_idx) in enumerate(
                    zip(top_ranks.values, top_ranks.indices)
                ):
                    print(f"      {j+1}. {rank_names[rank_idx]}: {prob:.3f}")

                # Анализ правильности
                strength_correct = "✅" if predicted_strength == true_strength else "❌"
                category_correct = "✅" if predicted_category == true_category else "❌"
                print(
                    f"   📊 Результат: сила {strength_correct}, категория {category_correct}"
                )

                shown_examples += 1

                if shown_examples >= sample_hands:
                    break

    print(f"\n💡 Показано {shown_examples} примеров предсказаний")


def analyze_optimal_sequence_length(df, percentile=95):
    """
    Анализирует данные и определяет оптимальную длину последовательности

    Args:
        df: DataFrame с данными
        percentile: процентиль для определения максимальной длины (по умолчанию 95%)

    Returns:
        dict с рекомендациями по длине последовательности
    """
    print(f"📊 === АНАЛИЗ ОПТИМАЛЬНОЙ ДЛИНЫ ПОСЛЕДОВАТЕЛЬНОСТИ ===")

    # Группируем по игрокам и рукам
    if "HandID" in df.columns:
        hand_lengths = df.groupby("HandID").size()
        print(f"📋 Анализ по рукам (HandID):")
        print(f"   Количество уникальных рук: {len(hand_lengths):,}")
        print(f"   Средняя длина руки: {hand_lengths.mean():.1f} действий")
        print(f"   Медиана: {hand_lengths.median():.0f}")
        print(f"   Мин/Макс: {hand_lengths.min()}/{hand_lengths.max()}")

        # Распределение длин
        print(f"\n   Распределение длин рук:")
        for pct in [50, 75, 90, 95, 99]:
            pct_value = hand_lengths.quantile(pct / 100)
            print(f"   {pct}%: {pct_value:.0f} действий")

    # Анализ по игрокам и временным сессиям
    player_sequences = []

    for player_id in df["PlayerID"].unique():
        player_data = df[df["PlayerID"] == player_id].copy()

        # Сортируем по времени
        if "Timestamp" in player_data.columns:
            player_data = player_data.sort_values("Timestamp")

        # Определяем сессии (группы действий с разрывом < threshold)
        if "Timestamp" in player_data.columns and player_data["Timestamp"].dtype in [
            "int64",
            "float64",
        ]:
            time_diffs = player_data["Timestamp"].diff()

            # Пороговое значение для разделения сессий (например, 30 минут или 30 раундов)
            session_threshold = time_diffs.quantile(0.9) if len(time_diffs) > 10 else 30

            # Создаем группы сессий
            session_breaks = (time_diffs > session_threshold).cumsum()
            sessions = player_data.groupby(session_breaks).size()

            player_sequences.extend(sessions.tolist())
        else:
            # Если нет временных меток, берем всю историю игрока как одну последовательность
            player_sequences.append(len(player_data))

    if player_sequences:
        player_sequences = pd.Series(player_sequences)
        print(f"\n📊 Анализ последовательностей игроков:")
        print(f"   Найдено последовательностей: {len(player_sequences):,}")
        print(f"   Средняя длина: {player_sequences.mean():.1f}")
        print(f"   Медиана: {player_sequences.median():.0f}")
        print(f"   Стд. отклонение: {player_sequences.std():.1f}")

        # Определяем оптимальные параметры
        optimal_max = int(player_sequences.quantile(percentile / 100))
        optimal_min = max(3, int(player_sequences.quantile(0.1)))  # Минимум 3 действия
        recommended = int(
            player_sequences.quantile(0.75)
        )  # 75 процентиль как рекомендация

        print(f"\n🎯 Рекомендации:")
        print(f"   Минимальная длина: {optimal_min}")
        print(f"   Рекомендуемая длина: {recommended}")
        print(f"   Максимальная длина: {optimal_max} ({percentile}% покрытие)")

        # Анализ покрытия данных
        coverage_80 = (
            (player_sequences <= recommended).sum() / len(player_sequences) * 100
        )
        coverage_95 = (
            (player_sequences <= optimal_max).sum() / len(player_sequences) * 100
        )

        print(f"\n📈 Покрытие данных:")
        print(f"   При длине {recommended}: {coverage_80:.1f}% последовательностей")
        print(f"   При длине {optimal_max}: {coverage_95:.1f}% последовательностей")

        # Анализ потерь при обрезке
        truncated_actions = (
            player_sequences[player_sequences > optimal_max].sum()
            - len(player_sequences[player_sequences > optimal_max]) * optimal_max
        )
        total_actions = player_sequences.sum()
        loss_percentage = truncated_actions / total_actions * 100

        print(f"\n✂️ Потери при обрезке до {optimal_max}:")
        print(f"   Обрезанных действий: {truncated_actions:,}")
        print(f"   Процент потерь: {loss_percentage:.2f}%")

        return {
            "min_length": optimal_min,
            "recommended_length": recommended,
            "max_length": optimal_max,
            "mean_length": player_sequences.mean(),
            "median_length": player_sequences.median(),
            "coverage_at_max": coverage_95,
            "data_loss_percentage": loss_percentage,
            "total_sequences": len(player_sequences),
        }

    else:
        print("⚠️ Не удалось проанализировать последовательности")
        # Возвращаем значения по умолчанию
        return {
            "min_length": 3,
            "recommended_length": 10,
            "max_length": 20,
            "mean_length": 10,
            "median_length": 10,
            "coverage_at_max": 100,
            "data_loss_percentage": 0,
            "total_sequences": 0,
        }


def create_adaptive_sequences(df, sequence_params, balance_strategy="adaptive"):
    """
    Создает последовательности с адаптивными параметрами

    Args:
        df: DataFrame с данными
        sequence_params: параметры из analyze_optimal_sequence_length
        balance_strategy: стратегия балансировки ('adaptive', 'fixed', 'mixed')
    """
    print(f"\n🔄 === СОЗДАНИЕ АДАПТИВНЫХ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ===")
    print(f"📏 Параметры:")
    print(f"   Минимум: {sequence_params['min_length']}")
    print(f"   Рекомендуемая: {sequence_params['recommended_length']}")
    print(f"   Максимум: {sequence_params['max_length']}")
    print(f"   Стратегия: {balance_strategy}")

    sequences = []
    sequence_info = []

    for player_id in df["PlayerID"].unique():
        player_data = df[df["PlayerID"] == player_id].copy()

        if "Timestamp" in player_data.columns:
            player_data = player_data.sort_values("Timestamp")

        player_records = len(player_data)

        if balance_strategy == "adaptive":
            # Адаптивная стратегия: разные длины для разных игроков
            if player_records < sequence_params["min_length"]:
                continue

            # Короткие последовательности для игроков с мало данных
            if player_records < sequence_params["recommended_length"]:
                sequences.append(player_data)
                sequence_info.append(
                    {
                        "player_id": player_id,
                        "length": player_records,
                        "type": "short_full",
                    }
                )

            # Средние последовательности
            elif player_records < sequence_params["max_length"]:
                # Создаем перекрывающиеся окна
                step = max(1, player_records // 3)
                for start in range(
                    0, player_records - sequence_params["min_length"] + 1, step
                ):
                    end = min(
                        start + sequence_params["recommended_length"], player_records
                    )
                    sequences.append(player_data.iloc[start:end])
                    sequence_info.append(
                        {
                            "player_id": player_id,
                            "length": end - start,
                            "type": "medium_window",
                        }
                    )

            # Длинные последовательности
            else:
                # Используем скользящее окно с адаптивным шагом
                window_size = sequence_params["recommended_length"]
                step = max(1, window_size // 2)

                for start in range(0, player_records - window_size + 1, step):
                    end = start + window_size
                    sequences.append(player_data.iloc[start:end])
                    sequence_info.append(
                        {
                            "player_id": player_id,
                            "length": window_size,
                            "type": "sliding_window",
                        }
                    )

                # Добавляем последнюю последовательность полной длины
                if player_records > sequence_params["max_length"]:
                    sequences.append(player_data.iloc[-sequence_params["max_length"] :])
                    sequence_info.append(
                        {
                            "player_id": player_id,
                            "length": sequence_params["max_length"],
                            "type": "tail_max",
                        }
                    )

        elif balance_strategy == "mixed":
            # Смешанная стратегия: комбинация разных длин
            if player_records >= sequence_params["min_length"]:
                # Короткие последовательности (для начала игры)
                for length in [5, 10, 15]:
                    if length <= player_records:
                        sequences.append(player_data.iloc[:length])
                        sequence_info.append(
                            {
                                "player_id": player_id,
                                "length": length,
                                "type": f"fixed_{length}",
                            }
                        )

                # Случайные сегменты
                if player_records > sequence_params["recommended_length"]:
                    for _ in range(2):
                        start = np.random.randint(
                            0, player_records - sequence_params["min_length"]
                        )
                        length = np.random.randint(
                            sequence_params["min_length"],
                            min(
                                sequence_params["recommended_length"],
                                player_records - start,
                            ),
                        )
                        sequences.append(player_data.iloc[start : start + length])
                        sequence_info.append(
                            {
                                "player_id": player_id,
                                "length": length,
                                "type": "random_segment",
                            }
                        )

    # Статистика
    lengths = [info["length"] for info in sequence_info]
    type_counts = {}
    for info in sequence_info:
        seq_type = info["type"]
        type_counts[seq_type] = type_counts.get(seq_type, 0) + 1

    print(f"\n📊 Создано последовательностей: {len(sequences)}")
    print(f"   Средняя длина: {np.mean(lengths):.1f}")
    print(f"   Распределение длин:")
    print(
        f"   - Короткие (<{sequence_params['recommended_length']}): "
        f"{sum(1 for l in lengths if l < sequence_params['recommended_length'])}"
    )
    print(
        f"   - Средние: "
        f"{sum(1 for l in lengths if sequence_params['recommended_length'] <= l < sequence_params['max_length'])}"
    )
    print(
        f"   - Максимальные (={sequence_params['max_length']}): "
        f"{sum(1 for l in lengths if l == sequence_params['max_length'])}"
    )

    print(f"\n📋 По типам создания:")
    for seq_type, count in sorted(
        type_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"   {seq_type}: {count}")

    return sequences, sequence_info, sequence_params


#############################################################################################################################
### NEW CODE ###
############################################################################################################################


# ===================== НОВЫЙ ПОДХОД К ДАННЫМ =====================





def prepare_hand_based_sequences(file_path, include_hole_cards=True, use_hm3=True):
    """
    Подготовка данных с разделением по рукам (новый подход)
    """
    print(f"🎯 === ПОДГОТОВКА ДАННЫХ (РАЗДЕЛЕНИЕ ПО РУКАМ) ===")
    
    # Загрузка данных
    df = pd.read_csv(file_path) if isinstance(file_path, str) else file_path
    print(f"📊 Загружено {len(df)} записей")
    
    # Фильтрация записей с шоудауном
    mask = (df['Showdown_1'].notna()) & (df['Showdown_2'].notna())
    df_filtered = df[mask].copy().reset_index(drop=True)
    
    print(f"🃏 Найдено {len(df_filtered)} записей с открытыми картами")
    
    # Добавляем HM3 классификацию
    if use_hm3:
        print("🔍 Анализ рук по системе HM3...")
        df_filtered = add_hand_evaluation_to_dataframe(df_filtered)
    
    # Добавляем контекст игроков
    df_filtered = add_player_context(df_filtered)
    
    # Добавляем контекст турнира
    df_filtered = add_tournament_context(df_filtered)
    
    # Разделение по рукам, а не по игрокам!
    train_df, val_df, test_df = split_by_hands(df_filtered)
    
    return create_hand_sequences_dataset(train_df, val_df, test_df, include_hole_cards)


def split_by_hands(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Разделение данных по уникальным рукам
    """
    print(f"\n🎲 === РАЗДЕЛЕНИЕ ПО РУКАМ ===")
    
    # Получаем уникальные руки
    unique_hands = df['HandID'].unique() if 'HandID' in df.columns else df['Hand'].unique()
    n_hands = len(unique_hands)
    
    print(f"📊 Статистика:")
    print(f"   🃏 Уникальных рук: {n_hands:,}")
    print(f"   📈 Всего записей: {len(df):,}")
    
    # Анализ распределения
    hand_counts = df.groupby('HandID' if 'HandID' in df.columns else 'Hand').size()
    print(f"   📊 Действий на руку: мин={hand_counts.min()}, макс={hand_counts.max()}, среднее={hand_counts.mean():.1f}")
    
    # Перемешиваем руки
    np.random.seed(random_state)
    shuffled_hands = np.random.permutation(unique_hands)
    
    # Разделяем
    n_test = int(n_hands * test_size)
    n_val = int(n_hands * val_size)
    
    test_hands = shuffled_hands[:n_test]
    val_hands = shuffled_hands[n_test:n_test + n_val]
    train_hands = shuffled_hands[n_test + n_val:]
    
    # Создаем выборки
    hand_col = 'HandID' if 'HandID' in df.columns else 'Hand'
    train_df = df[df[hand_col].isin(train_hands)].copy()
    val_df = df[df[hand_col].isin(val_hands)].copy()
    test_df = df[df[hand_col].isin(test_hands)].copy()
    
    print(f"\n✅ Результаты разделения:")
    print(f"   🎓 Train: {len(train_hands):,} рук ({len(train_df):,} записей)")
    print(f"   🔍 Val: {len(val_hands):,} рук ({len(val_df):,} записей)")
    print(f"   🧪 Test: {len(test_hands):,} рук ({len(test_df):,} записей)")
    
    # Проверка пересечений игроков (это нормально!)
    players_train = set(train_df['PlayerID'].unique() if 'PlayerID' in train_df else [])
    players_test = set(test_df['PlayerID'].unique() if 'PlayerID' in test_df else [])
    overlap = len(players_train & players_test)
    
    print(f"\n🔍 Пересечение игроков train/test: {overlap}")
    print(f"   ✅ Это нормально! Модель учится на паттернах рук, а не запоминает игроков")
    
    return train_df, val_df, test_df


def add_player_context(df):
    """
    Добавляет исторический контекст игроков как признаки
    """
    print(f"👥 Добавление контекста игроков...")
    
    # Статистика по игрокам за всю историю
    player_stats = df.groupby('PlayerID').agg({
        'Bet': ['mean', 'std', 'count'],
        'Allin': 'mean',
        'PlayerWins': 'mean',
        'WinAmount': 'mean'
    })
    
    player_stats.columns = [
        'player_avg_bet', 'player_bet_std', 'player_hands_count',
        'player_allin_rate', 'player_win_rate', 'player_avg_win'
    ]
    
    # Добавляем к основному датафрейму
    df = df.merge(player_stats, left_on='PlayerID', right_index=True, how='left')
    
    # Скользящие статистики последних N рук
    for window in [5, 10, 20]:
        df[f'player_win_rate_last{window}'] = (
            df.groupby('PlayerID')['PlayerWins']
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
    
    return df


def add_tournament_context(df):
    """
    Добавляет контекст турнира/сессии
    """
    print(f"🏆 Добавление контекста турнира...")
    
    if 'TournamentNumber' in df.columns:
        # Статистика по турниру
        tournament_stats = df.groupby('TournamentNumber').agg({
            'Pot': 'mean',
            'Stack': 'mean',
            'Level': 'max'
        })
        
        tournament_stats.columns = ['tournament_avg_pot', 'tournament_avg_stack', 'tournament_max_level']
        df = df.merge(tournament_stats, left_on='TournamentNumber', right_index=True, how='left')
    
    # Позиция в турнире (ранняя/средняя/поздняя стадия)
    if 'Level' in df.columns:
        df['tournament_stage'] = pd.cut(df['Level'], bins=3, labels=['early', 'middle', 'late'])
    
    return df

def prepare_features(df, include_hole_cards):
    """Подготовка списка признаков"""
    feature_columns = [
        'Level', 'Pot', 'Stack', 'SPR', 'Street_id', 'Round',
        'ActionOrder', 'Seat', 'Dealer', 'Bet', 'Allin',
        'PlayerWins', 'WinAmount'
    ]
    
    # Добавляем контекстные признаки игрока
    player_context_cols = [
        'player_avg_bet', 'player_bet_std', 'player_hands_count',
        'player_allin_rate', 'player_win_rate', 'player_avg_win'
    ]
    
    for col in player_context_cols:
        if col in df.columns:
            feature_columns.append(col)
    
    # Карты стола
    for i in range(1, 6):
        if f'Card{i}_rank' in df.columns:
            feature_columns.extend([f'Card{i}_rank', f'Card{i}_suit'])
    
    # Карты игрока (если нужно)
    if include_hole_cards:
        if 'hole1_rank' in df.columns:
            feature_columns.extend(['hole1_rank', 'hole1_suit', 'hole2_rank', 'hole2_suit'])
    
    return feature_columns

class HandSequenceDataset(Dataset):
    """Dataset для последовательностей рук"""
    def __init__(self, sequences, feature_columns, scaler, max_seq_length=30):
        self.sequences = sequences
        self.feature_columns = feature_columns
        self.scaler = scaler
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_df = self.sequences[idx]
        
        # Извлекаем признаки
        features = seq_df[self.feature_columns].values
        features_scaled = self.scaler.transform(features)
        
        # Padding если нужно
        seq_len = len(features_scaled)
        if seq_len < self.max_seq_length:
            padding = np.zeros((self.max_seq_length - seq_len, features_scaled.shape[1]))
            features_scaled = np.vstack([features_scaled, padding])
        
        # Целевые переменные (берем с последней записи)
        last_row = seq_df.iloc[-1]
        targets = {
            'strength': last_row['hand_strength'],
            'categories': np.zeros(73)  # one-hot для категории
        }
        
        if 'hand_type_hm3' in last_row:
            # Найти индекс категории
            category_idx = 0  # заглушка
            targets['categories'][category_idx] = 1
        
        return {
            'features': torch.tensor(features_scaled, dtype=torch.float32),
            'targets': targets,
            'mask': torch.tensor([False] * seq_len + [True] * (self.max_seq_length - seq_len))
        }

def hand_sequence_collate_fn(batch):
    """Функция для батчинга последовательностей"""
    features = torch.stack([item['features'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    targets = {
        'strength': torch.tensor([item['targets']['strength'] for item in batch], dtype=torch.long),
        'categories': torch.tensor([item['targets']['categories'] for item in batch], dtype=torch.float32)
    }
    
    return {
        'features': features,
        'targets': targets,
        'mask': masks
    }

def evaluate_improved_model(model, data_dict):
    """Оценка улучшенной модели"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_dict['test_loader']:
            inputs = batch['features'].to(device)
            targets = batch['targets']['strength'].to(device)
            mask = batch.get('mask')
            
            outputs = model(inputs, mask=mask)
            _, predicted = torch.max(outputs['hand_strength'], 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    return {'strength_accuracy': accuracy}



def create_hand_sequences_dataset(train_df, val_df, test_df, include_hole_cards):
    """
    Создает последовательности внутри каждой руки
    """
    print(f"\n🔄 === СОЗДАНИЕ ПОСЛЕДОВАТЕЛЬНОСТЕЙ ВНУТРИ РУК ===")
    
    def create_sequences_for_hand(hand_data):
        """Создает все возможные префиксы действий в руке"""
        sequences = []
        
        # Сортируем по порядку действий
        hand_data = hand_data.sort_values(['Street_id', 'ActionOrder'])
        
        # Создаем последовательности разной длины
        for end_idx in range(1, len(hand_data) + 1):
            seq = hand_data.iloc[:end_idx].copy()
            
            # Добавляем позиционную информацию
            seq['sequence_length'] = end_idx
            seq['actions_remaining'] = len(hand_data) - end_idx
            seq['sequence_progress'] = end_idx / len(hand_data)
            
            sequences.append(seq)
        
        return sequences
    
    # Обрабатываем каждую выборку
    all_sequences = {'train': [], 'val': [], 'test': []}
    
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        hand_col = 'HandID' if 'HandID' in split_df.columns else 'Hand'
        
        for hand_id in split_df[hand_col].unique():
            hand_data = split_df[split_df[hand_col] == hand_id]
            
            # Создаем последовательности для этой руки
            hand_sequences = create_sequences_for_hand(hand_data)
            all_sequences[split_name].extend(hand_sequences)
        
        print(f"   {split_name}: {len(all_sequences[split_name])} последовательностей")
    
    # Подготовка признаков и создание датасетов
    feature_columns = prepare_features(train_df, include_hole_cards)
    
    # Нормализация
    scaler = StandardScaler()
    scaler.fit(pd.concat([seq[feature_columns] for seq in all_sequences['train']]))
    
    # Создание DataLoader'ов
    datasets = {}
    loaders = {}
    
    for split in ['train', 'val', 'test']:
        datasets[split] = HandSequenceDataset(
            all_sequences[split], 
            feature_columns, 
            scaler,
            max_seq_length=30
        )
        
        batch_size = 32 if split == 'train' else 64
        shuffle = split == 'train'
        
        loaders[split] = DataLoader(
            datasets[split], 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=hand_sequence_collate_fn
        )
    
    return {
        'train_loader': loaders['train'],
        'val_loader': loaders['val'],
        'test_loader': loaders['test'],
        'scaler': scaler,
        'feature_columns': feature_columns,
        'input_dim': len(feature_columns),
        'num_categories': 73 if 'hand_type_hm3' in train_df.columns else 9
    }


# ===================== УЛУЧШЕННАЯ МОДЕЛЬ =====================

class ImprovedHandRWKV(nn.Module):
    """
    Улучшенная RWKV модель с правильной рецептивностью
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Эмбеддинги для категориальных признаков
        self.action_embedding = nn.Embedding(10, 32)
        self.position_embedding = nn.Embedding(10, 16)
        self.street_embedding = nn.Embedding(5, 16)
        
        # Проекция входа
        self.input_projection = nn.Linear(input_dim + 64, hidden_dim)
        
        # Улучшенные RWKV блоки
        self.rwkv_blocks = nn.ModuleList([
            ImprovedRWKVBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Multi-head attention для ключевых моментов
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Нормализация
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Выходные головы
        self.strength_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 5)  # 5 классов силы
        )
        
        self.category_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 73)  # 73 HM3 категории
        )
        
    def forward(self, x, actions=None, positions=None, streets=None, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Комбинируем входные данные
        if actions is not None:
            action_emb = self.action_embedding(actions)
            x = torch.cat([x, action_emb], dim=-1)
        
        if positions is not None:
            position_emb = self.position_embedding(positions)
            x = torch.cat([x, position_emb], dim=-1)
            
        if streets is not None:
            street_emb = self.street_embedding(streets)
            x = torch.cat([x, street_emb], dim=-1)
        
        # Проекция
        x = self.input_projection(x)
        
        # Проход через RWKV блоки
        for block in self.rwkv_blocks:
            x = block(x, mask=mask)
            x = self.layer_norm(x)
            x = self.dropout(x)
        
        # Attention для выделения важных моментов
        x, _ = self.attention(x, x, x, key_padding_mask=mask)
        
        # Берем последний валидный токен для каждой последовательности
        if mask is not None:
            # Находим последний не-masked элемент
            lengths = (~mask).sum(dim=1)
            batch_indices = torch.arange(batch_size).to(x.device)
            last_tokens = x[batch_indices, lengths - 1]
        else:
            last_tokens = x[:, -1, :]
        
        # Предсказания
        strength_logits = self.strength_head(last_tokens)
        category_logits = self.category_head(last_tokens)
        
        return {
            'hand_strength': strength_logits,
            'category_probs': category_logits
        }


class ImprovedRWKVBlock(nn.Module):
    """
    Улучшенный RWKV блок с правильной реализацией
    """
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Time mixing параметры
        self.time_mix_k = nn.Parameter(torch.zeros(hidden_dim))
        self.time_mix_v = nn.Parameter(torch.zeros(hidden_dim))
        self.time_mix_r = nn.Parameter(torch.zeros(hidden_dim))
        
        # Time decay и bonus
        self.time_decay = nn.Parameter(torch.zeros(hidden_dim))
        self.time_first = nn.Parameter(torch.zeros(hidden_dim))
        
        # Линейные слои
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Инициализация
        self._init_weights()
        
    def _init_weights(self):
        # Специальная инициализация для покера
        nn.init.normal_(self.time_decay, -5, 1)  # Начальный decay
        nn.init.normal_(self.time_first, 0, 0.1)  # Небольшой bonus
        nn.init.uniform_(self.time_mix_k, 0.4, 0.6)
        nn.init.uniform_(self.time_mix_v, 0.4, 0.6) 
        nn.init.uniform_(self.time_mix_r, 0.4, 0.6)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        outputs = []
        state = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        
        for t in range(seq_len):
            xt = x[:, t]
            
            if t == 0:
                prev_x = torch.zeros_like(xt)
            else:
                prev_x = x[:, t-1]
            
            # Time mixing
            k = self.key(xt * self.time_mix_k + prev_x * (1 - self.time_mix_k))
            v = self.value(xt * self.time_mix_v + prev_x * (1 - self.time_mix_v))
            r = torch.sigmoid(self.receptance(xt * self.time_mix_r + prev_x * (1 - self.time_mix_r)))
            
            # WKV computation
            if t == 0:
                wkv = (self.time_first + k) * v
            else:
                # Обновляем состояние
                state = state * torch.exp(-torch.exp(self.time_decay)) + k * v
                wkv = state
            
            # Применяем рецептивность
            out = r * self.output(wkv)
            outputs.append(out)
            
        return torch.stack(outputs, dim=1)


# ===================== ФУНКЦИЯ ОБУЧЕНИЯ =====================

def train_improved_model(data_dict, model_params=None, training_params=None):
    """
    Обучение улучшенной модели
    """
    if model_params is None:
        model_params = {
            'hidden_dim': 256,
            'num_layers': 4,
            'num_heads': 8
        }
    
    if training_params is None:
        training_params = {
            'epochs': 30,
            'lr': 0.001,
            'patience': 5
        }
    
    print(f"\n🚀 === ОБУЧЕНИЕ УЛУЧШЕННОЙ МОДЕЛИ ===")
    print(f"📊 Параметры модели: {model_params}")
    print(f"🎯 Параметры обучения: {training_params}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создаем модель
    model = ImprovedHandRWKV(
        input_dim=data_dict['input_dim'],
        **model_params
    ).to(device)
    
    # Оптимизатор и планировщик
    optimizer = optim.AdamW(model.parameters(), lr=training_params['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Функции потерь
    strength_criterion = nn.CrossEntropyLoss()
    category_criterion = nn.BCEWithLogitsLoss()
    
    # История обучения
    history = defaultdict(list)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(training_params['epochs']):
        # Training
        model.train()
        train_losses = []
        
        for batch in data_dict['train_loader']:
            inputs = batch['features'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            
            optimizer.zero_grad()
            
            outputs = model(
                inputs,
                actions=batch.get('actions'),
                positions=batch.get('positions'),
                streets=batch.get('streets'),
                mask=batch.get('mask')
            )
            
            # Потери
            strength_loss = strength_criterion(outputs['hand_strength'], targets['strength'])
            category_loss = category_criterion(outputs['category_probs'], targets['categories'])
            
            total_loss = strength_loss + 0.5 * category_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(total_loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in data_dict['val_loader']:
                inputs = batch['features'].to(device)
                targets = {k: v.to(device) for k, v in batch['targets'].items()}
                
                outputs = model(
                    inputs,
                    actions=batch.get('actions'),
                    positions=batch.get('positions'),
                    streets=batch.get('streets'),
                    mask=batch.get('mask')
                )
                
                strength_loss = strength_criterion(outputs['hand_strength'], targets['strength'])
                category_loss = category_criterion(outputs['category_probs'], targets['categories'])
                total_loss = strength_loss + 0.5 * category_loss
                
                val_losses.append(total_loss.item())
                
                # Точность
                _, predicted = torch.max(outputs['hand_strength'], 1)
                val_total += targets['strength'].size(0)
                val_correct += (predicted == targets['strength']).sum().item()
        
        # Метрики эпохи
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_accuracy = val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{training_params['epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.3f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Сохраняем лучшую модель
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= training_params['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        scheduler.step(avg_val_loss)
    
    # Загружаем лучшую модель
    model.load_state_dict(best_model_state)
    
    return model, history


# ===================== ГЛАВНАЯ ФУНКЦИЯ ДЛЯ ПАРАЛЛЕЛЬНОГО ОБУЧЕНИЯ =====================

def run_parallel_training(data_path):
    """
    Запускает параллельное обучение: старый подход vs новый подход
    """
    print(f"\n🏁 === ПАРАЛЛЕЛЬНОЕ СРАВНЕНИЕ ПОДХОДОВ ===")
    print(f"📊 Данные: {data_path}")
    
    results = {}
    
    # 1. Старый подход (разделение по игрокам)
    print(f"\n" + "="*80)
    print(f"📌 1. СТАРЫЙ ПОДХОД (разделение по игрокам)")
    print(f"="*80)
    
    try:
        old_data = prepare_sequence_hand_range_data_with_hm3(
            data_path,
            include_hole_cards=False,
            max_sequence_length=20
        )
        
        old_model, old_history = train_sequence_hand_range_model(
            old_data,
            hidden_dim=128,
            num_layers=3,
            epochs=25
        )
        
        old_performance = evaluate_sequence_model_performance(old_model, old_data)
        results['old_approach'] = {
            'accuracy': old_performance['strength_accuracy'],
            'history': old_history
        }
        
        print(f"✅ Старый подход: точность = {old_performance['strength_accuracy']:.3f}")
        
    except Exception as e:
        print(f"❌ Ошибка в старом подходе: {e}")
        
    # 2. Новый подход (разделение по рукам)
    print(f"\n" + "="*80)
    print(f"🎯 2. НОВЫЙ ПОДХОД (разделение по рукам)")
    print(f"="*80)
    
    try:
        new_data = prepare_hand_based_sequences(
            data_path,
            include_hole_cards=False,
            use_hm3=True
        )
        
        new_model, new_history = train_improved_model(new_data)
        
        # Оценка
        new_performance = evaluate_improved_model(new_model, new_data)
        results['new_approach'] = {
            'accuracy': new_performance['strength_accuracy'],
            'history': new_history
        }
        
        print(f"✅ Новый подход: точность = {new_performance['strength_accuracy']:.3f}")
        
    except Exception as e:
        print(f"❌ Ошибка в новом подходе: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Сравнение результатов
    if len(results) == 2:
        print(f"\n" + "="*80)
        print(f"🏆 === СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")
        print(f"="*80)
        
        old_acc = results['old_approach']['accuracy']
        new_acc = results['new_approach']['accuracy']
        improvement = new_acc - old_acc
        
        print(f"📊 Старый подход (по игрокам): {old_acc:.3f}")
        print(f"🎯 Новый подход (по рукам): {new_acc:.3f}")
        print(f"📈 Улучшение: {improvement:+.3f} ({improvement/old_acc*100:+.1f}%)")
        
        if improvement > 0:
            print(f"\n✅ Новый подход показывает лучшие результаты!")
            print(f"💡 Это подтверждает, что разделение по рукам более корректно")
        else:
            print(f"\n🤔 Старый подход работает лучше или одинаково")
            print(f"💡 Возможные причины:")
            print(f"   - Недостаточно данных")
            print(f"   - Нужна настройка гиперпараметров")
            print(f"   - Специфика данных")
    
    return results


# -----------------------------------------------------------------





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
            comparison_report_safe = safe_json_serialize(comparison_report)
            json.dump(comparison_report_safe, f, indent=2, ensure_ascii=False)

        print(f"   📋 Отчет сохранен: {report_path}")

    print(f"\n🎉 Обучение завершено!")
    print(f"📁 Результаты сохранены в папках:")
    print(f"   🤖 models/     - обученные модели")
    print(f"   📊 plots/      - графики обучения")
    print(f"   📋 results/    - отчеты и конфигурации")
    print(f"   🃏 results/poker_categories.json - категории рук")


# if __name__ == "__main__":
#     main()


def choose_data_file_with_percentage():
    """Выбор файла с данными с возможностью выбора процента файлов"""
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

    # Сортируем по размеру
    sorted_files = sorted(file_stats, key=lambda x: x[1], reverse=True)

    print(
        f"📁 Найдено {len(data_files)} файлов с данными (общий размер: {total_size_mb:.1f} MB):"
    )

    # Показываем топ-10 файлов
    print(f"📊 Топ-10 крупнейших файлов:")
    for i, (file, size_mb) in enumerate(sorted_files[:10]):
        percentage = (size_mb / total_size_mb) * 100
        print(
            f"  {i+1}. {os.path.basename(file)} ({size_mb:.1f} MB, {percentage:.1f}%)"
        )

    if len(data_files) > 10:
        print(f"\n📋 Показаны только топ-10 файлов из {len(data_files)}")

    print(f"\n📊 ОПЦИИ ВЫБОРА:")
    print(f"  0. ⭐ ОБЪЕДИНИТЬ ВСЕ {len(data_files)} ФАЙЛОВ (100%)")
    print(f"  25. 📊 Объединить 25% файлов (~{len(data_files)//4} файлов)")
    print(f"  50. 📊 Объединить 50% файлов (~{len(data_files)//2} файлов)")
    print(f"  75. 📊 Объединить 75% файлов (~{3*len(data_files)//4} файлов)")
    print(f"  1-10. 📄 Выбрать конкретный файл из топ-10")
    print(f"  p. 🎯 Ввести свой процент (например: p30 для 30%)")

    while True:
        try:
            choice = input("\nВыберите опцию: ").strip().lower()

            if choice in ["q", "quit", "exit"]:
                return None

            # Проверяем процентный ввод
            if choice.startswith("p"):
                try:
                    custom_percent = int(choice[1:])
                    if 1 <= custom_percent <= 100:
                        num_files = max(1, int(len(data_files) * custom_percent / 100))
                        print(
                            f"✅ Выбрано: объединить {custom_percent}% файлов ({num_files} из {len(data_files)})"
                        )
                        return ("COMBINE_PERCENT", custom_percent)
                    else:
                        print("❌ Процент должен быть от 1 до 100")
                        continue
                except ValueError:
                    print("❌ Неверный формат. Используйте p25 для 25%")
                    continue

            # Обработка числовых вводов
            choice_num = int(choice)

            if choice_num == 0:
                print(f"✅ Выбрано: объединить все {len(data_files)} файлов")
                return ("COMBINE_PERCENT", 100)
            elif choice_num == 25:
                print(
                    f"✅ Выбрано: объединить 25% файлов ({len(data_files)//4} из {len(data_files)})"
                )
                return ("COMBINE_PERCENT", 25)
            elif choice_num == 50:
                print(
                    f"✅ Выбрано: объединить 50% файлов ({len(data_files)//2} из {len(data_files)})"
                )
                return ("COMBINE_PERCENT", 50)
            elif choice_num == 75:
                print(
                    f"✅ Выбрано: объединить 75% файлов ({3*len(data_files)//4} из {len(data_files)})"
                )
                return ("COMBINE_PERCENT", 75)
            elif 1 <= choice_num <= min(10, len(data_files)):
                selected_file = sorted_files[choice_num - 1][0]
                print(f"✅ Выбран файл: {selected_file}")
                return selected_file
            else:
                print(f"❌ Неверный выбор")

        except ValueError:
            print("❌ Введите число или процент (p30)")


def combine_percentage_of_files(percentage):
    """Объединение указанного процента файлов"""
    data_files = find_data_files()

    if not data_files:
        print("❌ Не найдено файлов для объединения")
        return None

    # Сортируем файлы по размеру (берем сначала большие)
    file_stats = []
    for file in data_files:
        size = os.path.getsize(file)
        file_stats.append((file, size))

    # Сортируем по размеру в убывающем порядке
    sorted_files = sorted(file_stats, key=lambda x: x[1], reverse=True)

    # Выбираем нужное количество файлов
    num_files_to_take = max(1, int(len(data_files) * percentage / 100))
    selected_files = [f[0] for f in sorted_files[:num_files_to_take]]

    print(f"\n🔗 === ОБЪЕДИНЕНИЕ {percentage}% ФАЙЛОВ ===")
    print(f"📊 Выбрано {num_files_to_take} из {len(data_files)} файлов")
    print(f"📈 Это самые большие файлы по размеру")

    all_dataframes = []
    total_records = 0
    total_showdowns = 0
    total_size_mb = 0

    for i, data_path in enumerate(selected_files, 1):
        print(
            f"📄 Загружаем файл {i}/{num_files_to_take}: {os.path.basename(data_path)}"
        )

        try:
            df = pd.read_csv(data_path)

            # Добавляем информацию об источнике
            df["source_file"] = os.path.basename(data_path)
            df["file_index"] = i

            # Подсчет статистики
            showdowns_in_file = (
                (df["Showdown_1"].notna()) & (df["Showdown_2"].notna())
            ).sum()
            file_size_mb = os.path.getsize(data_path) / 1024 / 1024

            print(
                f"   📊 Размер: {len(df):,} строк, {file_size_mb:.1f} MB, шоудаунов: {showdowns_in_file:,}"
            )

            all_dataframes.append(df)
            total_records += len(df)
            total_showdowns += showdowns_in_file
            total_size_mb += file_size_mb

        except Exception as e:
            print(f"   ❌ Ошибка загрузки: {e}")
            continue

    if not all_dataframes:
        print("❌ Не удалось загрузить ни одного файла")
        return None

    # Объединяем данные
    print(f"\n🔄 Объединяем данные...")
    combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)

    print(f"✅ Объединение завершено:")
    print(f"   📈 Общий размер: {total_records:,} строк")
    print(f"   💾 Общий объем: {total_size_mb:.1f} MB")
    print(f"   🃏 Общее количество шоудаунов: {total_showdowns:,}")
    print(
        f"   📊 Процент записей с шоудауном: {total_showdowns/total_records*100:.1f}%"
    )
    print(f"   📁 Использовано файлов: {num_files_to_take} ({percentage}%)")

    # Сохраняем объединенный файл
    combined_dir = os.path.join("data", "combined")
    os.makedirs(combined_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_filename = os.path.join(
        combined_dir, f"combined_{percentage}pct_poker_data_{timestamp}.csv"
    )

    combined_df.to_csv(combined_filename, index=False)
    print(f"💾 Объединенные данные сохранены: {combined_filename}")

    # Создаем сводку
    summary = {
        "timestamp": datetime.now().isoformat(),
        "percentage_selected": percentage,
        "total_files_available": len(data_files),
        "files_used": num_files_to_take,
        "total_records": int(total_records),
        "total_showdowns": int(total_showdowns),
        "total_size_mb": float(total_size_mb),
        "combined_file": combined_filename,
        "files_included": [
            os.path.basename(f) for f in selected_files[:10]
        ],  # Первые 10 для примера
    }

    summary_path = f"results/data_combination_{percentage}pct_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"📋 Сводка сохранена: {summary_path}")

    return combined_filename, summary


def main_with_sequences():
    """Обновленная главная функция с поддержкой HM3"""
    print("🎰 === ОБУЧЕНИЕ RWKV МОДЕЛЕЙ С ПОСЛЕДОВАТЕЛЬНОСТЯМИ (HM3) ===\n")

    # Создаем необходимые папки
    setup_directories()
    save_categories_json()

    # Сохраняем маппинг HM3 категорий
    save_hm3_categories_mapping()

    # Выбираем файл данных с новой функцией
    data_choice = choose_data_file_with_percentage()
    if not data_choice:
        print("👋 Выход из программы.")
        return

    # Обрабатываем выбор
    if isinstance(data_choice, tuple) and data_choice[0] == "COMBINE_PERCENT":
        percentage = data_choice[1]

        # Объединяем указанный процент файлов
        result = combine_percentage_of_files(percentage)
        if result is None:
            print("❌ Не удалось объединить файлы")
            return

        data_path, combination_summary = result
        print(f"\n✅ Будем обучать модели на {percentage}% данных")

    else:
        # Обычная обработка одного файла
        data_path = data_choice
        combination_summary = None

    # Анализ файла данных
    print(f"\n📊 === АНАЛИЗ ДАННЫХ ===")
    df_check = pd.read_csv(data_path)
    showdown_count = (
        (df_check["Showdown_1"].notna()) & (df_check["Showdown_2"].notna())
    ).sum()
    total_rows = len(df_check)

    print(f"📈 Общий размер файла: {total_rows:,} строк")
    print(f"🃏 Записей с открытыми картами: {showdown_count:,}")
    print(f"📊 Процент шоудаунов: {showdown_count/total_rows*100:.1f}%")

    if showdown_count < 50:
        print("⚠️  Внимание: мало данных для качественного обучения!")
        print("💡 Рекомендуется использовать опцию объединения всех файлов")
    else:
        print("✅ Достаточно данных для обучения последовательных моделей")

    results = {}
    all_plots = []

    # Параметры обучения
    max_sequence_length = 15  # Оптимальная длина для покера
    training_params = {"hidden_dim": 128, "num_layers": 3, "epochs": 25, "lr": 0.001}

    print(f"\n🎯 Параметры обучения:")
    print(f"   📏 Максимальная длина последовательности: {max_sequence_length}")
    print(f"   🧠 Скрытых нейронов: {training_params['hidden_dim']}")
    print(f"   🏗️  Слоев RWKV: {training_params['num_layers']}")
    print(f"   📚 Эпох обучения: {training_params['epochs']}")
    print(f"   📈 Learning rate: {training_params['lr']}")

    # ======================================================================
    # 1. МОДЕЛЬ С КАРТАМИ ИГРОКА
    # ======================================================================
    print(f"\n" + "=" * 80)
    print(f"🎯 1. ОБУЧЕНИЕ МОДЕЛИ С КАРТАМИ ИГРОКА (последовательности)")
    print(f"=" * 80)

    try:
        print(f"📥 Подготовка данных с картами игрока...")
        data_with_cards = prepare_sequence_hand_range_data_with_hm3(
            data_path,
            include_hole_cards=True,
            max_sequence_length=max_sequence_length,
            use_hm3_classification=True,  # Включаем HM3!
        )

        if data_with_cards is not None:
            print(f"🚀 Запуск обучения...")
            model_with_cards, history_with_cards = train_sequence_hand_range_model(
                data_with_cards, **training_params
            )

            # Оценка модели
            print(f"🔍 Оценка производительности...")
            results["with_cards"] = evaluate_sequence_model_performance(
                model_with_cards, data_with_cards, include_hole_cards=True
            )

            # Визуализация
            print(f"📊 Создание графиков...")
            fig1, plot_path1 = visualize_sequence_results(
                model_with_cards, data_with_cards, history_with_cards, True
            )
            all_plots.append(plot_path1)
            plt.show()

            # Примеры предсказаний
            predict_sequence_hand_ranges(
                model_with_cards, data_with_cards, sample_hands=3
            )

            # Сохранение модели
            model_path, best_path = save_sequence_model(
                model_with_cards, data_with_cards, results["with_cards"], True
            )

            print(f"✅ Модель с картами обучена успешно!")

        else:
            print("❌ Не удалось подготовить данные для модели с картами")

    except Exception as e:
        print(f"❌ Ошибка при обучении модели с картами: {e}")
        import traceback

        traceback.print_exc()

    # ======================================================================
    # 2. МОДЕЛЬ БЕЗ КАРТ ИГРОКА
    # ======================================================================
    print(f"\n" + "=" * 80)
    print(f"🎲 2. ОБУЧЕНИЕ МОДЕЛИ БЕЗ КАРТ ИГРОКА (последовательности)")
    print(f"   🎯 Цель: угадать карты по поведению в последовательности действий")
    print(f"=" * 80)

    try:
        print(f"📥 Подготовка данных без карт игрока...")
        data_without_cards = prepare_sequence_hand_range_data_with_hm3(
            data_path, include_hole_cards=False, max_sequence_length=max_sequence_length
        )

        if data_without_cards is not None:
            print(f"🚀 Запуск обучения...")
            model_without_cards, history_without_cards = (
                train_sequence_hand_range_model(data_without_cards, **training_params)
            )

            # Оценка модели
            print(f"🔍 Оценка производительности...")
            results["without_cards"] = evaluate_sequence_model_performance(
                model_without_cards, data_without_cards, include_hole_cards=False
            )

            # Визуализация
            print(f"📊 Создание графиков...")
            fig2, plot_path2 = visualize_sequence_results(
                model_without_cards, data_without_cards, history_without_cards, False
            )
            all_plots.append(plot_path2)
            plt.show()

            # Примеры предсказаний
            predict_sequence_hand_ranges(
                model_without_cards, data_without_cards, sample_hands=3
            )

            # Сохранение модели
            model_path, best_path = save_sequence_model(
                model_without_cards, data_without_cards, results["without_cards"], False
            )

            print(f"✅ Модель без карт обучена успешно!")

        else:
            print("❌ Не удалось подготовать данные для модели без карт")

    except Exception as e:
        print(f"❌ Ошибка при обучении модели без карт: {e}")
        import traceback

        traceback.print_exc()

    # ======================================================================
    # 3. СРАВНЕНИЕ МОДЕЛЕЙ
    # ======================================================================
    print(f"\n" + "=" * 80)
    print(f"🏆 === СРАВНЕНИЕ ПОСЛЕДОВАТЕЛЬНЫХ МОДЕЛЕЙ ===")
    print(f"=" * 80)

    if "with_cards" in results and "without_cards" in results:
        with_cards_acc = results["with_cards"]["strength_accuracy"]
        without_cards_acc = results["without_cards"]["strength_accuracy"]

        with_cards_cat_acc = results["with_cards"]["category_accuracy"]
        without_cards_cat_acc = results["without_cards"]["category_accuracy"]

        with_cards_mse = results["with_cards"]["specific_mse"]
        without_cards_mse = results["without_cards"]["specific_mse"]

        print(f"📊 Сравнение точности предсказания силы руки:")
        print(f"   🎯 С картами игрока: {with_cards_acc:.3f}")
        print(f"   🎲 Без карт игрока: {without_cards_acc:.3f}")
        print(f"   📈 Разница: {with_cards_acc - without_cards_acc:.3f}")

        print(f"\n📊 Сравнение точности категорий:")
        print(f"   🎯 С картами: {with_cards_cat_acc:.3f}")
        print(f"   🎲 Без карт: {without_cards_cat_acc:.3f}")
        print(f"   📈 Разница: {with_cards_cat_acc - without_cards_cat_acc:.3f}")

        print(f"\n📊 Сравнение MSE рангов:")
        print(f"   🎯 С картами: {with_cards_mse:.4f}")
        print(f"   🎲 Без карт: {without_cards_mse:.4f}")
        print(f"   📈 Разница: {with_cards_mse - without_cards_mse:.4f}")

        # Интерпретация результатов
        print(f"\n🧠 === ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ ===")

        strength_diff = with_cards_acc - without_cards_acc
        if strength_diff > 0.1:
            print(f"✅ Модель с картами значительно лучше (+{strength_diff:.1%})")
            print(f"   💡 Это ожидаемо: зная карты, модель должна предсказывать лучше")
        elif strength_diff > 0.05:
            print(f"✅ Модель с картами лучше (+{strength_diff:.1%})")
            print(f"   💡 Умеренное преимущество")
        elif abs(strength_diff) <= 0.05:
            print(
                f"🤔 Модели показывают сопоставимую точность (±{abs(strength_diff):.1%})"
            )
            print(
                f"   💡 Это интересно: модель научилась угадывать карты по поведению!"
            )
            print(f"   🎯 Возможные причины:")
            print(f"      - Игроки слишком предсказуемы в своих действиях")
            print(f"      - Сильные корреляции между ситуацией и силой руки")
            print(f"      - Недостаточно разнообразия в данных")
        else:
            print(f"⚠️  Модель без карт работает лучше (+{abs(strength_diff):.1%})")
            print(f"   🤔 Это странно и требует дополнительного анализа")

        # Анализ по количеству последовательностей
        with_cards_seqs = results["with_cards"]["total_sequences"]
        without_cards_seqs = results["without_cards"]["total_sequences"]

        print(f"\n📈 Статистика обучения:")
        print(f"   🎯 С картами: {with_cards_seqs} последовательностей")
        print(f"   🎲 Без карт: {without_cards_seqs} последовательностей")

        # Сохранение отчета
        comparison_report = {
            "timestamp": datetime.now().isoformat(),
            "data_file": data_path,
            "total_records": total_rows,
            "showdown_records": showdown_count,
            "sequence_params": {
                "max_sequence_length": max_sequence_length,
            },
            "training_params": training_params,
            "models": {
                "with_cards": {
                    "strength_accuracy": float(with_cards_acc),
                    "category_accuracy": float(with_cards_cat_acc),
                    "specific_mse": float(with_cards_mse),
                    "total_sequences": int(with_cards_seqs),
                },
                "without_cards": {
                    "strength_accuracy": float(without_cards_acc),
                    "category_accuracy": float(without_cards_cat_acc),
                    "specific_mse": float(without_cards_mse),
                    "total_sequences": int(without_cards_seqs),
                },
            },
            "differences": {
                "strength_accuracy_diff": float(strength_diff),
                "category_accuracy_diff": float(
                    with_cards_cat_acc - without_cards_cat_acc
                ),
                "specific_mse_diff": float(with_cards_mse - without_cards_mse),
            },
            "plots": all_plots,
        }

        report_path = f"results/sequence_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            # Используем safe_json_serialize для конвертации numpy типов
            comparison_report_safe = safe_json_serialize(comparison_report)
            json.dump(comparison_report_safe, f, indent=2, ensure_ascii=False)

        print(f"📋 Детальный отчет сохранен: {report_path}")

    elif "with_cards" in results:
        print(f"✅ Успешно обучена только модель С картами")
        print(
            f"   🎯 Точность силы руки: {results['with_cards']['strength_accuracy']:.3f}"
        )
    elif "without_cards" in results:
        print(f"✅ Успешно обучена только модель БЕЗ карт")
        print(
            f"   🎲 Точность силы руки: {results['without_cards']['strength_accuracy']:.3f}"
        )
    else:
        print(f"❌ Не удалось обучить ни одну модель")
        return

    # ======================================================================
    # 4. ЗАКЛЮЧЕНИЕ
    # ======================================================================
    print(f"\n🎉 === ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
    print(f"📁 Результаты сохранены в папках:")
    print(f"   🤖 models/     - обученные RWKV модели")
    print(f"   📊 plots/      - графики обучения и анализа")
    print(f"   📋 results/    - отчеты и конфигурации")
    print(f"   🃏 results/poker_categories.json - справочник категорий рук")

    print(f"\n💡 Рекомендации:")
    print(f"   📚 Изучите графики обучения на предмет переобучения")
    print(f"   🔍 Проанализируйте примеры предсказаний для понимания логики модели")
    print(f"   📊 Сравните результаты с базовыми методами (логистическая регрессия)")
    print(f"   🎯 Попробуйте разные длины последовательностей для оптимизации")

    return results


def save_sequence_model(model, data_dict, performance, include_hole_cards):
    """
    Сохранение обученной последовательной модели с полной информацией
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = (
        "sequence_with_cards" if include_hole_cards else "sequence_without_cards"
    )

    model_name = f"hand_range_{model_type}"
    model_file = f"{model_name}_{timestamp}.pth"

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", model_file)

    # Сохраняем модель со всей необходимой информацией
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_class": "SequenceHandRangeRWKV",
            "scaler": data_dict["scaler"],
            "encoders": data_dict.get("encoders", {}),
            "feature_columns": data_dict["feature_columns"],
            "input_dim": data_dict["input_dim"],
            "max_sequence_length": data_dict["max_sequence_length"],
            "include_hole_cards": include_hole_cards,
            "performance": performance,
            "timestamp": timestamp,
            "categories": PokerHandAnalyzer.get_all_categories(),
            "category_mapping": PokerHandAnalyzer.get_category_mapping(),
            "rank_names": PokerHandAnalyzer.get_rank_names(),
            "model_type": "sequence_based",
            "framework": "pytorch_rwkv",
        },
        model_path,
    )

    # Создаем ссылку на лучшую модель
    best_model_path = os.path.join("models", f"{model_name}_best.pth")
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

    import shutil

    shutil.copy2(model_path, best_model_path)

    print(f"💾 Модель сохранена:")
    print(f"   📁 {model_path}")
    print(f"   🔗 {best_model_path} (лучшая)")

    return model_path, best_model_path


def process_all_files_with_sequences():
    """
    Массовая обработка всех файлов с последовательностями
    """
    print(f"\n🚀 === МАССОВАЯ ОБРАБОТКА С ПОСЛЕДОВАТЕЛЬНОСТЯМИ ===")

    # Объединяем все файлы
    result = combine_all_data_files()
    if result is None:
        print("❌ Не удалось объединить файлы")
        return

    combined_filename, combination_summary = result

    print(f"\n🎯 === ОБУЧЕНИЕ ПОСЛЕДОВАТЕЛЬНЫХ МОДЕЛЕЙ НА ОБЪЕДИНЕННЫХ ДАННЫХ ===")
    print(f"📁 Файл: {combined_filename}")
    print(f"📊 Шоудаунов: {combination_summary['total_showdowns']:,}")

    if combination_summary["total_showdowns"] < 100:
        print("⚠️  Мало данных для последовательных моделей!")
        return

    results = {}
    all_plots = []

    max_sequence_length = 20  # Больше для объединенных данных
    training_params = {
        "hidden_dim": 256,  # Больше нейронов
        "num_layers": 4,  # Больше слоев
        "epochs": 30,  # Больше эпох
        "lr": 0.0008,  # Чуть меньше lr
    }

    print(f"🎯 Параметры для большого датасета:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    print(f"   max_sequence_length: {max_sequence_length}")

    # Обучаем обе модели
    for model_type, include_cards in [("БЕЗ карт", False), ("С картами", True)]:
        print(f"\n{'='*80}")
        print(
            f"🎲 Обучение последовательной модели {model_type} на объединенных данных"
        )
        print(f"{'='*80}")

        try:
            data_dict = prepare_sequence_hand_range_data_with_hm3(
                combined_filename,
                include_hole_cards=include_cards,
                max_sequence_length=max_sequence_length,
            )

            if data_dict is not None:
                model, history = train_sequence_hand_range_model(
                    data_dict, **training_params
                )

                performance = evaluate_sequence_model_performance(
                    model, data_dict, include_hole_cards=include_cards
                )

                fig, plot_path = visualize_sequence_results(
                    model, data_dict, history, include_cards
                )
                all_plots.append(plot_path)
                plt.show()

                predict_sequence_hand_ranges(model, data_dict, sample_hands=5)

                model_path, best_path = save_sequence_model(
                    model, data_dict, performance, include_cards
                )

                model_key = "with_cards" if include_cards else "without_cards"

                # Конвертируем numpy типы в обычные Python типы для JSON
                results[model_key] = {
                    "strength_accuracy": float(performance["strength_accuracy"]),
                    "category_accuracy": float(performance["category_accuracy"]),
                    "specific_mse": float(performance["specific_mse"]),
                    "total_sequences": int(performance["total_sequences"]),
                }

                print(f"✅ Последовательная модель {model_type} обучена!")

            else:
                print(f"❌ Ошибка подготовки данных для модели {model_type}")

        except Exception as e:
            print(f"❌ Ошибка обучения модели {model_type}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Финальное сравнение
    if "with_cards" in results and "without_cards" in results:
        print(f"\n🏆 === ФИНАЛЬНОЕ СРАВНЕНИЕ ПОСЛЕДОВАТЕЛЬНЫХ МОДЕЛЕЙ ===")

        with_cards_acc = results["with_cards"]["strength_accuracy"]
        without_cards_acc = results["without_cards"]["strength_accuracy"]

        print(f"🎯 С картами: {with_cards_acc:.3f}")
        print(f"🎲 Без карт: {without_cards_acc:.3f}")
        print(f"📈 Разница: {with_cards_acc - without_cards_acc:.3f}")

        # Создаем финальный отчет с правильными типами данных
        final_report = {
            "timestamp": datetime.now().isoformat(),
            "model_type": "sequence_based_rwkv",
            "combined_data_summary": {
                "total_files": int(combination_summary["total_files"]),
                "total_records": int(combination_summary["total_records"]),
                "total_showdowns": int(combination_summary["total_showdowns"]),
                "combined_file": str(combination_summary["combined_file"]),
            },
            "sequence_params": {"max_length": int(max_sequence_length)},
            "training_params": {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in training_params.items()
            },
            "results": results,
            "plots": [str(p) for p in all_plots],
        }

        report_path = f"results/sequence_combined_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        print(f"📋 Финальный отчет: {report_path}")

    print(f"\n🎉 Массовое обучение последовательных моделей завершено!")


def main_parallel():
    """
    Главная функция для параллельного сравнения подходов
    """
    print("🎰 === ПАРАЛЛЕЛЬНОЕ СРАВНЕНИЕ ПОДХОДОВ К ОБУЧЕНИЮ ===")
    
    # Создаем папки
    setup_directories()
    
    # Выбираем данные
    data_choice = choose_data_file_with_percentage()
    if not data_choice:
        return
    
    # Обрабатываем выбор
    if isinstance(data_choice, tuple) and data_choice[0] == "COMBINE_PERCENT":
        data_path, _ = combine_percentage_of_files(data_choice[1])
    else:
        data_path = data_choice
    
    # Запускаем параллельное обучение
    results = run_parallel_training(data_path)
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"results/parallel_comparison_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📋 Отчет сохранен: {report_path}")
    print(f"🎉 Параллельное сравнение завершено!")


if __name__ == "__main__":
    main_parallel()
    

# Обновляем главную функцию
if __name__ == "__main__":
    # Выбор режима запуска
    print("Выберите режим:")
    print("1. Старый подход (разделение по игрокам)")
    print("2. Новый подход (разделение по рукам)")
    print("3. Параллельное сравнение обоих подходов")
    
    choice = input("Ваш выбор (1/2/3): ").strip()
    
    if choice == "1":
        main_with_sequences()  # Старый код
    elif choice == "2":
        # Только новый подход
        setup_directories()
        data_choice = choose_data_file_with_percentage()
        if data_choice:
            if isinstance(data_choice, tuple):
                data_path, _ = combine_percentage_of_files(data_choice[1])
            else:
                data_path = data_choice
            new_data = prepare_hand_based_sequences(data_path, include_hole_cards=False)
            model, history = train_improved_model(new_data)
    else:
        main_parallel()  # Параллельное сравнение
