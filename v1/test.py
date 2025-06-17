import pandas as pd
import numpy as np


def analyze_data_issues():
    """Анализ проблемных значений в данных"""

    # Загружаем данные
    df = pd.read_csv("parsed_PDOM1844027118_v2.csv")
    print(f"=== АНАЛИЗ ДАННЫХ ===")
    print(f"Общий размер: {len(df)} строк, {len(df.columns)} колонок")

    # Получаем числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Числовых колонок: {len(numeric_cols)}")

    print(f"\nЧисловые колонки: {list(numeric_cols)}")

    # 1. Анализ NaN значений
    print(f"\n=== NaN ЗНАЧЕНИЯ ===")
    nan_found = False
    for col in numeric_cols:
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            nan_percent = (nan_count / len(df)) * 100
            print(f"  {col}: {nan_count} NaN ({nan_percent:.2f}%)")
            nan_found = True

    if not nan_found:
        print("  ✅ NaN значения не найдены в числовых колонках")

    # 2. Анализ бесконечных значений
    print(f"\n=== БЕСКОНЕЧНЫЕ ЗНАЧЕНИЯ ===")
    inf_found = False
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_percent = (inf_count / len(df)) * 100
            print(f"  {col}: {inf_count} inf ({inf_percent:.2f}%)")

            # Показываем примеры inf значений
            inf_mask = np.isinf(df[col])
            if inf_mask.any():
                inf_samples = df[inf_mask][col].head(3)
                print(f"    Примеры: {list(inf_samples)}")
            inf_found = True

    if not inf_found:
        print("  ✅ Бесконечные значения не найдены")

    # 3. Специальный анализ SPR (Stack-to-Pot Ratio)
    if "SPR" in df.columns:
        print(f"\n=== АНАЛИЗ SPR (Stack-to-Pot Ratio) ===")
        spr_stats = df["SPR"].describe()
        print(f"  Статистика SPR:")
        print(f"    Среднее: {spr_stats['mean']:.3f}")
        print(f"    Медиана: {spr_stats['50%']:.3f}")
        print(f"    Максимум: {spr_stats['max']:.3f}")
        print(f"    Минимум: {spr_stats['min']:.3f}")

        # Проверяем экстремальные значения
        very_high_spr = (df["SPR"] > 1000).sum()
        very_low_spr = (df["SPR"] < 0).sum()
        inf_spr = np.isinf(df["SPR"]).sum()

        print(f"    SPR > 1000: {very_high_spr} записей")
        print(f"    SPR < 0: {very_low_spr} записей")
        print(f"    SPR = inf: {inf_spr} записей")

        # Проверяем потенциальную причину inf - нулевые pot
        if "Pot" in df.columns:
            zero_pot = (df["Pot"] == 0).sum()
            very_small_pot = (df["Pot"] > 0) & (df["Pot"] < 0.01)
            very_small_pot_count = very_small_pot.sum()

            print(
                f"    Pot = 0: {zero_pot} записей (может вызывать inf при SPR = Stack/Pot)"
            )
            print(
                f"    Pot < 0.01: {very_small_pot_count} записей (может вызывать очень большие SPR)"
            )

            if zero_pot > 0:
                print(f"    ⚠️  ПРОБЛЕМА: Pot = 0 вызывает inf при расчете SPR!")

    # 4. Анализ данных с шоудауном
    print(f"\n=== ДАННЫЕ С ШОУДАУНОМ ===")
    showdown_mask = (df["Showdown_1"].notna()) & (df["Showdown_2"].notna())
    df_showdown = df[showdown_mask]
    print(
        f"  Записей с шоудауном: {len(df_showdown)} из {len(df)} ({len(df_showdown)/len(df)*100:.1f}%)"
    )

    if len(df_showdown) > 0:
        print(f"  Проверка проблемных значений в данных с шоудауном:")
        problems_in_showdown = False

        for col in numeric_cols:
            if col in df_showdown.columns:
                nan_count = df_showdown[col].isnull().sum()
                inf_count = np.isinf(df_showdown[col]).sum()

                if nan_count > 0 or inf_count > 0:
                    print(f"    {col}: {nan_count} NaN, {inf_count} inf")
                    problems_in_showdown = True

        if not problems_in_showdown:
            print(f"    ✅ Проблемных значений в данных с шоудауном не найдено")

    # 5. Проверка конкретных колонок, которые могут вызывать проблемы
    print(f"\n=== ПРОБЛЕМНЫЕ КОЛОНКИ ===")

    problem_columns = []

    for col in numeric_cols:
        issues = []

        # NaN
        nan_count = df[col].isnull().sum()
        if nan_count > 0:
            issues.append(f"{nan_count} NaN")

        # Inf
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            issues.append(f"{inf_count} inf")

        # Экстремальные значения
        if df[col].dtype in ["float64", "int64"]:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            extreme_high = (
                df[col] > q99 * 1000
            ).sum()  # Значения в 1000 раз больше 99-го процентиля
            extreme_low = (
                df[col] < q01 / 1000
            ).sum()  # Значения в 1000 раз меньше 1-го процентиля

            if extreme_high > 0:
                issues.append(f"{extreme_high} экстремально больших")
            if extreme_low > 0:
                issues.append(f"{extreme_low} экстремально маленьких")

        if issues:
            problem_columns.append((col, issues))

    if problem_columns:
        print(f"  Найдено {len(problem_columns)} проблемных колонок:")
        for col, issues in problem_columns:
            print(f"    {col}: {', '.join(issues)}")
    else:
        print(f"  ✅ Проблемных колонок не найдено")

    # 6. Рекомендации по обработке
    print(f"\n=== РЕКОМЕНДАЦИИ ===")

    if inf_found:
        print(f"  🔧 Бесконечные значения найдены - нужна замена на конечные значения")
        print(f"     Варианты: заменить на NaN, затем на медиану/среднее")

    if nan_found:
        print(f"  🔧 NaN значения найдены - нужно заполнение пропусков")
        print(f"     Варианты: медиана для числовых, режим для категориальных")

    if "SPR" in df.columns and (
        np.isinf(df["SPR"]).sum() > 0 or (df["Pot"] == 0).sum() > 0
    ):
        print(f"  🔧 SPR содержит inf из-за Pot=0 - нужна специальная обработка")
        print(f"     Вариант: SPR = min(Stack/max(Pot, 0.01), max_reasonable_spr)")

    print(f"\n=== ВЛИЯНИЕ НА МОДЕЛЬ ===")
    print(f"  • NaN/inf в признаках → ошибки при нормализации (StandardScaler)")
    print(f"  • inf значения → некорректные веса в нейросети")
    print(f"  • Экстремальные значения → доминирование одного признака")
    print(f"  • Исправление НЕ исказит данные, если делать разумно:")
    print(f"    - inf → заменить на разумный максимум (например, 99-й процентиль)")
    print(f"    - NaN → заменить на медиану (робастная статистика)")
    print(f"    - SPR = inf → ограничить разумным максимумом (например, 100)")


if __name__ == "__main__":
    analyze_data_issues()
