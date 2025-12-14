"""
Скрипт для навчання ML моделей для всіх гіпотез
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')


class Dota2ModelTrainer:
    """
    Клас для навчання всіх моделей проекту
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Завантаження даних
        print("📊 Завантаження даних...")
        self.load_data()
    
    def load_data(self):
        """Завантаження та базова обробка даних"""
        self.matches = pd.read_csv(self.data_path / "matches.csv")
        self.players = pd.read_csv(self.data_path / "players.csv")
        self.hero_names = pd.read_csv(self.data_path / "hero_names.csv")
        self.player_ratings = pd.read_csv(self.data_path / "player_ratings.csv")
        
        print(f"✓ Завантажено {len(self.matches)} матчів")
        print(f"✓ Завантажено {len(self.players)} записів гравців")
    
    def preprocess_data(self):
        """
        Попередня обробка даних: створення колонок героїв з таблиці гравців
        """
        print("\n🔧 Попередня обробка даних...")
        
        # 1. Видалення матчів з пропущеними даними
        self.matches = self.matches.dropna(subset=['radiant_win', 'duration'])
        
        # 2. Трансформація players.csv у колонки r1_hero...d5_hero
        print("   ⟳ Трансформація списку гравців у колонки героїв...")
        
        # Розділяємо гравців на команди за player_slot
        # 0-127 = Radiant, 128+ = Dire
        # Сортуємо по match_id та player_slot, щоб порядок був фіксований
        players_sorted = self.players.sort_values(['match_id', 'player_slot'])
        
        # Створюємо допоміжні датафрейми
        radiant_players = players_sorted[players_sorted['player_slot'] < 128]
        dire_players = players_sorted[players_sorted['player_slot'] >= 128]
        
        # Групуємо по match_id і збираємо героїв у списки
        r_heroes = radiant_players.groupby('match_id')['hero_id'].apply(list)
        d_heroes = dire_players.groupby('match_id')['hero_id'].apply(list)
        
        # Створюємо DataFrame з героями Radiant (r1_hero ... r5_hero)
        r_cols = pd.DataFrame(r_heroes.tolist(), index=r_heroes.index).add_prefix('r')
        r_cols.columns = [f'r{i+1}_hero' for i in range(len(r_cols.columns))]
        
        # Створюємо DataFrame з героями Dire (d1_hero ... d5_hero)
        d_cols = pd.DataFrame(d_heroes.tolist(), index=d_heroes.index).add_prefix('d')
        d_cols.columns = [f'd{i+1}_hero' for i in range(len(d_cols.columns))]
        
        # 3. Об'єднуємо все в matches
        heroes_df = pd.concat([r_cols, d_cols], axis=1)
        
        # Використовуємо inner join, щоб залишити тільки матчі з повним складом (де є інфо про героїв)
        self.matches = self.matches.merge(heroes_df, left_on='match_id', right_index=True, how='inner')
        
        # Перевірка на цілісність (має бути 10 героїв)
        hero_columns = [f'r{i}_hero' for i in range(1, 6)] + [f'd{i}_hero' for i in range(1, 6)]
        self.matches = self.matches.dropna(subset=hero_columns)
        
        print(f"✓ Оброблено {len(self.matches)} матчів з повним складом героїв")
        
        # 4. (Опціонально) Оновлюємо full_data для інших гіпотез, якщо потрібно
        self.full_data = self.matches.merge(
            self.players, 
            left_on='match_id', 
            right_on='match_id',
            how='left'
        )
    def generate_and_save_matrices(self, output_path: str = "models"):
        """
        Розрахунок та збереження матриць синергії та контр-піків
        на основі реальних даних матчів.
        """
        print("\n🧮 Генерація матриць синергії та контр-піків...")
        
        # Визначаємо максимальний ID героя
        max_hero_id = 130 # З запасом
        
        # Ініціалізація матриць
        # synergy[A][B] = Winrate коли A і B в одній команді
        synergy_matrix = np.zeros((max_hero_id, max_hero_id))
        synergy_counts = np.zeros((max_hero_id, max_hero_id))
        
        # counter[A][B] = Winrate героя A проти героя B
        counter_matrix = np.zeros((max_hero_id, max_hero_id))
        counter_counts = np.zeros((max_hero_id, max_hero_id))

        print("   ⟳ Обробка матчів (це може зайняти час)...")
        
        # Проходимо по матчах (використовуємо self.matches, де вже є колонки r1_hero...d5_hero)
        # Переконайтеся, що викликали preprocess_data() перед цим
        
        hero_cols_radiant = [f'r{i}_hero' for i in range(1, 6)]
        hero_cols_dire = [f'd{i}_hero' for i in range(1, 6)]
        
        for row in self.matches.itertuples():
            radiant_win = row.radiant_win
            
            # Отримуємо ID героїв (ігноруємо NaN)
            r_heroes = [int(getattr(row, c)) for c in hero_cols_radiant if pd.notna(getattr(row, c))]
            d_heroes = [int(getattr(row, c)) for c in hero_cols_dire if pd.notna(getattr(row, c))]
            
            # --- 1. СИНЕРГІЯ (Союзники) ---
            # Для Radiant
            for i in range(len(r_heroes)):
                for j in range(i + 1, len(r_heroes)):
                    h1, h2 = r_heroes[i], r_heroes[j]
                    if h1 < max_hero_id and h2 < max_hero_id:
                        synergy_counts[h1][h2] += 1
                        synergy_counts[h2][h1] += 1
                        if radiant_win:
                            synergy_matrix[h1][h2] += 1
                            synergy_matrix[h2][h1] += 1
                            
            # Для Dire
            for i in range(len(d_heroes)):
                for j in range(i + 1, len(d_heroes)):
                    h1, h2 = d_heroes[i], d_heroes[j]
                    if h1 < max_hero_id and h2 < max_hero_id:
                        synergy_counts[h1][h2] += 1
                        synergy_counts[h2][h1] += 1
                        if not radiant_win: # Dire win
                            synergy_matrix[h1][h2] += 1
                            synergy_matrix[h2][h1] += 1

            # --- 2. КОНТР-ПІКИ (Вороги) ---
            for rh in r_heroes:
                for dh in d_heroes:
                    if rh < max_hero_id and dh < max_hero_id:
                        counter_counts[rh][dh] += 1
                        counter_counts[dh][rh] += 1
                        
                        if radiant_win:
                            counter_matrix[rh][dh] += 1 # Radiant hero beat Dire hero
                        else:
                            counter_matrix[dh][rh] += 1 # Dire hero beat Radiant hero

        # Нормалізація (розрахунок середнього)
        # Уникаємо ділення на нуль
        with np.errstate(divide='ignore', invalid='ignore'):
            synergy_matrix = np.divide(synergy_matrix, synergy_counts)
            counter_matrix = np.divide(counter_matrix, counter_counts)
            
        # Заповнюємо NaN (де не було ігор) значенням 0.5 (рівні шанси)
        synergy_matrix = np.nan_to_num(synergy_matrix, nan=0.5)
        counter_matrix = np.nan_to_num(counter_matrix, nan=0.5)

        # Збереження
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        np.save(output_dir / "synergy_matrix.npy", synergy_matrix)
        np.save(output_dir / "counter_matrix.npy", counter_matrix)
        
        print(f"✓ Матриці збережено в {output_dir}")
    # ==========================================
    # ГІПОТЕЗА 1: Вплив складу героїв
    # ==========================================
    def train_hypothesis1_hero_composition(self):
        """
        Навчання моделі для прогнозування переможця на основі героїв
        """
        print("\n" + "="*60)
        print("🎯 ГІПОТЕЗА 1: Вплив складу героїв на результат матчу")
        print("="*60)
        
        # Підготовка даних
        X, y = self.prepare_hero_features()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Навчання різних моделей
        models_to_train = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models_to_train.items():
            print(f"\n📈 Навчання {name}...")
            model.fit(X_train, y_train)
            
            # Оцінка моделі
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC:  {roc_auc:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
        
        # Зберігаємо найкращу модель
        self.models['hero_composition'] = best_model
        self.results['hypothesis1'] = {
            'best_model': type(best_model).__name__,
            'accuracy': best_score,
            'conclusion': 'Склад героїв має значний вплив на результат матчу'
        }
        
        print(f"\n✓ Найкраща модель: {type(best_model).__name__} з accuracy {best_score:.4f}")
    
    def prepare_hero_features(self):
        """
        Підготовка фічів на основі героїв
        """
        # Створюємо one-hot encoding для кожного героя
        # Припускаємо, що є колонки r1_hero, r2_hero, ... r5_hero для Radiant
        # та d1_hero, d2_hero, ... d5_hero для Dire
        
        hero_columns = [f'r{i}_hero' for i in range(1, 6)] + [f'd{i}_hero' for i in range(1, 6)]
        
        # Фільтруємо матчі де є всі герої
        valid_matches = self.matches.dropna(subset=hero_columns + ['radiant_win'])
        
        # Створюємо матрицю героїв (binary encoding)
        max_hero_id = 120  # Максимальний ID героя
        X = np.zeros((len(valid_matches), max_hero_id * 2))
        
        for idx, row in enumerate(valid_matches.itertuples()):
            # Radiant heroes
            for i in range(1, 6):
                hero_id = getattr(row, f'r{i}_hero')
                if pd.notna(hero_id) and 0 <= int(hero_id) < max_hero_id:
                    X[idx, int(hero_id)] = 1
            
            # Dire heroes
            for i in range(1, 6):
                hero_id = getattr(row, f'd{i}_hero')
                if pd.notna(hero_id) and 0 <= int(hero_id) < max_hero_id:
                    X[idx, max_hero_id + int(hero_id)] = 1
        
        y = valid_matches['radiant_win'].astype(int).values
        
        return X, y
    
    # ==========================================
    # ГІПОТЕЗА 2: Вплив рейтингу гравців
    # ==========================================
    def train_hypothesis2_player_ratings(self):
        """
        Дослідження впливу рейтингу гравців на результат
        """
        print("\n" + "="*60)
        print("🎯 ГІПОТЕЗА 2: Вплив рейтингу гравців на перемогу")
        print("="*60)
        
        # Агрегуємо рейтинги по командах
        team_ratings = self.aggregate_team_ratings()
        
        if team_ratings is None or len(team_ratings) < 100:
            print("⚠ Недостатньо даних для аналізу рейтингів")
            return
        
        # Підготовка даних
        X = team_ratings[['radiant_avg_rating', 'dire_avg_rating', 'rating_diff']].values
        y = team_ratings['radiant_win'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Навчання Gradient Boosting
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Оцінка
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n📊 Результати:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        # Кореляційний аналіз
        correlation = team_ratings[['rating_diff', 'radiant_win']].corr().iloc[0, 1]
        print(f"  Кореляція rating_diff з radiant_win: {correlation:.4f}")
        
        self.models['player_ratings'] = model
        self.results['hypothesis2'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'correlation': correlation,
            'conclusion': f'Рейтинг гравців має {"сильну" if abs(correlation) > 0.6 else "помірну"} кореляцію з результатом'
        }
    
    def aggregate_team_ratings(self):
        """
        Агрегація рейтингів по командах
        """
        try:
            # Merge ratings з players
            players_with_ratings = self.players.merge(
                self.player_ratings,
                on='account_id',
                how='left'
            )
            
            # Групуємо по матчам та командам
            team_stats = []
            
            for match_id in self.matches['match_id'].unique()[:10000]:  # Беремо перші 10к матчів
                match_players = players_with_ratings[players_with_ratings['match_id'] == match_id]
                
                if len(match_players) < 10:
                    continue
                
                radiant_players = match_players[match_players['player_slot'] < 128]
                dire_players = match_players[match_players['player_slot'] >= 128]
                
                if len(radiant_players) == 5 and len(dire_players) == 5:
                    radiant_avg = radiant_players['trueskill_mu'].mean()
                    dire_avg = dire_players['trueskill_mu'].mean()
                    
                    if pd.notna(radiant_avg) and pd.notna(dire_avg):
                        match_info = self.matches[self.matches['match_id'] == match_id].iloc[0]
                        
                        team_stats.append({
                            'match_id': match_id,
                            'radiant_avg_rating': radiant_avg,
                            'dire_avg_rating': dire_avg,
                            'rating_diff': radiant_avg - dire_avg,
                            'radiant_win': int(match_info['radiant_win'])
                        })
            
            return pd.DataFrame(team_stats)
        
        except Exception as e:
            print(f"⚠ Помилка агрегації рейтингів: {e}")
            return None
    
    # ==========================================
    # ГІПОТЕЗА 3: Синергія героїв
    # ==========================================
    def train_hypothesis3_hero_synergy(self):
        """
        Пошук найкращих комбінацій героїв за допомогою Association Rules
        """
        print("\n" + "="*60)
        print("🎯 ГІПОТЕЗА 3: Синергія героїв (Hero Combinations)")
        print("="*60)
        
        # Підготовка даних для Association Rules
        hero_basket = self.prepare_hero_basket()
        
        if hero_basket is None or len(hero_basket) < 100:
            print("⚠ Недостатньо даних для аналізу синергії")
            return
        
        # Apriori algorithm
        print("\n🔍 Пошук частих комбінацій героїв...")
        frequent_itemsets = apriori(hero_basket, min_support=0.01, use_colnames=True)
        
        print(f"✓ Знайдено {len(frequent_itemsets)} частих комбінацій")
        
        # Association rules
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            rules = rules.sort_values('lift', ascending=False)
            
            print(f"\n📊 Топ-10 найкращих комбінацій:")
            for idx, rule in rules.head(10).iterrows():
                print(f"  {list(rule['antecedents'])} → {list(rule['consequents'])}")
                print(f"    Support: {rule['support']:.4f}, Confidence: {rule['confidence']:.4f}, Lift: {rule['lift']:.4f}")
            
            self.results['hypothesis3'] = {
                'total_combinations': len(frequent_itemsets),
                'strong_rules': len(rules[rules['lift'] > 1.2]),
                'top_synergy': list(rules.iloc[0]['antecedents']) if len(rules) > 0 else [],
                'conclusion': 'Виявлено значущі синергії між героями'
            }
    
    def prepare_hero_basket(self):
        """
        Підготовка даних у форматі basket для Apriori
        """
        try:
            hero_columns = [f'r{i}_hero' for i in range(1, 6)]
            winning_matches = self.matches[self.matches['radiant_win'] == True]
            
            # Створюємо binary matrix для winning combinations
            basket_data = []
            
            for idx, row in winning_matches[hero_columns].head(5000).iterrows():
                heroes = [int(h) for h in row.values if pd.notna(h)]
                hero_set = {f"Hero_{h}": 1 for h in heroes if 0 <= h < 120}
                basket_data.append(hero_set)
            
            # Конвертуємо в DataFrame
            basket_df = pd.DataFrame(basket_data).fillna(0).astype(bool)
            
            return basket_df
        
        except Exception as e:
            print(f"⚠ Помилка підготовки basket: {e}")
            return None
    
    # ==========================================
    # ГІПОТЕЗА 4: Вплив тривалості гри
    # ==========================================
    def train_hypothesis4_game_duration(self):
        """
        Аналіз впливу тривалості матчу на результат
        """
        print("\n" + "="*60)
        print("🎯 ГІПОТЕЗА 4: Вплив тривалості гри на результат")
        print("="*60)
        
        # Підготовка даних
        valid_matches = self.matches.dropna(subset=['duration', 'radiant_win'])
        
        # Категоризація тривалості
        valid_matches['duration_category'] = pd.cut(
            valid_matches['duration'] / 60,  # конвертуємо в хвилини
            bins=[0, 20, 30, 40, 60, 100],
            labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
        )
        
        # Аналіз winrate по категоріях
        duration_analysis = valid_matches.groupby('duration_category').agg({
            'radiant_win': ['mean', 'count']
        }).round(4)
        
        print("\n📊 Winrate Radiant по тривалості матчу:")
        print(duration_analysis)
        
        # Chi-square test
        from scipy.stats import chi2_contingency
        
        contingency_table = pd.crosstab(
            valid_matches['duration_category'],
            valid_matches['radiant_win']
        )
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        print(f"\n📈 Chi-square test:")
        print(f"  Chi² = {chi2:.4f}")
        print(f"  p-value = {p_value:.4f}")
        print(f"  Статистично значущий: {'Так' if p_value < 0.05 else 'Ні'}")
        
        self.results['hypothesis4'] = {
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': f'Тривалість матчу {"має" if p_value < 0.05 else "не має"} статистично значущий вплив'
        }
    
    # ==========================================
    # ГІПОТЕЗА 5: Географічний вплив
    # ==========================================
    def train_hypothesis5_regional_analysis(self):
        """
        Регіональний аналіз результатів
        """
        print("\n" + "="*60)
        print("🎯 ГІПОТЕЗА 5: Географічний вплив (регіон/кластер)")
        print("="*60)
        
        if 'cluster' not in self.matches.columns:
            print("⚠ Немає даних про кластери/регіони")
            return
        
        # Аналіз по регіонах
        regional_stats = self.matches.groupby('cluster').agg({
            'radiant_win': 'mean',
            'match_id': 'count',
            'duration': 'mean'
        }).round(4)
        
        regional_stats.columns = ['radiant_winrate', 'match_count', 'avg_duration']
        regional_stats = regional_stats[regional_stats['match_count'] > 100]
        
        print("\n📊 Статистика по регіонах (кластерах):")
        print(regional_stats.head(10))
        
        # ANOVA test
        from scipy.stats import f_oneway
        
        groups = [
            group['radiant_win'].values 
            for name, group in self.matches.groupby('cluster') 
            if len(group) > 100
        ]
        
        if len(groups) > 2:
            f_stat, p_value = f_oneway(*groups)
            
            print(f"\n📈 ANOVA test:")
            print(f"  F-statistic = {f_stat:.4f}")
            print(f"  p-value = {p_value:.4f}")
            
            self.results['hypothesis5'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'regions_analyzed': len(groups),
                'conclusion': f'Регіон {"має" if p_value < 0.05 else "не має"} статистично значущий вплив'
            }
    
    # ==========================================
    # ЗБЕРЕЖЕННЯ МОДЕЛЕЙ
    # ==========================================
    def save_models(self, output_path: str = "models"):
        """
        Збереження всіх навчених моделей
        """
        print("\n💾 Збереження моделей...")
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            filepath = output_dir / f"{name}_model.pkl"
            joblib.dump(model, filepath)
            print(f"  ✓ {name} → {filepath}")
        
        # Збереження результатів
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(output_dir / "training_results.csv")
        print(f"  ✓ Результати → {output_dir / 'training_results.csv'}")
    
    def train_all(self):
        """
        Навчання всіх моделей для всіх гіпотез
        """
        self.preprocess_data()
        
        self.train_hypothesis1_hero_composition()
        self.train_hypothesis2_player_ratings()
        self.train_hypothesis3_hero_synergy()
        self.train_hypothesis4_game_duration()
        self.train_hypothesis5_regional_analysis()
        
        self.generate_and_save_matrices()
        self.save_models()
        
        print("\n" + "="*60)
        print("✅ НАВЧАННЯ ЗАВЕРШЕНО!")
        print("="*60)
        print("\nПідсумок результатів:")
        for hypothesis, results in self.results.items():
            print(f"\n{hypothesis}:")
            print(f"  {results.get('conclusion', 'Аналіз виконано')}")


if __name__ == "__main__":
    # Запуск навчання
    trainer = Dota2ModelTrainer(data_path="./data/raw")
    trainer.train_all()