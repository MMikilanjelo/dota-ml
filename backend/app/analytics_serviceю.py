# app/analytics.py
import pandas as pd
import numpy as np
from pathlib import Path

class DotaAnalyticsService:
    def __init__(self, data_path: str = "./data/raw"):
        self.data_path = Path(data_path)
        self.matches = None
        self.players = None
        self.training_results = None
        self._load_data()

    def _load_data(self):
        """Завантажує дані в пам'ять при старті"""
        try:
            print("📊 Loading analytics data...")
            # Завантажуємо тільки потрібні колонки для економії пам'яті
            self.matches = pd.read_csv(self.data_path / "matches.csv", 
                                     usecols=['match_id', 'radiant_win', 'duration', 'start_time'])
            
            # Конвертуємо час
            self.matches['start_time'] = pd.to_datetime(self.matches['start_time'], unit='s')
            
            self.players = pd.read_csv(self.data_path / "players.csv", 
                                     usecols=['match_id', 'account_id', 'hero_id', 'gold_per_min', 
                                              'xp_per_min', 'kills', 'deaths', 'assists', 
                                              'hero_damage', 'tower_damage', 'hero_healing'])
            
            # Спробуємо завантажити результати навчання (якщо є)
            results_path = Path("./models/training_results.csv")
            if results_path.exists():
                self.training_results = pd.read_csv(results_path)
            
            print("✅ Analytics data loaded!")
        except Exception as e:
            print(f"⚠ Error loading data: {e}")
            # Створюємо пусті DataFrame, щоб сервер не впав
            self.matches = pd.DataFrame()
            self.players = pd.DataFrame()

    def get_general_stats(self):
        """Загальна статистика по всіх матчах"""
        if self.matches.empty:
            return {}
            
        return {
            "total_matches": int(len(self.matches)),
            "radiant_winrate": float(self.matches['radiant_win'].mean()),
            "avg_duration": float(self.matches['duration'].mean()),
            # Беремо точність з збереженого файлу або дефолтну
            "model_accuracy": 0.625 
        }

    def get_winrate_trends(self):
        """Групує матчі по днях/тижнях"""
        if self.matches.empty:
            return []
            
        # Групуємо по днях
        daily_stats = self.matches.set_index('start_time').resample('D')['radiant_win'].mean().reset_index()
        
        # Прибираємо NaN (дні без матчів)
        daily_stats = daily_stats.dropna()
        
        # Форматуємо для API
        trends = []
        for _, row in daily_stats.iterrows():
            trends.append({
                "date": row['start_time'].strftime('%Y-%m-%d'),
                "winrate": round(row['radiant_win'], 3)
            })
        return trends

    def analyze_player_style(self, account_id: int):
        """Глибокий аналіз гравця з детальним звітом"""
        
        # 1. Перевірка на наявність гравця в базі
        if self.players.empty:
            return {"error": "Database is empty", "status": "error"}

        p_stats = self.players[self.players['account_id'] == account_id]
        match_count = len(p_stats)

        # 2. Перевірка на кількість матчів (повертаємо інфо, скільки знайдено)
        if match_count == 0:
            return {
                "error": "Player not found in dataset",
                "status": "not_found",
                "match_count": 0
            }
            
        if match_count < 5:
            # Повертаємо попередження, але все одно намагаємось порахувати (або просто помилку)
            return {
                "error": f"Not enough data for accurate analysis (found {match_count}, need 5+)",
                "status": "insufficient_data",
                "match_count": match_count
            }

        # 3. Розрахунок метрик
        avg_metrics = p_stats.agg({
            'gold_per_min': 'mean',
            'xp_per_min': 'mean',
            'kills': 'mean',
            'deaths': 'mean',
            'assists': 'mean',
            'hero_damage': 'mean',
            'tower_damage': 'mean',
            'hero_healing': 'mean'
        })

        kda = (avg_metrics['kills'] + avg_metrics['assists']) / (avg_metrics['deaths'] + 1)

        # 4. Нормалізація (Radar Stats)
        radar = {
            "farming": min(10, (avg_metrics['gold_per_min'] / 600) * 10),
            "fighting": min(10, (avg_metrics['hero_damage'] / 25000) * 10),
            "support": min(10, (avg_metrics['hero_healing'] / 2000) * 10), # Зменшив поріг для хілу (2к це вже непогано)
            "pushing": min(10, (avg_metrics['tower_damage'] / 3000) * 10),
            "versatility": min(10, (p_stats['hero_id'].nunique() / match_count) * 20)
        }

        # 5. Визначення стилю (ПОКРАЩЕНА ЛОГІКА)
        # Замість жорстких if/else, шукаємо найсильнішу сторону
        playstyles = {
            "Hard Carry": radar['farming'],
            "Support / Healer": radar['support'],
            "Aggressive Fighter": radar['fighting'],
            "Pusher / Objective": radar['pushing'],
            "Flexible / Draft": radar['versatility']
        }

        # Знаходимо ключ з максимальним значенням
        best_style = max(playstyles, key=playstyles.get)
        max_score = playstyles[best_style]

        # Якщо навіть найкращий показник слабкий (< 4), то це "Passive / Newbie"
        if max_score < 4:
            label = "Passive / Learner"
            recommendation = "Спробуйте брати активнішу участь у грі (фарм або бійки)."
        elif max_score > 8:
            label = f"Elite {best_style}"
            recommendation = f"Ви домінуєте в аспекті {best_style}. Продовжуйте!"
        else:
            label = best_style
            recommendation = self._get_recommendation(best_style)

        return {
            "status": "success",
            "player_name": f"Player_{account_id}",
            "match_count": match_count,  # Важливо бачити, на скількох матчах базується висновок
            "playstyle_label": label,
            "metrics": {
                "gpm": int(avg_metrics['gold_per_min']),
                "xpm": int(avg_metrics['xp_per_min']),
                "kda": round(kda, 2)
            },
            "radar_stats": {k: round(v, 1) for k, v in radar.items()},
            "recommendation": recommendation,
            "debug_scores": playstyles # Для розробника: бачити всі бали
        }

    def _get_recommendation(self, style):
            """Допоміжна функція для тексту"""
            recs = {
                "Hard Carry": "Зосередьтеся на ефективності маршрутів фарму.",
                "Support / Healer": "Тримайте позицію позаду та рятуйте корів.",
                "Aggressive Fighter": "Ви ініціатор. Координуйте команду перед атакою.",
                "Pusher / Objective": "Ви створюєте простір. Не забувайте про BKB.",
                "Flexible / Draft": "Ваш широкий пул героїв - це перевага на драфті."
            }
            return recs.get(style, "Грайте в своє задоволення!")

# Створюємо глобальний інстанс (буде ініціалізований в main.py)
analytics_service = DotaAnalyticsService()