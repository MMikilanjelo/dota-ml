from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict , Any
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
from .helper import (
    generate_synergy_matrix,
    generate_counter_matrix,
    generate_fallback_hero_data,
    generate_hero_stats,
    load_hero_statistics,
    calculate_team_composition_score,
    get_hero_role_distribution,
    calculate_game_phase_advantage,
    get_recommended_items_for_hero,
    analyze_draft_timing
)
from app.analytics_serviceю import analytics_service
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Dota 2 ML Backend...")
    yield
    print("Shutting down...")

app = FastAPI(
    title="Dota 2 Match Prediction API",
    description="API для прогнозування результатів матчів Dota 2",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================
# ЗАВАНТАЖЕННЯ МОДЕЛЕЙ ТА ДАНИХ
# =============================================
MODEL_PATH = Path("models")
DATA_PATH = Path("data")
models = {}
hero_data = {}
synergy_matrix = None
counter_matrix = None

# Завантаження моделей
try:
    if (MODEL_PATH / "hero_composition_model.pkl").exists():
        models['hero_composition'] = joblib.load(MODEL_PATH / "hero_composition_model.pkl")
        print("Модель Hero Composition завантажено")
    else:
        print("Файл hero_composition_model.pkl не знайдено")
except Exception as e:
    print(f"Помилка hero_composition: {e}")

try:
    if (MODEL_PATH / "player_ratings_model.pkl").exists():
        models['player_ratings'] = joblib.load(MODEL_PATH / "player_ratings_model.pkl")
        print("Модель Player Ratings завантажено")
    else:
        print("Файл player_ratings_model.pkl не знайдено")
except Exception as e:
    print(f"Помилка player_ratings: {e}")

# Завантаження даних про героїв
try:
    # Спроба завантажити реальні дані
    if (DATA_PATH / "raw" / "hero_names.csv").exists():
        heroes_df = pd.read_csv(DATA_PATH / "raw" / "hero_names.csv")
        for _, row in heroes_df.iterrows():
            hero_data[int(row.get('hero_id', 0))] = {
                'name': row.get('localized_name', f"Hero_{row.get('hero_id', 0)}"),
                'id': int(row.get('hero_id', 0))
            }
        print(f"✓ Завантажено {len(hero_data)} героїв")
    else:
        # Fallback дані
        hero_data = generate_fallback_hero_data()
        print("⚠ Використовуються fallback дані героїв")
except Exception as e:
    hero_data = generate_fallback_hero_data()
    print(f"⚠ Помилка завантаження героїв: {e}")

try:
    if (DATA_PATH / "raw" / "hero_names.csv").exists():
        heroes_df = pd.read_csv(DATA_PATH / "raw" / "hero_names.csv")
        for _, row in heroes_df.iterrows():
            hero_data[int(row.get('hero_id', 0))] = {
                'name': row.get('localized_name', f"Hero_{row.get('hero_id', 0)}"),
                'id': int(row.get('hero_id', 0))
            }
    else:
        hero_data = generate_fallback_hero_data()
except:
    hero_data = generate_fallback_hero_data()

try:
    if (MODEL_PATH / "synergy_matrix.npy").exists():
        synergy_matrix = np.load(MODEL_PATH / "synergy_matrix.npy")
    else:
        synergy_matrix = generate_synergy_matrix()
        
    if (MODEL_PATH / "counter_matrix.npy").exists():
        counter_matrix = np.load(MODEL_PATH / "counter_matrix.npy")
    else:
        counter_matrix = generate_counter_matrix()
except:
    synergy_matrix = generate_synergy_matrix()
    counter_matrix = generate_counter_matrix()

hero_stats = load_hero_statistics()

# =============================================
# PYDANTIC МОДЕЛІ
# =============================================

class MatchPredictionRequest(BaseModel):
    radiant_heroes: List[int]
    dire_heroes: List[int]
    radiant_avg_rating: Optional[float] = 0.0  # Default 0.0
    dire_avg_rating: Optional[float] = 0.0     # Default 0.0

class MatchPredictionResponse(BaseModel):
    radiant_win_probability: float
    dire_win_probability: float
    predicted_winner: str
    confidence: float
    analysis: Dict[str, Any] = {}


class HeroStatsResponse(BaseModel):
    hero_id: int
    hero_name: str
    total_picks: int
    total_wins: int
    win_rate: float
    avg_kda: float
    popular_positions: List[str]
    roles: List[str]


class TeamCompositionAnalysis(BaseModel):
    radiant_heroes: List[int]
    dire_heroes: List[int]
    radiant_synergy: float
    dire_synergy: float
    radiant_advantages: List[str]
    dire_advantages: List[str]
    counter_picks: List[Dict]
    recommendations: List[str]


class HeroSynergyResponse(BaseModel):
    heroes: List[int]
    synergy_score: float
    rating: str
    best_pairs: List[Dict]
    recommendations: List[str]


class RegionalAnalysis(BaseModel):
    region: str
    total_matches: int
    radiant_winrate: float
    dire_winrate: float
    top_heroes: List[Dict]
    meta_trends: str
    avg_game_duration: int


# =============================================
# ОСНОВНІ ENDPOINTS
# =============================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "message": "Dota 2 ML API v2.0 - Full Implementation",
        "models_loaded": len(models) > 0,
        "heroes_loaded": len(hero_data),
        "endpoints": [
            "/predict/match-winner",
            "/predict/hero-synergy",
            "/stats/hero/{hero_id}",
            "/analysis/regional",
            "/analysis/team-composition",
            "/heroes/list",
            "/heroes/search",
            
        ]
    }


@app.post("/predict/match-winner", response_model=MatchPredictionResponse)
async def predict_match_winner(request: MatchPredictionRequest):
    # Валідація
    if len(request.radiant_heroes) != 5 or len(request.dire_heroes) != 5:
        raise HTTPException(status_code=400, detail="Потрібно по 5 героїв у кожній команді")

    try:
        probabilities = []
        weights = []

        # --- 1. МОДЕЛЬ ГЕРОЇВ (Hero Composition) ---
        if 'hero_composition' in models:
            # Готуємо ТІЛЬКИ вектор героїв (240 елементів)
            hero_features = np.zeros(240)
            for h in request.radiant_heroes:
                if 0 <= h < 120: hero_features[h] = 1
            for h in request.dire_heroes:
                if 0 <= h < 120: hero_features[120 + h] = 1
            
            # Predict
            hero_prob = float(models['hero_composition'].predict_proba([hero_features])[0][1])
            probabilities.append(hero_prob)
            weights.append(0.7) # Вага моделі героїв (70%)
        
        # --- 2. МОДЕЛЬ РЕЙТИНГІВ (Player Ratings) ---
        # Використовуємо тільки якщо передані ненульові рейтинги
        if 'player_ratings' in models and request.radiant_avg_rating > 0:
            # Готуємо фічі: [radiant_avg, dire_avg, diff]
            rating_diff = request.radiant_avg_rating - request.dire_avg_rating
            rating_features = np.array([[
                request.radiant_avg_rating,
                request.dire_avg_rating,
                rating_diff
            ]])
            
            # Predict (Regressor повертає число, яке може бути >1 або <0, тому обрізаємо)
            rating_pred = float(models['player_ratings'].predict(rating_features)[0])
            rating_prob = np.clip(rating_pred, 0.0, 1.0) # Обмежуємо в межах [0, 1]
            
            probabilities.append(rating_prob)
            weights.append(0.3) # Вага моделі рейтингів (30%)

        # --- 3. РОЗРАХУНОК ФІНАЛЬНОЇ ЙМОВІРНОСТІ ---
        if not probabilities:
            # Fallback якщо моделей немає
            final_prob = calculate_win_probability_fallback(request.radiant_heroes, request.dire_heroes)
        else:
            # Зважене середнє (Weighted Average)
            final_prob = np.average(probabilities, weights=weights)

        # Результат
        dire_prob = 1 - final_prob
        predicted_winner = "Radiant" if final_prob > 0.5 else "Dire"
        confidence = max(final_prob, dire_prob)
        
        # Аналіз (використовуємо існуючу функцію)
        # Переконайтесь, що generate_match_analysis визначена у вашому helper.py або в цьому файлі
        analysis = {
             "models_used": len(probabilities),
             "hero_model_prob": round(probabilities[0], 4) if probabilities else "N/A",
             "rating_model_prob": round(probabilities[1], 4) if len(probabilities) > 1 else "N/A"
        }

        return MatchPredictionResponse(
            radiant_win_probability=round(final_prob, 4),
            dire_win_probability=round(dire_prob, 4),
            predicted_winner=predicted_winner,
            confidence=round(confidence, 4),
            analysis=analysis
        )

    except Exception as e:
        import traceback
        traceback.print_exc() # Вивід помилки в консоль сервера
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/hero-synergy", response_model=HeroSynergyResponse)
async def predict_hero_synergy(heroes: List[int]):
    """
    Детальний аналіз синергії команди героїв
    """
    if len(heroes) != 5:
        raise HTTPException(status_code=400, detail="Потрібно 5 героїв")
    
    if len(set(heroes)) != 5:
        raise HTTPException(status_code=400, detail="Герої не повинні повторюватись")
    
    # Розрахунок синергії
    synergy_score = calculate_synergy(heroes)
    
    # Визначення рейтингу
    if synergy_score > 0.7:
        rating = "Excellent"
    elif synergy_score > 0.6:
        rating = "Good"
    elif synergy_score > 0.4:
        rating = "Average"
    else:
        rating = "Poor"
    
    # Найкращі пари героїв
    best_pairs = find_best_hero_pairs(heroes)
    
    # Рекомендації
    recommendations = generate_synergy_recommendations(heroes, synergy_score)
    
    return HeroSynergyResponse(
        heroes=heroes,
        synergy_score=round(synergy_score, 3),
        rating=rating,
        best_pairs=best_pairs,
        recommendations=recommendations
    )


@app.get("/stats/hero/{hero_id}", response_model=HeroStatsResponse)
async def get_hero_stats(hero_id: int):
    """
    Детальна статистика героя
    """
    if hero_id < 1 or hero_id > 130:
        raise HTTPException(status_code=400, detail="Невірний ID героя")
    
    # Отримуємо статистику з кешу або генеруємо
    stats = hero_stats.get(hero_id, generate_hero_stats(hero_id))
    hero_name = hero_data.get(hero_id, {}).get('name', f"Hero_{hero_id}")
    
    return HeroStatsResponse(
        hero_id=hero_id,
        hero_name=hero_name,
        total_picks=stats['total_picks'],
        total_wins=stats['total_wins'],
        win_rate=round(stats['win_rate'], 2),
        avg_kda=round(stats['avg_kda'], 2),
        popular_positions=stats['positions'],
        roles=stats['roles']
    )


@app.get("/analysis/regional", response_model=RegionalAnalysis)
async def regional_analysis(region: Optional[str] = None):
    """
    Регіональний аналіз winrate та мета-героїв
    """
    # Якщо є реальні дані, використовуємо їх
    try:
        if (DATA_PATH / "raw" / "matches.csv").exists():
            matches_df = pd.read_csv(DATA_PATH / "raw" / "matches.csv")
            
            # Фільтруємо по регіону якщо вказано
            if region and 'cluster' in matches_df.columns:
                region_map = {'EU': [111, 112], 'US': [121, 122], 'SEA': [131, 132]}
                if region in region_map:
                    matches_df = matches_df[matches_df['cluster'].isin(region_map[region])]
            
            total_matches = len(matches_df)
            radiant_wins = matches_df['radiant_win'].sum() if 'radiant_win' in matches_df.columns else total_matches * 0.52
            radiant_winrate = (radiant_wins / total_matches * 100) if total_matches > 0 else 52.0
            avg_duration = int(matches_df['duration'].mean()) if 'duration' in matches_df.columns else 2400
            
        else:
            raise FileNotFoundError("No data")
    except:
        # Fallback дані
        total_matches = 50000
        radiant_winrate = 52.1
        avg_duration = 2400
    
    # Топ герої (з статистики або згенеровані)
    top_heroes = get_top_heroes_by_region(region)
    
    # Мета тренди
    meta_trends = analyze_meta_trends(top_heroes)
    
    return RegionalAnalysis(
        region=region or "all",
        total_matches=total_matches,
        radiant_winrate=round(radiant_winrate, 2),
        dire_winrate=round(100 - radiant_winrate, 2),
        top_heroes=top_heroes,
        meta_trends=meta_trends,
        avg_game_duration=avg_duration
    )


@app.post("/analysis/team-composition", response_model=TeamCompositionAnalysis)
async def analyze_team_composition(
    radiant_heroes: List[int],
    dire_heroes: List[int]
):
    """
    Повний аналіз складу обох команд
    """
    if len(radiant_heroes) != 5 or len(dire_heroes) != 5:
        raise HTTPException(status_code=400, detail="Кожна команда повинна мати 5 героїв")
    
    # Синергія команд
    radiant_synergy = calculate_synergy(radiant_heroes)
    dire_synergy = calculate_synergy(dire_heroes)
    
    # Переваги команд
    radiant_advantages = analyze_team_advantages(radiant_heroes, dire_heroes)
    dire_advantages = analyze_team_advantages(dire_heroes, radiant_heroes)
    
    # Контр-піки
    counter_picks = find_counter_picks_detailed(radiant_heroes, dire_heroes)
    
    # Рекомендації
    recommendations = generate_team_recommendations(
        radiant_heroes, dire_heroes, 
        radiant_synergy, dire_synergy
    )
    
    return TeamCompositionAnalysis(
        radiant_heroes=radiant_heroes,
        dire_heroes=dire_heroes,
        radiant_synergy=round(radiant_synergy, 3),
        dire_synergy=round(dire_synergy, 3),
        radiant_advantages=radiant_advantages,
        dire_advantages=dire_advantages,
        counter_picks=counter_picks,
        recommendations=recommendations
    )


@app.get("/heroes/list")
async def get_heroes_list(
    limit: int = Query(50, ge=1, le=130),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("id", regex="^(id|name|winrate|pickrate)$")
):
    """
    Список всіх героїв з можливістю сортування
    """
    heroes_list = []
    
    for hero_id, hero_info in hero_data.items():
        stats = hero_stats.get(hero_id, generate_hero_stats(hero_id))
        heroes_list.append({
            'id': hero_id,
            'name': hero_info['name'],
            'win_rate': stats['win_rate'],
            'pick_rate': stats['total_picks'] / 50000 * 100,  # Відсоток від всіх матчів
            'total_picks': stats['total_picks']
        })
    
    # Сортування
    if sort_by == "name":
        heroes_list.sort(key=lambda x: x['name'])
    elif sort_by == "winrate":
        heroes_list.sort(key=lambda x: x['win_rate'], reverse=True)
    elif sort_by == "pickrate":
        heroes_list.sort(key=lambda x: x['pick_rate'], reverse=True)
    else:
        heroes_list.sort(key=lambda x: x['id'])
    
    # Пагінація
    total = len(heroes_list)
    heroes_list = heroes_list[offset:offset + limit]
    
    return {
        'heroes': heroes_list,
        'total': total,
        'limit': limit,
        'offset': offset
    }


@app.get("/heroes/search")
async def search_heroes(query: str = Query(..., min_length=2)):
    """
    Пошук героїв по назві
    """
    results = []
    query_lower = query.lower()
    
    for hero_id, hero_info in hero_data.items():
        if query_lower in hero_info['name'].lower():
            stats = hero_stats.get(hero_id, generate_hero_stats(hero_id))
            results.append({
                'id': hero_id,
                'name': hero_info['name'],
                'win_rate': stats['win_rate'],
                'total_picks': stats['total_picks']
            })
    
    return {'results': results, 'count': len(results)}

@app.get("/player/{account_id}/playstyle")
def get_player_playstyle(account_id: str):
    try:
        acc_id_int = int(account_id)
        if acc_id_int == 0:
             raise HTTPException(status_code=404, detail="Anonymous player (ID 0)")

        result = analytics_service.analyze_player_style(acc_id_int)
        
        # Обробка статусів
        if result.get("status") == "error":
             raise HTTPException(status_code=500, detail=result['error'])
             
        if result.get("status") == "not_found":
             raise HTTPException(status_code=404, detail=f"Player {account_id} not found in dataset")
             
        # Якщо даних мало, але вони є - повертаємо результат (можна додати warning на фронтенді)
        return result
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Account ID")

@app.get("/stats/general")
def get_general_stats():
    return analytics_service.get_general_stats()

@app.get("/stats/trends")
def get_trends():
    return analytics_service.get_winrate_trends()



# =============================================
# ДОПОМІЖНІ ФУНКЦІЇ
# =============================================

def prepare_match_features(
    radiant_heroes: List[int],
    dire_heroes: List[int],
    radiant_rating: Optional[float],
    dire_rating: Optional[float]
) -> np.ndarray:
    """Підготовка фічів для моделі"""
    features = np.zeros(240)
    
    for hero_id in radiant_heroes:
        if 0 <= hero_id < 120:
            features[hero_id] = 1
    
    for hero_id in dire_heroes:
        if 0 <= hero_id < 120:
            features[120 + hero_id] = 1
    
    if radiant_rating and dire_rating:
        features = np.append(features, [radiant_rating, dire_rating])
    
    return features


def calculate_synergy(heroes: List[int]) -> float:
    """Розрахунок синергії команди"""
    if synergy_matrix is None:
        return 0.5
    
    total_synergy = 0
    pairs = 0
    
    for i in range(len(heroes)):
        for j in range(i + 1, len(heroes)):
            h1, h2 = heroes[i], heroes[j]
            if 0 <= h1 < 120 and 0 <= h2 < 120:
                total_synergy += synergy_matrix[h1, h2]
                pairs += 1
    
    return total_synergy / pairs if pairs > 0 else 0.5


def calculate_win_probability_fallback(radiant: List[int], dire: List[int]) -> float:
    """Fallback розрахунок ймовірності без моделі"""
    radiant_synergy = calculate_synergy(radiant)
    dire_synergy = calculate_synergy(dire)
    
    # Враховуємо контр-піки
    radiant_advantage = calculate_counter_advantage(radiant, dire)
    dire_advantage = calculate_counter_advantage(dire, radiant)
    
    # Комбінований скор
    radiant_score = radiant_synergy * 0.6 + radiant_advantage * 0.4
    dire_score = dire_synergy * 0.6 + dire_advantage * 0.4
    
    # Нормалізація до ймовірності
    total = radiant_score + dire_score
    return radiant_score / total if total > 0 else 0.5


def calculate_counter_advantage(team: List[int], enemies: List[int]) -> float:
    """Розрахунок переваги від контр-піків"""
    if counter_matrix is None:
        return 0.5
    
    total_advantage = 0
    count = 0
    
    for hero in team:
        for enemy in enemies:
            if 0 <= hero < 120 and 0 <= enemy < 120:
                total_advantage += counter_matrix[hero, enemy]
                count += 1
    
    return total_advantage / count if count > 0 else 0.5


def find_best_hero_pairs(heroes: List[int]) -> List[Dict]:
    """Знаходження найкращих пар героїв"""
    pairs = []
    
    for i in range(len(heroes)):
        for j in range(i + 1, len(heroes)):
            h1, h2 = heroes[i], heroes[j]
            if synergy_matrix is not None and 0 <= h1 < 120 and 0 <= h2 < 120:
                synergy = synergy_matrix[h1, h2]
                pairs.append({
                    'hero1_id': h1,
                    'hero1_name': hero_data.get(h1, {}).get('name', f'Hero_{h1}'),
                    'hero2_id': h2,
                    'hero2_name': hero_data.get(h2, {}).get('name', f'Hero_{h2}'),
                    'synergy': round(float(synergy), 3)
                })
    
    # Сортуємо по синергії
    pairs.sort(key=lambda x: x['synergy'], reverse=True)
    return pairs[:3]  # Топ-3 пари


def find_counter_picks_detailed(radiant: List[int], dire: List[int]) -> List[Dict]:
    """Детальний аналіз контр-піків"""
    counters = []
    
    for dire_hero in dire:
        best_counter = None
        best_effectiveness = 0
        
        for radiant_hero in radiant:
            if counter_matrix is not None and 0 <= radiant_hero < 120 and 0 <= dire_hero < 120:
                effectiveness = counter_matrix[radiant_hero, dire_hero]
                if effectiveness > best_effectiveness:
                    best_effectiveness = effectiveness
                    best_counter = radiant_hero
        
        if best_counter:
            counters.append({
                'enemy_hero_id': dire_hero,
                'enemy_hero_name': hero_data.get(dire_hero, {}).get('name', f'Hero_{dire_hero}'),
                'counter_hero_id': best_counter,
                'counter_hero_name': hero_data.get(best_counter, {}).get('name', f'Hero_{best_counter}'),
                'effectiveness': round(float(best_effectiveness), 2)
            })
    
    return sorted(counters, key=lambda x: x['effectiveness'], reverse=True)[:5]


def analyze_team_advantages(team: List[int], enemies: List[int]) -> List[str]:
    """Аналіз переваг команди"""
    advantages = []
    
    team_synergy = calculate_synergy(team)
    enemy_synergy = calculate_synergy(enemies)
    
    if team_synergy > enemy_synergy + 0.1:
        advantages.append("Сильна синергія команди")
    
    # Аналіз контр-піків
    counter_advantage = calculate_counter_advantage(team, enemies)
    if counter_advantage > 0.6:
        advantages.append("Хороші контр-піки проти ворога")
    
    # Аналіз ролей (якщо є дані)
    roles = get_team_roles(team)
    if 'Initiator' in roles and 'Support' in roles:
        advantages.append("Збалансований склад з ініціаторами та підтримкою")
    
    if not advantages:
        advantages.append("Стандартний склад команди")
    
    return advantages


def generate_synergy_recommendations(heroes: List[int], synergy_score: float) -> List[str]:
    """Генерація рекомендацій по синергії"""
    recommendations = []
    
    if synergy_score < 0.4:
        recommendations.append("❗ Низька синергія команди. Розгляньте інших героїв")
    elif synergy_score < 0.6:
        recommendations.append("⚠️ Середня синергія. Можна покращити підбір")
    else:
        recommendations.append("✅ Відмінна синергія команди!")
    
    # Аналіз ролей
    roles = get_team_roles(heroes)
    
    if 'Support' not in roles:
        recommendations.append("Команді бракує підтримки (Support)")
    
    if 'Carry' not in roles:
        recommendations.append("Немає керрі для пізньої гри")
    
    if 'Initiator' not in roles:
        recommendations.append("Додайте ініціатора для початку файтів")
    
    if len(recommendations) == 1:
        recommendations.append("Добре збалансований склад команди")
    
    return recommendations


def generate_team_recommendations(
    radiant: List[int], 
    dire: List[int],
    radiant_synergy: float,
    dire_synergy: float
) -> List[str]:
    """Генерація рекомендацій для матчу"""
    recommendations = []
    
    # Порівняння синергії
    if radiant_synergy > dire_synergy + 0.15:
        recommendations.append("🟢 Radiant має значну перевагу в синергії команди")
    elif dire_synergy > radiant_synergy + 0.15:
        recommendations.append("🔴 Dire має значну перевагу в синергії команди")
    else:
        recommendations.append("⚖️ Команди мають приблизно однакову синергію")
    
    # Аналіз контр-піків
    radiant_advantage = calculate_counter_advantage(radiant, dire)
    dire_advantage = calculate_counter_advantage(dire, radiant)
    
    if radiant_advantage > dire_advantage + 0.1:
        recommendations.append("Radiant має перевагу в контр-піках")
    elif dire_advantage > radiant_advantage + 0.1:
        recommendations.append("Dire має перевагу в контр-піках")
    
    # Рекомендації по грі
    if radiant_synergy > 0.6:
        recommendations.append("💡 Radiant: Грайте агресивно, використовуйте синергію")
    
    if dire_synergy > 0.6:
        recommendations.append("💡 Dire: Фокусуйтесь на груповій грі")
    
    return recommendations


def generate_match_analysis(radiant: List[int], dire: List[int], radiant_prob: float) -> Dict:
    """Генерація детального аналізу матчу"""
    return {
        'radiant_synergy': round(float(calculate_synergy(radiant)), 3),
        'dire_synergy': round(float(calculate_synergy(dire)), 3),
        'radiant_heroes_names': [hero_data.get(h, {}).get('name', f'Hero_{h}') for h in radiant],
        'dire_heroes_names': [hero_data.get(h, {}).get('name', f'Hero_{h}') for h in dire],
        'probability_confidence': 'High' if abs(radiant_prob - 0.5) > 0.15 else 'Medium' if abs(radiant_prob - 0.5) > 0.05 else 'Low',
        'game_phase_advantage': {
            'early_game': 'Radiant' if radiant_prob > 0.5 else 'Dire',
            'late_game': 'Dire' if radiant_prob < 0.55 else 'Radiant'
        }
    }


def get_team_roles(heroes: List[int]) -> List[str]:
    """Отримання ролей команди"""
    roles = set()
    role_mapping = {
        range(1, 20): 'Carry',
        range(20, 40): 'Support',
        range(40, 60): 'Initiator',
        range(60, 80): 'Nuker',
        range(80, 100): 'Disabler',
        range(100, 120): 'Durable'
    }
    
    for hero in heroes:
        for hero_range, role in role_mapping.items():
            if hero in hero_range:
                roles.add(role)
                break
    
    return list(roles)


def get_top_heroes_by_region(region: Optional[str]) -> List[Dict]:
    """Топ героїв по регіону"""
    # Сортуємо героїв по winrate
    heroes_list = []
    for hero_id, stats in hero_stats.items():
        heroes_list.append({
            'hero_id': hero_id,
            'name': hero_data.get(hero_id, {}).get('name', f'Hero_{hero_id}'),
            'winrate': stats['win_rate'],
            'pickrate': stats['total_picks'] / 50000 * 100
        })
    
    heroes_list.sort(key=lambda x: x['winrate'], reverse=True)
    return heroes_list[:10]


def analyze_meta_trends(top_heroes: List[Dict]) -> str:
    """Аналіз мета трендів"""
    # Простий аналіз на основі топ героїв
    hero_ids = [h['hero_id'] for h in top_heroes[:3]]
    
    carry_count = sum(1 for h in hero_ids if h < 30)
    support_count = sum(1 for h in hero_ids if 30 <= h < 60)
    
    if carry_count > support_count:
        return "Carry-focused meta"
    elif support_count > carry_count:
        return "Support-focused meta"
    else:
        return "Balanced meta"


# =============================================
# ГЕНЕРАЦІЯ FALLBACK ДАНИХ
# =============================================

def generate_fallback_hero_data() -> Dict:
    """Генерація даних героїв якщо немає файлу"""
    from .helper import HERO_NAMES
    heroes = {}
    for i in range(1, min(len(HERO_NAMES) + 1, 121)):
        heroes[i] = {
            'id': i,
            'name': HERO_NAMES[i-1] if i <= len(HERO_NAMES) else f"Hero_{i}"
        }
    return heroes


def generate_synergy_matrix(size: int = 120) -> np.ndarray:
    """Генерація матриці синергії"""
    from .helper import generate_synergy_matrix as gen_matrix
    return gen_matrix(size)


def generate_counter_matrix(size: int = 120) -> np.ndarray:
    """Генерація матриці контр-піків"""
    from .helper import generate_counter_matrix as gen_matrix
    return gen_matrix(size)


def load_hero_statistics() -> Dict:
    """Завантаження статистики героїв"""
    from .helper import load_hero_statistics as load_stats
    return load_stats()


def generate_hero_stats(hero_id: int) -> Dict:
    """Генерація статистики героя"""
    from .helper import generate_hero_stats as gen_stats
    return gen_stats(hero_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)