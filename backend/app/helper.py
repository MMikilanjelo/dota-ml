# backend/app/helpers.py
"""
Допоміжні функції для backend
"""
import numpy as np
from typing import Dict, List
import random

# Список імен героїв Dota 2
HERO_NAMES = [
    "Anti-Mage", "Axe", "Bane", "Bloodseeker", "Crystal Maiden",
    "Drow Ranger", "Earthshaker", "Juggernaut", "Mirana", "Morphling",
    "Shadow Fiend", "Phantom Lancer", "Puck", "Pudge", "Razor",
    "Sand King", "Storm Spirit", "Sven", "Tiny", "Vengeful Spirit",
    "Windranger", "Zeus", "Kunkka", "Lina", "Lion",
    "Shadow Shaman", "Slardar", "Tidehunter", "Witch Doctor", "Lich",
    "Riki", "Enigma", "Tinker", "Sniper", "Necrophos",
    "Warlock", "Beastmaster", "Queen of Pain", "Venomancer", "Faceless Void",
    "Wraith King", "Death Prophet", "Phantom Assassin", "Pugna", "Templar Assassin",
    "Viper", "Luna", "Dragon Knight", "Dazzle", "Clockwerk",
    "Leshrac", "Nature's Prophet", "Lifestealer", "Dark Seer", "Clinkz",
    "Omniknight", "Enchantress", "Huskar", "Night Stalker", "Broodmother",
    "Bounty Hunter", "Weaver", "Jakiro", "Batrider", "Chen",
    "Spectre", "Ancient Apparition", "Doom", "Ursa", "Spirit Breaker",
    "Gyrocopter", "Alchemist", "Invoker", "Silencer", "Outworld Destroyer",
    "Lycan", "Brewmaster", "Shadow Demon", "Lone Druid", "Chaos Knight",
    "Meepo", "Treant Protector", "Ogre Magi", "Undying", "Rubick",
    "Disruptor", "Nyx Assassin", "Naga Siren", "Keeper of the Light", "Io",
    "Visage", "Slark", "Medusa", "Troll Warlord", "Centaur Warrunner",
    "Magnus", "Timbersaw", "Bristleback", "Tusk", "Skywrath Mage",
    "Abaddon", "Elder Titan", "Legion Commander", "Techies", "Ember Spirit",
    "Earth Spirit", "Underlord", "Terrorblade", "Phoenix", "Oracle",
    "Winter Wyvern", "Arc Warden", "Monkey King", "Dark Willow", "Pangolier",
    "Grimstroke", "Hoodwink", "Void Spirit", "Snapfire", "Mars",
    "Dawnbreaker", "Marci", "Primal Beast", "Muerta"
]


def generate_fallback_hero_data() -> Dict:
    """Генерація fallback даних героїв"""
    heroes = {}
    for i in range(1, min(len(HERO_NAMES) + 1, 121)):
        heroes[i] = {
            'id': i,
            'name': HERO_NAMES[i-1] if i <= len(HERO_NAMES) else f"Hero_{i}"
        }
    return heroes


def generate_synergy_matrix(size: int = 120) -> np.ndarray:
    """
    Генерація матриці синергії героїв
    Значення від 0 до 1, де 1 = максимальна синергія
    """
    np.random.seed(42)
    
    # Базова випадкова матриця
    matrix = np.random.rand(size, size) * 0.3 + 0.4  # Від 0.4 до 0.7
    
    # Робимо симетричною
    matrix = (matrix + matrix.T) / 2
    
    # Діагональ = 1 (герой сам з собою)
    np.fill_diagonal(matrix, 1.0)
    
    # Додаємо деякі "золоті комбінації"
    golden_pairs = [
        (1, 5), (2, 3), (4, 7), (10, 15), (20, 25),
        (30, 35), (40, 45), (50, 55), (60, 65), (70, 75)
    ]
    
    for h1, h2 in golden_pairs:
        if h1 < size and h2 < size:
            matrix[h1, h2] = 0.9
            matrix[h2, h1] = 0.9
    
    return matrix


def generate_counter_matrix(size: int = 120) -> np.ndarray:
    """
    Генерація матриці контр-піків
    matrix[i,j] = наскільки герой i є контром для героя j
    """
    np.random.seed(43)
    
    # Базова матриця
    matrix = np.random.rand(size, size) * 0.3 + 0.4
    
    # Додаємо сильні контри
    strong_counters = [
        (1, 10), (5, 20), (10, 30), (15, 25), (20, 40),
        (25, 50), (30, 60), (35, 70), (40, 80), (45, 90)
    ]
    
    for counter, target in strong_counters:
        if counter < size and target < size:
            matrix[counter, target] = 0.85
            # Зворотній контр слабший
            matrix[target, counter] = 0.35
    
    return matrix


def load_hero_statistics() -> Dict:
    """
    Завантаження або генерація статистики героїв
    """
    stats = {}
    
    np.random.seed(44)
    
    for hero_id in range(1, 121):
        # Генеруємо реалістичну статистику
        total_picks = int(np.random.randint(5000, 20000))
        win_rate = float(np.random.uniform(45, 55))
        total_wins = int(total_picks * win_rate / 100)
        
        # KDA базується на ролі героя
        if hero_id < 30:  # Керрі
            avg_kda = float(np.random.uniform(3.0, 5.0))
            positions = ["Carry", "Mid"]
            roles = ["Carry", "Escape"]
        elif hero_id < 60:  # Саппорти
            avg_kda = float(np.random.uniform(2.0, 3.5))
            positions = ["Support", "Roaming"]
            roles = ["Support", "Disabler"]
        elif hero_id < 90:  # Ініціатори/танки
            avg_kda = float(np.random.uniform(2.5, 4.0))
            positions = ["Offlane", "Roaming"]
            roles = ["Initiator", "Durable"]
        else:  # Нюкери/спеціалісти
            avg_kda = float(np.random.uniform(2.8, 4.5))
            positions = ["Mid", "Support"]
            roles = ["Nuker", "Support"]
        
        stats[hero_id] = {
            'total_picks': total_picks,
            'total_wins': total_wins,
            'win_rate': win_rate,
            'avg_kda': avg_kda,
            'positions': positions,
            'roles': roles
        }
    
    return stats


def generate_hero_stats(hero_id: int) -> Dict:
    """Генерація статистики для одного героя"""
    np.random.seed(hero_id)
    
    total_picks = int(np.random.randint(8000, 18000))
    win_rate = float(np.random.uniform(48, 54))
    
    return {
        'total_picks': total_picks,
        'total_wins': int(total_picks * win_rate / 100),
        'win_rate': win_rate,
        'avg_kda': float(np.random.uniform(2.5, 4.0)),
        'positions': random.sample(["Carry", "Mid", "Offlane", "Support", "Roaming"], 2),
        'roles': random.sample(["Carry", "Support", "Nuker", "Disabler", "Initiator", "Durable", "Escape"], 2)
    }


def calculate_team_composition_score(heroes: List[int]) -> Dict:
    """
    Оцінка складу команди по різним параметрам
    """
    scores = {
        'damage': 0,
        'control': 0,
        'tankiness': 0,
        'mobility': 0,
        'push': 0
    }
    
    for hero in heroes:
        # Спрощена логіка на основі ID героя
        if hero < 30:  # Керрі
            scores['damage'] += 0.8
            scores['mobility'] += 0.6
        elif hero < 60:  # Саппорти
            scores['control'] += 0.7
            scores['damage'] += 0.3
        elif hero < 90:  # Танки
            scores['tankiness'] += 0.9
            scores['control'] += 0.5
        else:  # Нюкери
            scores['damage'] += 0.7
            scores['push'] += 0.6
    
    # Нормалізація (максимум 5 для кожної категорії)
    for key in scores:
        scores[key] = min(scores[key], 5.0)
        scores[key] = round(scores[key], 2)
    
    return scores


def get_hero_role_distribution(heroes: List[int]) -> Dict[str, int]:
    """Розподіл ролей у команді"""
    roles = {
        'Carry': 0,
        'Support': 0,
        'Initiator': 0,
        'Nuker': 0,
        'Disabler': 0
    }
    
    for hero in heroes:
        if hero < 30:
            roles['Carry'] += 1
        elif hero < 60:
            roles['Support'] += 1
        elif hero < 90:
            roles['Initiator'] += 1
        else:
            roles['Nuker'] += 1
    
    return roles


def calculate_game_phase_advantage(radiant: List[int], dire: List[int]) -> Dict:
    """
    Розрахунок переваги на різних фазах гри
    """
    def team_early_game_score(heroes):
        # Герої 20-60 сильніші на ранній стадії
        return sum(1 for h in heroes if 20 <= h < 60) * 0.3
    
    def team_late_game_score(heroes):
        # Герої 1-30 сильніші в пізній грі
        return sum(1 for h in heroes if h < 30) * 0.35
    
    radiant_early = team_early_game_score(radiant)
    dire_early = team_early_game_score(dire)
    
    radiant_late = team_late_game_score(radiant)
    dire_late = team_late_game_score(dire)
    
    return {
        'early_game': {
            'radiant': round(radiant_early, 2),
            'dire': round(dire_early, 2),
            'advantage': 'Radiant' if radiant_early > dire_early else 'Dire' if dire_early > radiant_early else 'Equal'
        },
        'late_game': {
            'radiant': round(radiant_late, 2),
            'dire': round(dire_late, 2),
            'advantage': 'Radiant' if radiant_late > dire_late else 'Dire' if dire_late > radiant_late else 'Equal'
        }
    }


def get_recommended_items_for_hero(hero_id: int) -> List[str]:
    """Рекомендовані предмети для героя"""
    # Спрощена логіка
    if hero_id < 30:  # Керрі
        return ["Battle Fury", "Black King Bar", "Butterfly", "Satanic"]
    elif hero_id < 60:  # Саппорти
        return ["Observer Ward", "Glimmer Cape", "Force Staff", "Aether Lens"]
    elif hero_id < 90:  # Танки
        return ["Blink Dagger", "Heart of Tarrasque", "Blade Mail", "Crimson Guard"]
    else:  # Нюкери
        return ["Aghanim's Scepter", "Refresher Orb", "Scythe of Vyse", "Octarine Core"]


def analyze_draft_timing(radiant: List[int], dire: List[int]) -> Dict:
    """Аналіз timing attack'ів команд"""
    def get_power_spike_timing(heroes):
        avg_id = sum(heroes) / len(heroes)
        if avg_id < 40:
            return "30-40 min"
        elif avg_id < 70:
            return "15-25 min"
        else:
            return "10-20 min"
    
    return {
        'radiant_power_spike': get_power_spike_timing(radiant),
        'dire_power_spike': get_power_spike_timing(dire),
        'recommendation': "Push early" if sum(radiant) < sum(dire) else "Farm and scale"
    }