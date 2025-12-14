# http://localhost:8501/
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

API_URL = "http://backend:8000"

st.set_page_config(
    page_title="Dota 2 Match Predictor",
    page_icon="🎮",
    layout="wide"
)

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);}
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
    }
    .stat-card {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .winner-text {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        text-shadow: 0 0 10px rgba(0,0,0,0.5);
    }
    .recommendation-box {
        padding: 1rem;
        border-left: 5px solid #4ECDC4;
        background-color: rgba(78, 205, 196, 0.1);
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def fetch_api(endpoint, method="GET", payload=None):
    try:
        url = f"{API_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, json=payload, timeout=10)
        
        # Якщо статус OK (200-299)
        if response.ok:
            return response.json(), None
            
        # Якщо помилка API (4xx, 5xx), пробуємо дістати деталі
        try:
            error_data = response.json()
            return None, error_data.get('detail', f"Error {response.status_code}")
        except:
            return None, f"HTTP Error {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return None, "❌ Неможливо підключитися до Backend сервера. Перевірте Docker."
    except Exception as e:
        return None, f"System Error: {str(e)}"


def create_radar_chart(metrics):


    categories = ['Farming (GPM)', 'Fighting (KDA)', 'Support', 'Pushing', 'Versatility']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[
            metrics.get('farming', 0),
            metrics.get('fighting', 0),
            metrics.get('support', 0),
            metrics.get('pushing', 0),
            metrics.get('versatility', 0)
        ],
        theta=categories,
        fill='toself',
        name='Player Stats',
        line_color='#4ECDC4'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title="Skill Graph",
        margin=dict(t=30, b=30)
    )
    return fig


def main():
    st.markdown('<h1 class="main-header">🎮 DOTA 2 ML ANALYTICS</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota2_social.jpg")
        st.title("⚙️ Меню")
        
        page = st.radio(
            "Навігація:",
            ["🔮 Прогноз Матчу", "👤 Аналіз Гравця", "📊 Глобальна Статистика", "🦸 Герої"]
        )
        
        st.markdown("---")
        try:
            requests.get(f"{API_URL}/", timeout=1)
            st.success("🟢 Сервер онлайн")
        except:
            st.error("🔴 Сервер офлайн")

    if page == "🔮 Прогноз Матчу":
        st.header("⚔️ Передбачення переможця та аналіз драфту")
        
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Radiant Team (Green)")
            radiant_heroes = [st.number_input(f"R-Hero {i+1}", 1, 130, i+1, key=f"r{i}") for i in range(5)]
            r_rating = st.number_input("Сер. рейтинг Radiant (опціонально)", 0, 10000, 0)
        
        with col2:
            st.subheader("Dire Team (Red)")
            dire_heroes = [st.number_input(f"D-Hero {i+1}", 1, 130, 10+i, key=f"d{i}") for i in range(5)]
            d_rating = st.number_input("Сер. рейтинг Dire (опціонально)", 0, 10000, 0)

        if st.button("🔮 Аналізувати Драфт", type="primary", use_container_width=True):

            all_heroes = radiant_heroes + dire_heroes
            if len(set(all_heroes)) != 10:
                st.error("❌ Герої не повинні повторюватись!")
            else:
                with st.spinner("🤖 ML модель проводить глибокий аналіз..."):
                    payload = {
                        "radiant_heroes": radiant_heroes, 
                        "dire_heroes": dire_heroes,
                        "radiant_avg_rating": r_rating,
                        "dire_avg_rating": d_rating
                    }
                    
                    # 1. Прогноз переможця
                    prediction, err_pred = fetch_api("/predict/match-winner", "POST", payload)
                    # 2. Детальний аналіз складу (Нова фіча)
                    composition, err_comp = fetch_api("/analysis/team-composition", "POST", payload)
                    
                    if err_pred:
                        st.error(f"Помилка прогнозу: {err_pred}")
                    
                    if prediction:
                        # Відображення переможця
                        winner = prediction['predicted_winner']
                        prob = prediction['radiant_win_probability']
                        color = "#32CD32" if winner == "Radiant" else "#DC143C"
                        
                        st.markdown(f'<div class="winner-text" style="color:{color}">🏆 {winner.upper()} WINS</div>', unsafe_allow_html=True)
                        st.progress(prob)
                        st.caption(f"Впевненість моделі: {prediction['confidence']*100:.1f}%")

                    if composition:
                        st.markdown("---")
                        st.subheader("🔍 Глибокий аналіз складу")
                        
                        # Синергія
                        c1, c2 = st.columns(2)
                        with c1:
                            syn_r = composition['radiant_synergy']
                            st.metric("Синергія Radiant", f"{syn_r:.2f}")
                            st.progress(min(syn_r, 1.0))
                        with c2:
                            syn_d = composition['dire_synergy']
                            st.metric("Синергія Dire", f"{syn_d:.2f}")
                            st.progress(min(syn_d, 1.0))
                        
                        # Контр-піки
                        with st.expander("⚔️ Контр-піки (Хто кого контрить?)", expanded=True):
                            if composition.get('counter_picks'):
                                for cp in composition['counter_picks']:
                                    st.markdown(f"**{cp['counter_hero_name']}** контрить **{cp['enemy_hero_name']}** (Ефект: {cp['effectiveness']})")
                            else:
                                st.write("Значних контр-піків не виявлено.")
                        
                        # Поради
                        st.markdown("### 💡 Рекомендації")
                        if composition.get('recommendations'):
                            for rec in composition['recommendations']:
                                st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)
                                st.write("")

    elif page == "👤 Аналіз Гравця":
        st.header("👤 Визначення Playstyle Гравця")
        st.info("Введіть **реальний** Account ID (наприклад, 4, 88470, 111). Анонімні гравці (ID 0) не підтримуються.")
        
        account_id = st.text_input("Account ID", value="4") 
        
        if st.button("🔎 Аналізувати стиль"):
            with st.spinner("Завантаження історії матчів..."):
                player_data, error = fetch_api(f"/player/{account_id}/playstyle")
                
                if error:
                    # Обробка специфічних помилок від бекенду
                    if "not found" in error.lower():
                        st.warning(f"⚠ Гравця з ID {account_id} не знайдено в датасеті або у нього менше 5 ігор.")
                        st.caption("Спробуйте інший ID (наприклад, з діапазону 1-100 для цього датасету).")
                    elif "anonymous" in error.lower():
                        st.error("🚫 Цей профіль прихований (Anonymous). Аналіз неможливий.")
                    else:
                        st.error(f"❌ Помилка: {error}")
                
                elif player_data:
                    # Перевірка, чи є метрики
                    if 'metrics' not in player_data:
                        st.warning(f"⚠ {player_data.get('error', 'Недостатньо даних для аналізу')}")
                    else:
                        # Якщо метрики є - будуємо графіки
                        col_info, col_chart = st.columns([1, 2])
                        
                        with col_info:
                            st.markdown(f"""
                            <div class="stat-card">
                                <h3>{player_data.get('player_name')}</h3>
                                <h1 style="color: #4ECDC4;">{player_data.get('playstyle_label')}</h1>
                                <p>Базується на {player_data.get('match_count', '?')} матчах</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                            metrics = player_data['metrics']
                            st.write(f"💰 **GPM:** {metrics['gpm']}")
                            st.write(f"✨ **XPM:** {metrics['xpm']}")
                            st.write(f"⚔️ **KDA:** {metrics['kda']}")
                    
                        with col_chart:
                            fig = create_radar_chart(player_data['radar_stats'])
                            st.plotly_chart(fig, use_container_width=True)
                            
                        st.success(f"🤖 **ML вердикт:** {player_data.get('recommendation')}")
                    
                    
    elif page == "📊 Глобальна Статистика":
        st.header("📊 Аналітика Датасету")
        
        stats, err = fetch_api("/stats/general")
        
        if stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Всього матчів", f"{stats.get('total_matches', 0):,}")
            c2.metric("Radiant Winrate", f"{stats.get('radiant_winrate', 0)*100:.1f}%")
            c3.metric("Сер. тривалість", f"{stats.get('avg_duration', 0)/60:.0f} хв")
            c4.metric("Точність моделі", f"{stats.get('model_accuracy', 0)*100:.1f}%")
            
            st.markdown("---")
            st.subheader("📈 Динаміка Winrate")
            
            trends, err_tr = fetch_api("/stats/trends")
            if trends:
                df_trends = pd.DataFrame(trends)
                if not df_trends.empty:
                    fig = px.line(df_trends, x='date', y='winrate', title='Radiant Winrate Trend', template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Недостатньо даних для графіку трендів.")
        else:
            st.warning("Не вдалося завантажити статистику.")

    elif page == "🦸 Герої":
        st.header("📚 Енциклопедія Героїв")
        h_id = st.number_input("ID Героя", 1, 130, 1)
        
        if st.button("Показати статистику"):
            h_stats, err = fetch_api(f"/stats/hero/{h_id}")
            
            if h_stats:
                st.subheader(h_stats.get('hero_name', f'Hero {h_id}'))
                c1, c2, c3 = st.columns(3)
                c1.metric("Win Rate", f"{h_stats.get('win_rate', 0):.1f}%")
                c2.metric("Total Picks", h_stats.get('total_picks', 0))
                c3.metric("Avg KDA", f"{h_stats.get('avg_kda', 0):.2f}")

                if h_stats.get('popular_positions'):
                    st.write("**Популярні ролі:**", ", ".join(h_stats['popular_positions']))
            else:
                st.error(f"Героя не знайдено або помилка: {err}")

if __name__ == "__main__":
    main()