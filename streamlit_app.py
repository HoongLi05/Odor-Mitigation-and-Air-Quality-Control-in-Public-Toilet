import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from stable_baselines3 import PPO, A2C, SAC, DQN
import warnings
warnings.filterwarnings('ignore')

# =============================
# 頁面配置
# =============================
st.set_page_config(
    page_title="公共廁所RL模型視覺化",
    layout="wide",
    page_icon="🚽",
    initial_sidebar_state="expanded"
)

# 自定義CSS
st.markdown("""
<style>
    .model-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid;
    }
    .ppo-card { border-left-color: #FF6B6B; background-color: #FF6B6B10; }
    .a2c-card { border-left-color: #4ECDC4; background-color: #4ECDC410; }
    .sac-card { border-left-color: #45B7D1; background-color: #45B7D110; }
    .dqn-card { border-left-color: #96CEB4; background-color: #96CEB410; }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .good { background-color: #28a745; }
    .warning { background-color: #ffc107; }
    .bad { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# =============================
# 假設的模型載入函數
# =============================
# 注意：這裡需要根據您實際的模型儲存方式進行調整

def load_pretrained_model(model_name, model_path):
    """
    載入預訓練的Stable-Baselines3模型
    """
    try:
        if os.path.exists(model_path):
            if model_path.endswith(".zip"):
                # 根據模型類型載入
                if model_name == "PPO":
                    model_data = PPO.load(model_path)
                elif model_name == "A2C":
                    model_data = A2C.load(model_path)
                elif model_name == "SAC":
                    model_data = SAC.load(model_path)
                elif model_name == "DQN":
                    model_data = DQN.load(model_path)
                else:
                    model_data = None

                st.sidebar.success(f"✅ {model_name} 模型載入成功")
                
                # 返回字典，用來顯示統計信息
                return {
                    'name': model_name,
                    'data': {
                        'model_obj': model_data,
                        'total_timesteps': getattr(model_data, 'num_timesteps', 'N/A'),
                        'avg_reward': 'N/A'
                    },
                    'color': get_model_color(model_name),
                    'loaded': True
                }
            else:
                # 如果是 pickle 文件
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                st.sidebar.success(f"✅ {model_name} 模型載入成功")
                return {
                    'name': model_name,
                    'data': model_data,
                    'color': get_model_color(model_name),
                    'loaded': True
                }
        else:
            st.sidebar.warning(f"⚠️ 未找到模型文件，使用模擬數據")
            return create_mock_model(model_name)
        
    except Exception as e:
        st.sidebar.error(f"❌ 載入{model_name}模型失敗: {e}")
        return create_mock_model(model_name)

def get_model_color(model_name):
    """獲取模型對應的顏色"""
    colors = {
        'PPO': '#FF6B6B',
        'A2C': '#4ECDC4', 
        'SAC': '#45B7D1',
        'DQN': '#96CEB4'
    }
    return colors.get(model_name, '#6c757d')

def create_mock_model(model_name):
    """創建模擬模型數據（用於演示）"""
    return {
        'name': model_name,
        'data': {
            'episodes_trained': np.random.randint(1000, 5000),
            'total_timesteps': np.random.randint(10000, 50000),
            'avg_reward': np.random.uniform(-50, 100),
            'best_reward': np.random.uniform(0, 150),
            'training_time': timedelta(minutes=np.random.randint(30, 180))
        },
        'color': get_model_color(model_name),
        'loaded': False
    }

# =============================
# 模擬推論函數
# =============================

def simulate_model_inference(model_info, steps=100):
    """
    使用載入的模型進行模擬推論
    返回模擬數據
    """
    # 創建模擬數據 - 這裡應該替換為實際的模型推論
    hours = steps // 60 if steps > 60 else 1
    
    # 創建時間序列
    timestamps = pd.date_range(
        start=datetime.now().replace(hour=6, minute=0, second=0),
        periods=steps,
        freq='1min'
    )
    
    # 污染物濃度模擬
    base_nh3 = np.random.uniform(0.5, 2.0)
    base_co2 = np.random.uniform(400, 600)
    base_temp = np.random.uniform(24, 28)
    base_humidity = np.random.uniform(55, 70)
    
    # 根據模型類型調整趨勢
    if model_info['name'] == 'PPO':
        # PPO: 較為平穩
        nh3_trend = np.sin(np.linspace(0, 4*np.pi, steps)) * 0.5 + base_nh3
        co2_trend = np.sin(np.linspace(0, 2*np.pi, steps)) * 100 + base_co2
    elif model_info['name'] == 'A2C':
        # A2C: 波動較大
        nh3_trend = np.sin(np.linspace(0, 8*np.pi, steps)) * 1.0 + base_nh3
        co2_trend = np.sin(np.linspace(0, 4*np.pi, steps)) * 150 + base_co2
    elif model_info['name'] == 'SAC':
        # SAC: 探索性強
        nh3_trend = base_nh3 + np.cumsum(np.random.randn(steps)) * 0.1
        co2_trend = base_co2 + np.cumsum(np.random.randn(steps)) * 10
    else:  # DQN
        # DQN: 較為保守
        nh3_trend = np.ones(steps) * base_nh3 + np.random.randn(steps) * 0.3
        co2_trend = np.ones(steps) * base_co2 + np.random.randn(steps) * 50
    
    # 創建數據框
    df = pd.DataFrame({
        'timestamp': timestamps,
        'hour': timestamps.hour,
        'minute': timestamps.minute,
        'time_minutes': np.arange(steps),
        'nh3_ppm': np.clip(nh3_trend, 0, 30),
        'h2s_ppm': np.clip(np.random.exponential(0.05, steps), 0, 2),
        'co2_ppm': np.clip(co2_trend, 300, 2000),
        'temperature_c': base_temp + np.sin(np.linspace(0, 2*np.pi, steps)) * 2,
        'humidity_percent': np.clip(base_humidity + np.sin(np.linspace(0, np.pi, steps)) * 10, 40, 85),
        'user_count': (np.sin(np.linspace(0, 4*np.pi, steps)) * 2 + 3).clip(0, 10).astype(int),
        'energy_consumption': np.random.uniform(0.5, 3.0, steps),
        'reward': np.random.normal(10, 3, steps),
        'action_taken': np.random.choice(ACTION_SPACE, size=steps),
        'model': model_info['name']
    })
    
    # 添加設備狀態（基於動作）
    df['exhaust_fan'] = df['action_taken'].apply(lambda x: 'exhaust' in x or x == 'all_on')
    df['ceiling_fan'] = df['action_taken'].apply(lambda x: 'ceiling' in x or x == 'all_on')
    df['dehumidifier'] = df['action_taken'].apply(lambda x: 'dehum' in x or x == 'all_on')
    
    return df

# =============================
# 視覺化函數
# =============================

def create_pollutant_chart(df, current_step=None):
    """創建污染物濃度圖表（支持動態顯示）"""
    fig = go.Figure()
    
    # 確定要顯示的數據範圍
    if current_step is not None and current_step < len(df):
        display_df = df.iloc[:current_step+1]
    else:
        display_df = df
    
    # NH3
    fig.add_trace(go.Scatter(
        x=display_df['time_minutes'],
        y=display_df['nh3_ppm'],
        mode='lines+markers',
        name='NH3 (ppm)',
        line=dict(color='#FF6B6B', width=2),
        marker=dict(size=4)
    ))
    
    # CO2 (右側Y軸)
    fig.add_trace(go.Scatter(
        x=display_df['time_minutes'],
        y=display_df['co2_ppm'],
        mode='lines+markers',
        name='CO2 (ppm)',
        line=dict(color='#4ECDC4', width=2),
        marker=dict(size=4),
        yaxis='y2'
    ))
    
    # 如果正在播放，添加當前位置標記
    if current_step is not None and current_step < len(df):
        current_time = df.iloc[current_step]['time_minutes']
        current_nh3 = df.iloc[current_step]['nh3_ppm']
        current_co2 = df.iloc[current_step]['co2_ppm']
        
        # 添加垂直線標記當前位置
        fig.add_vline(x=current_time, line_dash="dash", line_color="gray", opacity=0.5)
        
        # 添加當前點標記
        fig.add_trace(go.Scatter(
            x=[current_time],
            y=[current_nh3],
            mode='markers',
            name='當前NH3',
            marker=dict(size=12, color='#FF0000'),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=[current_time],
            y=[current_co2],
            mode='markers',
            name='當前CO2',
            marker=dict(size=12, color='#00FF00'),
            showlegend=False,
            yaxis='y2'
        ))
    
    # 安全閾值線
    fig.add_hline(y=10, line_dash="dash", line_color="red", 
                  annotation_text="NH3安全限值", annotation_position="top right")
    fig.add_hline(y=1500, line_dash="dash", line_color="orange", 
                  annotation_text="CO2舒適限值", yref='y2')
    
    fig.update_layout(
        title=f'污染物濃度變化 ({len(display_df)}/{len(df)} 分鐘)',
        xaxis_title='時間 (分鐘)',
        yaxis_title='NH3 (ppm)',
        yaxis2=dict(
            title='CO2 (ppm)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_comfort_chart(df, current_step=None):
    """創建舒適度圖表（支持動態顯示）"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 確定要顯示的數據範圍
    if current_step is not None and current_step < len(df):
        display_df = df.iloc[:current_step+1]
    else:
        display_df = df
    
    # 溫度 (主Y軸)
    fig.add_trace(go.Scatter(
        x=display_df['time_minutes'],
        y=display_df['temperature_c'],
        mode='lines+markers',
        name='溫度 (°C)',
        line=dict(color='#FF9F1C', width=3),
        marker=dict(size=4)
    ), secondary_y=False)
    
    # 濕度 (次Y軸)
    fig.add_trace(go.Scatter(
        x=display_df['time_minutes'],
        y=display_df['humidity_percent'],
        mode='lines+markers',
        name='濕度 (%)',
        line=dict(color='#2EC4B6', width=3),
        marker=dict(size=4)
    ), secondary_y=True)
    
    # 如果正在播放，添加當前位置標記
    if current_step is not None and current_step < len(df):
        current_time = df.iloc[current_step]['time_minutes']
        current_temp = df.iloc[current_step]['temperature_c']
        current_hum = df.iloc[current_step]['humidity_percent']
        
        # 添加當前點標記
        fig.add_trace(go.Scatter(
            x=[current_time],
            y=[current_temp],
            mode='markers',
            name='當前溫度',
            marker=dict(size=12, color='#FF6B00'),
            showlegend=False
        ), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=[current_time],
            y=[current_hum],
            mode='markers',
            name='當前濕度',
            marker=dict(size=12, color='#0088FF'),
            showlegend=False
        ), secondary_y=True)
    
    # 舒適區間
    fig.add_hrect(y0=24, y1=28, line_width=0, fillcolor="green", opacity=0.1,
                  annotation_text="舒適溫度區間", annotation_position="top left",
                  secondary_y=False)
    
    fig.add_hrect(y0=50, y1=70, line_width=0, fillcolor="blue", opacity=0.1,
                  annotation_text="舒適濕度區間",
                  secondary_y=True)
    
    fig.update_yaxes(title_text="溫度 (°C)", secondary_y=False)
    fig.update_yaxes(title_text="濕度 (%)", secondary_y=True)
    
    fig.update_layout(
        title=f'溫濕度舒適度 ({len(display_df)}/{len(df)} 分鐘)',
        xaxis_title='時間 (分鐘)',
        height=350,
        template='plotly_white'
    )
    
    return fig

def create_equipment_chart(df):
    """創建設備使用圖表"""
    # 計算各設備使用時間
    equipment_usage = pd.DataFrame({
        '設備': ['排氣扇', '天花板風扇', '除濕機'],
        '使用時間 (分鐘)': [
            df['exhaust_fan'].sum(),
            df['ceiling_fan'].sum(),
            df['dehumidifier'].sum()
        ],
        '顏色': ['#FF6B6B', '#4ECDC4', '#45B7D1']
    })
    
    fig = px.bar(
        equipment_usage,
        x='設備',
        y='使用時間 (分鐘)',
        color='設備',
        color_discrete_map={
            '排氣扇': '#FF6B6B',
            '天花板風扇': '#4ECDC4',
            '除濕機': '#45B7D1'
        },
        title='設備使用時間統計'
    )
    
    fig.update_layout(
        height=300,
        template='plotly_white'
    )
    
    return fig

def create_reward_chart(df):
    """創建獎勵圖表"""
    fig = go.Figure()
    
    # 即時獎勵
    fig.add_trace(go.Scatter(
        x=df['time_minutes'],
        y=df['reward'],
        mode='lines',
        name='即時獎勵',
        line=dict(color='#7209B7', width=2)
    ))
    
    # 累積獎勵
    cumulative_reward = df['reward'].cumsum()
    fig.add_trace(go.Scatter(
        x=df['time_minutes'],
        y=cumulative_reward,
        mode='lines',
        name='累積獎勵',
        line=dict(color='#F72585', width=3)
    ))
    
    fig.update_layout(
        title='獎勵曲線',
        xaxis_title='時間 (分鐘)',
        yaxis_title='獎勵值',
        height=350,
        template='plotly_white'
    )
    
    return fig

def create_comparison_chart(comparison_data):
    """創建模型比較圖表"""
    fig = go.Figure()
    
    # 總獎勵比較
    fig.add_trace(go.Bar(
        x=comparison_data['Model'],
        y=comparison_data['Total Reward'],
        name='總獎勵',
        marker_color=[get_model_color(m) for m in comparison_data['Model']]
    ))
    
    fig.update_layout(
        title='模型總獎勵比較',
        xaxis_title='模型',
        yaxis_title='總獎勵',
        height=400,
        template='plotly_white'
    )
    
    return fig

# =============================
# 動作空間（從您的環境複製）
# =============================
ACTION_SPACE = [
    "all_off",
    "exhaust_only",
    "ceiling_only",
    "dehum_only",
    "exhaust_ceiling",
    "exhaust_dehum",
    "ceiling_dehum",
    "all_on"
]

# =============================
# 主應用程式
# =============================

def main():
    # 初始化 session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'playing' not in st.session_state:
        st.session_state.playing = False
    if 'animation_speed' not in st.session_state:
        st.session_state.animation_speed = 0.5  # 秒為單位
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = time.time()
    
    # 標題
    st.title("🚽 公共廁所RL模型視覺化儀表板")
    st.markdown("載入已訓練的PPO、A2C、SAC、DQN模型，並視覺化其表現")
    
    # 側邊欄
    with st.sidebar:
        st.header("📂 模型載入")
        
        # 模型路徑設定
        model_dir = st.text_input(
            "模型目錄路徑",
            value="./trained_models",
            help="包含訓練好的模型文件的目錄"
        )
        
        # 模型選擇
        selected_models = st.multiselect(
            "選擇要載入的模型",
            options=["PPO", "A2C", "SAC", "DQN"],
            default=["PPO", "DQN"]
        )
        
        # 模擬參數
        st.header("⚙️ 模擬設定")
        simulation_steps = st.slider(
            "模擬步數 (分鐘)",
            min_value=60,
            max_value=480,
            value=120,
            step=30
        )
        
        # 載入按鈕
        if st.button("🔍 載入並模擬模型", type="primary"):
            with st.spinner("載入模型中..."):
                
                # 載入選定的模型
                loaded_models = {}
                for model_name in selected_models:
                    # 判斷zip檔案
                    model_path_zip = os.path.join(model_dir, f"{model_name.lower()}_model.zip")
                    model_path_pkl = os.path.join(model_dir, f"{model_name.lower()}_model.pkl")
    
                    if os.path.exists(model_path_zip):
                        loaded_models[model_name] = load_pretrained_model(model_name, model_path_zip)
                    elif os.path.exists(model_path_pkl):
                        loaded_models[model_name] = load_pretrained_model(model_name, model_path_pkl)
                    else:
                        st.sidebar.warning(f"⚠️ {model_name} 模型文件不存在，將使用模擬數據")
                        loaded_models[model_name] = create_mock_model(model_name)
                
                st.session_state.loaded_models = loaded_models
                st.session_state.simulation_steps = simulation_steps
                
                # 重置播放狀態
                st.session_state.current_step = 0
                st.session_state.playing = True
                st.session_state.last_update_time = time.time()
                
                # 執行模擬
                simulation_results = {}
                for model_name, model_info in loaded_models.items():
                    df = simulate_model_inference(model_info, simulation_steps)
                    simulation_results[model_name] = df
                
                st.session_state.simulation_results = simulation_results
                
                # 計算比較數據
                comparison_data = []
                for model_name, df in simulation_results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Total Reward': df['reward'].sum(),
                        'Avg NH3': df['nh3_ppm'].mean(),
                        'Avg CO2': df['co2_ppm'].mean(),
                        'Energy Consumption': df['energy_consumption'].sum(),
                        'Safety Violations': len(df[df['nh3_ppm'] > 10]),
                        'Comfort Score': 100 - (abs(df['temperature_c'] - 26).mean() * 2)
                    })
                
                st.session_state.comparison_data = pd.DataFrame(comparison_data)
                
            st.success(f"✅ 成功載入 {len(selected_models)} 個模型")
            st.rerun()  # 重新運行以顯示數據
    
    # 檢查是否有載入的模型
    if 'loaded_models' not in st.session_state:
        st.info("👈 請在側邊欄選擇模型並點擊『載入並模擬模型』")
        return
    
    # 顯示載入的模型信息
    st.header("📊 已載入模型")
    cols = st.columns(len(st.session_state.loaded_models))
    
    for idx, (model_name, model_info) in enumerate(st.session_state.loaded_models.items()):
        with cols[idx]:
            card_class = f"{model_name.lower()}-card"
            total_timesteps = model_info['data'].get('total_timesteps', 'N/A')
            avg_reward = model_info['data'].get('avg_reward', 'N/A')
            avg_reward_str = f"{avg_reward:.2f}" if isinstance(avg_reward, (int, float)) else avg_reward
            st.markdown(f"""
            <div class="model-card {card_class}">
                <h4>{model_name}</h4>
                <p><strong>狀態:</strong> {"✅ 已載入" if model_info['loaded'] else "⚠️ 模擬數據"}</p>
                <p><strong>訓練步數:</strong> {total_timesteps}</p>
                <p><strong>平均獎勵:</strong> {avg_reward_str}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 模型選擇切換
    available_models = list(st.session_state.loaded_models.keys())
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = available_models[0]
    
    selected_model = st.selectbox(
        "選擇要詳細查看的模型",
        options=available_models,
        index=available_models.index(st.session_state.selected_model)
    )
    st.session_state.selected_model = selected_model
    df = st.session_state.simulation_results[selected_model]
    
    # ============================================
    # 關鍵修改：使用 st.form 來確保按鈕立即響應
    # ============================================
    st.subheader("🎬 動畫控制")
    
    # 創建一個 form 來包裹控制按鈕
    with st.form("animation_control_form"):
        control_col1, control_col2, control_col3, control_col4 = st.columns(4)
        
        with control_col1:
            play_button = st.form_submit_button("▶️ 播放", use_container_width=True)
        
        with control_col2:
            pause_button = st.form_submit_button("⏸️ 暫停", use_container_width=True)
        
        with control_col3:
            reset_button = st.form_submit_button("⏹️ 重置", use_container_width=True)
        
        with control_col4:
            # 速度控制放在 form 外面，因為它不需要立即響應
            pass
    
    # 處理按鈕點擊
    if play_button:
        st.session_state.playing = True
        st.session_state.last_update_time = time.time()
        st.rerun()
    
    if pause_button:
        st.session_state.playing = False
        st.rerun()
    
    if reset_button:
        st.session_state.current_step = 0
        st.session_state.playing = False
        st.rerun()
    
    # 速度控制（放在 form 外面）
    control_col4_1, control_col4_2 = st.columns([3, 1])
    with control_col4_1:
        st.session_state.animation_speed = st.select_slider(
            "播放速度 (秒/步)",
            options=[0.1, 0.3, 0.5, 1.0, 2.0],
            value=st.session_state.animation_speed,
            key="speed_slider"
        )
    with control_col4_2:
        st.metric("速度", f"{st.session_state.animation_speed}s")
    
    # 進度顯示
    progress_col1, progress_col2, progress_col3 = st.columns([2, 2, 1])
    
    with progress_col1:
        st.metric("當前步數", f"{st.session_state.current_step + 1}")
    
    with progress_col2:
        st.metric("總步數", f"{len(df)}")
    
    with progress_col3:
        progress_percent = (st.session_state.current_step + 1) / len(df) * 100
        st.metric("完成度", f"{progress_percent:.1f}%")
    
    # ============================================
    # 自動播放邏輯（簡化版本）
    # ============================================
    if st.session_state.playing:
        current_time = time.time()
        time_elapsed = current_time - st.session_state.last_update_time
    
        if time_elapsed >= st.session_state.animation_speed:
            # 前進一步
            if st.session_state.current_step < len(df) - 1:
                st.session_state.current_step += 1
                st.session_state.last_update_time = current_time
                # 使用 st.rerun() 而不是 st.experimental_rerun()
                st.rerun()
            else:
                # 到達最後一步，停止播放
                st.session_state.playing = False
                st.toast("🎬 模擬播放完成！", icon="✅")
    
    # ============================================
    # 顯示數據和圖表（保持不變）
    # ============================================
    
    # 顯示關鍵指標（使用到當前步的數據）
    st.header(f"📈 {selected_model} 模型表現")
    
    # 獲取當前幀數據
    current_frame = df.iloc[st.session_state.current_step]
    partial_df = df.iloc[:st.session_state.current_step+1]
    
    # 計算到當前步為止的統計數據
    total_reward = partial_df['reward'].sum()
    avg_nh3 = partial_df['nh3_ppm'].mean()
    avg_co2 = partial_df['co2_ppm'].mean()
    total_energy = partial_df['energy_consumption'].sum()
    safety_violations = len(partial_df[partial_df['nh3_ppm'] > 10])
    comfort_score = 100 - (abs(partial_df['temperature_c'] - 26).mean() * 2 + abs(partial_df['humidity_percent'] - 60).mean())
    
    # 顯示指標
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏆 累積獎勵", f"{total_reward:.2f}")
    with col2:
        nh3_status = "🟢" if current_frame['nh3_ppm'] < 5 else "🟡" if current_frame['nh3_ppm'] < 15 else "🔴"
        st.metric(f"{nh3_status} 當前NH3", f"{current_frame['nh3_ppm']:.2f} ppm")
    with col3:
        co2_status = "🟢" if current_frame['co2_ppm'] < 800 else "🟡" if current_frame['co2_ppm'] < 1500 else "🔴"
        st.metric(f"{co2_status} 當前CO2", f"{current_frame['co2_ppm']:.0f} ppm")
    with col4:
        st.metric("⚡ 累積能耗", f"{total_energy:.1f} kWh")
    
    # 顯示當前狀態信息
    st.markdown(f"""
    **當前時間:** {current_frame['time_minutes']} 分鐘 | 
    **當前動作:** {current_frame['action_taken']} | 
    **使用人數:** {current_frame['user_count']} | 
    **舒適度評分:** {max(0, comfort_score):.1f}%
    """)
    
    # 圖表顯示（傳入 current_step）
    tab1, tab2, tab3, tab4 = st.tabs(["污染物濃度", "舒適度", "設備使用", "獎勵曲線"])
    
    with tab1:
        fig1 = create_pollutant_chart(df, st.session_state.current_step)
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        fig2 = create_comfort_chart(df, st.session_state.current_step)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        # 修改設備圖表，顯示到當前步的數據
        partial_equipment_usage = pd.DataFrame({
            '設備': ['排氣扇', '天花板風扇', '除濕機'],
            '使用時間 (分鐘)': [
                partial_df['exhaust_fan'].sum(),
                partial_df['ceiling_fan'].sum(),
                partial_df['dehumidifier'].sum()
            ],
            '顏色': ['#FF6B6B', '#4ECDC4', '#45B7D1']
        })
        
        fig3 = px.bar(
            partial_equipment_usage,
            x='設備',
            y='使用時間 (分鐘)',
            color='設備',
            color_discrete_map={
                '排氣扇': '#FF6B6B',
                '天花板風扇': '#4ECDC4',
                '除濕機': '#45B7D1'
            },
            title=f'設備使用時間統計 ({len(partial_df)}/{len(df)} 分鐘)'
        )
        fig3.update_layout(height=300, template='plotly_white')
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab4:
        # 修改獎勵圖表，支持動態顯示
        fig5 = go.Figure()
        
        # 即時獎勵
        fig5.add_trace(go.Scatter(
            x=partial_df['time_minutes'],
            y=partial_df['reward'],
            mode='lines+markers',
            name='即時獎勵',
            line=dict(color='#7209B7', width=2),
            marker=dict(size=4)
        ))
        
        # 累積獎勵
        cumulative_reward = partial_df['reward'].cumsum()
        fig5.add_trace(go.Scatter(
            x=partial_df['time_minutes'],
            y=cumulative_reward,
            mode='lines',
            name='累積獎勵',
            line=dict(color='#F72585', width=3)
        ))
        
        # 添加當前位置標記
        if len(partial_df) > 0:
            current_time_val = partial_df.iloc[-1]['time_minutes']
            current_reward = partial_df.iloc[-1]['reward']
            fig5.add_vline(x=current_time_val, line_dash="dash", line_color="gray", opacity=0.5)
            fig5.add_trace(go.Scatter(
                x=[current_time_val],
                y=[current_reward],
                mode='markers',
                name='當前獎勵',
                marker=dict(size=12, color='#FF0000'),
                showlegend=False
            ))
        
        fig5.update_layout(
            title=f'獎勵曲線 ({len(partial_df)}/{len(df)} 分鐘)',
            xaxis_title='時間 (分鐘)',
            yaxis_title='獎勵值',
            height=350,
            template='plotly_white'
        )
        st.plotly_chart(fig5, use_container_width=True)

if __name__ == "__main__":
    main()