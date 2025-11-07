"""
interface.py - Streamlit UI for Credit Line Adjuster (fixed evaluation integration)
- Tries multiple evaluation method names before falling back to a safe internal evaluator
- Uses unique keys for all plotly_chart calls to avoid Streamlit duplicate-ID errors
- Keeps agent/env/training logic unchanged (uses your existing files)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import tempfile
import os
from typing import Any, Dict

# Import your existing modules (unchanged)
from environment import CreditEnvironment
from agent import TDAgent
from training import CreditLineTrainer
from evaluation import CreditLineEvaluator
from utils import create_feature_explanation, calculate_risk_score

# --------------------------
# Helper plotting functions
# --------------------------
def plot_training_overview(metrics: Dict[str, Any], title="Training Performance Overview"):
    episodes = list(range(1, len(metrics['episode_rewards']) + 1))
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Episode Rewards', 'Exploration Rate', 'Moving Average Reward', 'Reward Distribution'],
        vertical_spacing=0.12
    )

    fig.add_trace(go.Scatter(x=episodes, y=metrics['episode_rewards'],
                             mode='lines', name='Episode Reward',
                             line=dict(width=2, color='#1f77b4')), row=1, col=1)

    fig.add_trace(go.Scatter(x=episodes, y=metrics['exploration_rates'],
                             mode='lines', name='Exploration Rate',
                             line=dict(width=2, dash='dash', color='#ff7f0e')), row=1, col=2)

    window = max(1, min(50, len(episodes) // 10))
    if len(episodes) >= window:
        ma = np.convolve(metrics['episode_rewards'], np.ones(window)/window, mode='valid')
        fig.add_trace(go.Scatter(x=list(range(window, len(episodes)+1)), y=ma,
                                 mode='lines', name=f'Moving Avg (w={window})',
                                 line=dict(width=3, color='#2ca02c')), row=2, col=1)

    fig.add_trace(go.Histogram(x=metrics['episode_rewards'], nbinsx=30, name='Reward Dist',
                              marker_color='#1f77b4'), row=2, col=2)

    fig.update_layout(height=820, title_text=title, title_x=0.5, font=dict(size=13),
                      plot_bgcolor='rgba(248,249,250,1)',
                      legend=dict(orientation='h', y=-0.12, x=0.5))
    # Axis labels (applies globally where possible)
    fig.update_xaxes(title_text="Episode", row=1, col=1)
    fig.update_yaxes(title_text="Reward", row=1, col=1)
    fig.update_xaxes(title_text="Episode", row=1, col=2)
    fig.update_yaxes(title_text="Exploration (ε)", row=1, col=2)
    fig.update_xaxes(title_text="Episode", row=2, col=1)
    fig.update_yaxes(title_text="Moving Avg Reward", row=2, col=1)
    fig.update_xaxes(title_text="Reward", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    return fig

def plot_line(y, title="Plot", y_title="Value", key=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode='lines', line=dict(width=2, color='#1f77b4')))
    fig.update_layout(title=title, xaxis_title="Index", yaxis_title=y_title, 
                     height=420, font=dict(size=13), plot_bgcolor='rgba(248,249,250,1)')
    return fig

def plot_hist(x, title="Distribution", x_title="Value", key=None):
    fig = px.histogram(x=x, nbins=30, title=title)
    fig.update_layout(xaxis_title=x_title, yaxis_title="Count", height=420, 
                     font=dict(size=13), plot_bgcolor='rgba(248,249,250,1)')
    fig.update_traces(marker_color='#1f77b4')
    return fig

# --------------------------
# Main Interface class
# --------------------------
class CreditLineInterface:
    def __init__(self):
        st.set_page_config(page_title="Credit Line Adjuster", page_icon="💰", layout="wide")
        self._apply_css()
        self._init_session_state()

    def _apply_css(self):
        st.markdown(
            """
            <style>
            /* Professional Blue Theme */
            .main-header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 2.5rem;
                border-radius: 0px;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 20px rgba(30, 60, 114, 0.3);
            }
            
            .logo-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1.5rem;
            }
            
            .logo {
                height: 80px;
                max-width: 220px;
                object-fit: contain;
                filter: brightness(0) invert(1);
            }
            
            .footer {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                padding: 2rem;
                border-radius: 0px;
                margin-top: 3rem;
                text-align: center;
                box-shadow: 0 -4px 20px rgba(30, 60, 114, 0.3);
            }
            
            .professional-card {
                background: white;
                padding: 2rem;
                border-radius: 12px;
                border-left: 5px solid #2a5298;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                margin-bottom: 2rem;
                border: 1px solid #e1e8ed;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            
            .professional-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 25px rgba(0,0,0,0.12);
            }
            
            .metric-card {
                background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
                padding: 1.5rem;
                border-radius: 10px;
                border: 1px solid #e1e8ed;
                box-shadow: 0 2px 12px rgba(0,0,0,0.06);
                text-align: center;
            }
            
            [data-testid="stMetricValue"] { 
                font-size: 1.6rem; 
                font-weight: 700; 
                color: #1e3c72;
            }
            
            [data-testid="stMetricLabel"] {
                font-weight: 600;
                color: #495057;
                font-size: 0.9rem;
            }
            
            .block-container { 
                padding-top: 1rem; 
                padding-bottom: 1rem; 
            }
            
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
                border-right: 1px solid #e1e8ed;
            }
            
            section[data-testid="stSidebar"] h2 { 
                color: #1e3c72; 
                border-bottom: 3px solid #2a5298;
                padding-bottom: 0.7rem;
                font-weight: 700;
            }
            
            .stButton button {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.7rem 1.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
                width: 100%;
            }
            
            .stButton button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(42, 82, 152, 0.4);
                background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem;
                background: #f8f9fa;
                padding: 0.5rem;
                border-radius: 12px;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 60px;
                white-space: pre-wrap;
                background-color: white;
                border-radius: 8px;
                padding: 0px 25px;
                font-weight: 600;
                border: 1px solid #e1e8ed;
                transition: all 0.3s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                border: none;
                box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3);
            }
            
            .stTabs [aria-selected="false"]:hover {
                background-color: #e9ecef;
                border-color: #2a5298;
            }
            
            /* Custom slider styling */
            .stSlider [data-testid="stThumb"] {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                border: 2px solid white;
                box-shadow: 0 2px 6px rgba(30, 60, 114, 0.3);
            }
            
            .stSlider [data-testid="stTickBar"] {
                background: #e9ecef;
            }
            
            /* Success and warning messages */
            .stAlert [data-testid="stMarkdownContainer"] {
                font-weight: 500;
            }
            
            /* Dataframe styling */
            .dataframe {
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            }
            
            .progress-bar {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            }
            </style>
            """, unsafe_allow_html=True
        )

    def _init_session_state(self):
        keys = [
            'agent', 'environment', 'training_results', 'evaluation_results',
            'training_metrics', 'system_initialized', 'trained_agent', 'last_config'
        ]
        for k in keys:
            if k not in st.session_state:
                st.session_state[k] = None if k != 'system_initialized' else False

    def _render_header(self):
        st.markdown(
            """
            <div class="main-header">
                <div class="logo-container">
                    <img src="https://enit.rnu.tn/wp-content/uploads/2019/07/LOGO_ENIT_300.png" class="logo" alt="ENIT Logo">
                    <div>
                        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 800; letter-spacing: -0.5px;">💰 Credit Line Adjuster</h1>
                        <p style="margin: 0.8rem 0 0 0; font-size: 1.3rem; opacity: 0.95; font-weight: 400;">
                            AI-Powered Credit Optimization System
                        </p>
                    </div>
                    <img src="https://www.entreprises-magazine.com/wp-content/uploads/2023/03/Logo-bna-V213430.png" class="logo" alt="BIAT Logo">
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    def _render_footer(self):
        st.markdown(
            """
            <div class="footer">
                <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.8rem;">
                    👨‍💻 Developed by Ayoub ABIDI and Tasnim KHEDHRI
                </div>
                <div style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 0.5rem;">
                    🎓 Supervised by Dr. Wafa MEFTEH
                </div>
                <div style="font-size: 1rem; opacity: 0.8;">
                    3rd-year Telecommunications Engineering students at ENIT
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    # --------------------------
    # Sidebar
    # --------------------------
    def render_sidebar(self):
        st.sidebar.title(" System Configuration")
        st.sidebar.markdown("---")
        
        st.sidebar.subheader("Data Source")
        mode = st.sidebar.radio("Select data source:", ["Synthetic Data", "Upload Dataset"])
        uploaded_file = None
        if mode == "Upload Dataset":
            uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.sidebar.success(f"✅ Loaded {len(df)} records")
                    with st.sidebar.expander("📋 Data Preview"):
                        st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.sidebar.error(f"❌ Error reading file: {e}")

        st.sidebar.markdown("---")
        st.sidebar.subheader(" RL Hyperparameters")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            learning_rate = st.sidebar.slider("Learning Rate (α)", 0.01, 0.5, 0.1, 0.01)
            discount_factor = st.sidebar.slider("Discount Factor (γ)", 0.5, 0.99, 0.9, 0.01)
        with col2:
            exploration_rate = st.sidebar.slider("Initial Exploration (ε)", 0.01, 1.0, 0.3, 0.01)
            exploration_decay = st.sidebar.slider("Exploration Decay", 0.990, 0.9999, 0.995, 0.0001, format="%.4f")

        num_episodes = st.sidebar.number_input("Number of Episodes", min_value=100, max_value=20000, value=1000, step=100)

        st.sidebar.markdown("---")
        config = {
            'mode': mode.lower().replace(" ", "_"),
            'uploaded_file': uploaded_file,
            'learning_rate': float(learning_rate),
            'discount_factor': float(discount_factor),
            'exploration_rate': float(exploration_rate),
            'exploration_decay': float(exploration_decay),
            'num_episodes': int(num_episodes)
        }
        st.session_state.last_config = config
        
        if st.sidebar.button("🚀 Initialize System", use_container_width=True, key="btn_init"):
            self.initialize_system(config)
            
        return config

    # --------------------------
    # Initialize system
    # --------------------------
    def initialize_system(self, config):
        try:
            if config['mode'] == 'upload_dataset' and config['uploaded_file'] is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                    tmp.write(config['uploaded_file'].getvalue())
                    tmp_path = tmp.name
                env = CreditEnvironment(mode="dataset", data_path=tmp_path)
                os.unlink(tmp_path)
            else:
                env = CreditEnvironment(mode="synthetic")

            agent = TDAgent(
                state_space_size=env.get_state_space_size(),
                action_space_size=env.get_action_space_size(),
                learning_rate=config['learning_rate'],
                discount_factor=config['discount_factor'],
                exploration_rate=config['exploration_rate'],
                exploration_decay=config['exploration_decay']
            )

            st.session_state.environment = env
            st.session_state.agent = agent
            st.session_state.system_initialized = True
            st.session_state.training_results = None
            st.session_state.evaluation_results = None
            st.session_state.training_metrics = None
            st.success("✅ System initialized successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Initialization failed: {e}")
            return False

    # --------------------------
    # Training
    # --------------------------
    def render_training_section(self):
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.header(" Model Training")
        
        if not st.session_state.system_initialized:
            st.warning("⚠️ Please initialize the system first in the sidebar.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        cfg = st.session_state.last_config or {}
        num_episodes = cfg.get('num_episodes', 1000)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if st.button("🎯 Start Training", use_container_width=True, key="btn_start_training"):
                self.run_training(num_episodes)

            if st.session_state.training_results:
                if st.button("🔄 Retrain Model", use_container_width=True, key="btn_retrain"):
                    try:
                        st.session_state.agent.reset_training()
                    except Exception:
                        pass
                    self.run_training(num_episodes)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if st.session_state.training_results:
                r = st.session_state.training_results
                col2_1, col2_2, col2_3 = st.columns(3)
                with col2_1:
                    st.metric("Average Reward", f"{r['average_reward']:.3f}")
                with col2_2:
                    st.metric("Training Time", f"{r['total_training_time']:.2f}s")
                with col2_3:
                    st.metric("Convergence", f"Episode {r.get('convergence_episode', 'N/A')}")

        if st.session_state.training_metrics:
            st.markdown("---")
            st.subheader("📈 Training Analytics")
            fig = plot_training_overview(st.session_state.training_metrics)
            st.plotly_chart(fig, use_container_width=True, key="training_overview")

            # recent episodes table
            recent_n = min(50, len(st.session_state.training_metrics['episode_rewards']))
            df = pd.DataFrame({
                'Episode': list(range(len(st.session_state.training_metrics['episode_rewards']) - recent_n + 1,
                                       len(st.session_state.training_metrics['episode_rewards']) + 1)),
                'Reward': st.session_state.training_metrics['episode_rewards'][-recent_n:],
                'Exploration': st.session_state.training_metrics['exploration_rates'][-recent_n:]
            })
            st.subheader("📋 Recent Training Episodes")
            st.dataframe(df.style.format({'Reward': "{:.4f}", 'Exploration': "{:.4f}"}), 
                        height=280, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def run_training(self, num_episodes: int):
        env = st.session_state.environment
        agent = st.session_state.agent
        trainer = CreditLineTrainer(env, agent)
        
        st.subheader("🔄 Training Progress")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        metrics_placeholder = st.empty()

        episode_rewards = []
        exploration_rates = []
        q_changes = []

        start_time = time.time()
        for ep in range(num_episodes):
            # try reset signature variations
            try:
                state = env.reset()
            except TypeError:
                state, _ = env.reset()

            done = False
            ep_reward = 0.0
            ep_q_changes = []

            while not done:
                # choose action - try methods in order
                if hasattr(agent, "choose_action"):
                    action = agent.choose_action(state)
                elif hasattr(agent, "get_best_action"):
                    action = agent.get_best_action(state)
                elif hasattr(agent, "act"):
                    action = agent.act(state)
                else:
                    raise AttributeError("Agent missing choose_action/get_best_action/act")

                next_state, reward, done, info = env.step(action)
                # Update Q; adapt to different return signatures
                q_change = None
                try:
                    q_change = agent.update_q_value(state, action, reward, next_state)
                except TypeError:
                    # maybe update_q_value returns nothing; call but ignore
                    try:
                        agent.update_q_value(state, action, reward, next_state)
                    except Exception:
                        pass

                if isinstance(q_change, (int, float)):
                    ep_q_changes.append(abs(q_change))

                state = next_state
                ep_reward += float(reward)

            # exploration decay if exists
            if hasattr(agent, "decay_exploration"):
                try:
                    agent.decay_exploration()
                except Exception:
                    pass

            episode_rewards.append(ep_reward)
            exploration_rates.append(getattr(agent, "exploration_rate", 0.0))
            q_changes.append(np.mean(ep_q_changes) if ep_q_changes else 0.0)

            if (ep + 1) % max(1, (num_episodes // 20)) == 0:
                progress = (ep + 1) / num_episodes
                progress_bar.progress(progress)
                recent_avg = np.mean(episode_rewards[-20:]) if episode_rewards else 0.0
                status_text.text(f"📊 Episode {ep+1}/{num_episodes} | Recent Avg Reward: {recent_avg:.3f} | Exploration: {getattr(agent, 'exploration_rate', 0.0):.4f}")

        total_time = time.time() - start_time
        st.session_state.training_metrics = {
            'episode_rewards': episode_rewards,
            'exploration_rates': exploration_rates,
            'q_value_changes': q_changes
        }
        conv = self._find_convergence_episode(episode_rewards)
        st.session_state.training_results = {
            'total_episodes': num_episodes,
            'total_training_time': total_time,
            'average_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'final_exploration_rate': float(getattr(agent, "exploration_rate", 0.0)),
            'convergence_episode': conv,
            'max_reward': float(np.max(episode_rewards)),
            'min_reward': float(np.min(episode_rewards))
        }
        st.session_state.trained_agent = st.session_state.agent
        progress_bar.progress(1.0)
        status_text.text("✅ Training completed successfully!")
        st.success("🎉 Training finished! Model is ready for evaluation.")

    def _find_convergence_episode(self, rewards, window=50):
        if len(rewards) < window * 2:
            return len(rewards)
        mov = np.convolve(rewards, np.ones(window)/window, mode='valid')
        diffs = np.abs(np.diff(mov))
        idx = np.where(diffs < 1e-3)[0]
        return int(idx[0] + window) if idx.size else len(rewards)

    # --------------------------
    # Evaluation
    # --------------------------
    def render_evaluation_section(self):
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.header("📈 Model Evaluation")
        
        if st.session_state.trained_agent is None:
            st.warning("⚠️ Please train the agent first before evaluation.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if st.button("📊 Run Evaluation", use_container_width=True, key="btn_eval"):
                self.run_evaluation()
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.evaluation_results:
            res = st.session_state.evaluation_results
            # Friendly metric names (fallbacks)
            avg_reward = res.get('average_reward') or res.get('avg_reward') or (np.mean(res.get('test_rewards', [])) if res.get('test_rewards') else 0.0)
            
            st.subheader("📋 Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Test Reward", f"{avg_reward:.3f}")
            # Action distribution if present
            if 'action_percentages' in res:
                ap = res['action_percentages']
                with col2:
                    st.metric("Decrease Actions", f"{ap.get(0,0):.1f}%")
                with col3:
                    st.metric("Maintain Actions", f"{ap.get(1,0):.1f}%")
                with col1:
                    st.metric("Increase Actions", f"{ap.get(2,0):.1f}%")    

            st.markdown("---")
            st.subheader("📊 Performance Visualizations")
            # plots (unique keys)
            if 'test_rewards' in res and res['test_rewards']:
                fig1 = plot_hist(res['test_rewards'], title="Test Reward Distribution", x_title="Reward")
                st.plotly_chart(fig1, use_container_width=True, key="eval_rewards_hist")
            if 'history' in res and isinstance(res['history'], (list, pd.DataFrame)):
                # if it's a list of dicts convert to df
                hist_df = pd.DataFrame(res['history']) if not isinstance(res['history'], pd.DataFrame) else res['history']
                if 'profit' in hist_df.columns:
                    fig2 = px.line(hist_df, x=hist_df.index if 'episode' not in hist_df.columns else 'episode', y='profit', 
                                  title="Profit Over Episodes", line_shape='spline')
                    fig2.update_traces(line=dict(width=3, color='#1e3c72'))
                    fig2.update_layout(plot_bgcolor='rgba(248,249,250,1)')
                    st.plotly_chart(fig2, use_container_width=True, key="eval_profit_line")
        
        st.markdown('</div>', unsafe_allow_html=True)

    def run_evaluation(self, num_test_episodes: int = 200):
        env = st.session_state.environment
        agent = st.session_state.trained_agent or st.session_state.agent
        evaluator = None
        # try to instantiate evaluator safely
        try:
            evaluator = CreditLineEvaluator(env, agent)
        except Exception:
            # try alternate signatures
            try:
                evaluator = CreditLineEvaluator(agent, env)
            except Exception:
                evaluator = None

        results = None
        # Try multiple evaluation method names on evaluator object if available
        if evaluator is not None:
            for name in ("evaluate_model", "evaluate_policy", "evaluate", "evaluate_agent"):
                if hasattr(evaluator, name):
                    try:
                        fn = getattr(evaluator, name)
                        # attempt call with episodes or num_test_episodes
                        try:
                            results = fn(episodes=num_test_episodes)
                        except TypeError:
                            try:
                                results = fn(num_test_episodes)
                            except TypeError:
                                results = fn()
                        break
                    except Exception:
                        results = None

        # If no evaluator or call failed, fallback to internal evaluation loop
        if results is None:
            # Safe fallback: run episodes using agent's deterministic policy
            test_rewards = []
            action_counts = {}
            for ep in range(num_test_episodes):
                try:
                    state = env.reset()
                except TypeError:
                    state, _ = env.reset()
                done = False
                ep_reward = 0.0
                while not done:
                    # pick deterministic best action
                    if hasattr(agent, "get_best_action"):
                        action = agent.get_best_action(state)
                    elif hasattr(agent, "choose_action"):
                        # assume choose_action has deterministic behavior when not training, but if it's epsilon-greedy this isn't ideal
                        action = agent.choose_action(state)
                    elif hasattr(agent, "act"):
                        action = agent.act(state)
                    else:
                        raise AttributeError("Agent lacks get_best_action/choose_action/act for evaluation fallback")

                    next_state, reward, done, info = env.step(action)
                    ep_reward += float(reward)
                    action_counts[action] = action_counts.get(action, 0) + 1
                    state = next_state
                test_rewards.append(ep_reward)

            total_actions = sum(action_counts.values()) or 1
            action_percentages = {k: (v / total_actions) * 100 for k, v in action_counts.items()}
            results = {
                'test_rewards': test_rewards,
                'average_reward': float(np.mean(test_rewards)),
                'action_percentages': action_percentages
            }

        st.session_state.evaluation_results = results
        st.success("✅ Evaluation completed successfully!")

    # --------------------------
    # Simulation
    # --------------------------
    def render_simulation_section(self):
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.header("👥 Customer Credit Simulation")
        
        if st.session_state.trained_agent is None:
            st.warning("⚠️ Please train the agent first to get recommendations.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        st.subheader("📋 Customer Profile Input")
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            credit_utilization = st.slider("Credit Utilization Ratio", 0.0, 1.0, 0.5, 0.01)
            payment_history_score = st.slider("Payment History Score", 0.0, 1.0, 0.8, 0.01)
            income_stability = st.slider("Income Stability Index", 0.0, 1.0, 0.7, 0.01)
        with c2:
            late_payments = st.number_input("Late Payments (12 months)", 0, 50, 1)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, 0.01)
        st.markdown('</div>', unsafe_allow_html=True)

        customer_data = {
            'credit_utilization': float(credit_utilization),
            'payment_history_score': float(payment_history_score),
            'income_stability': float(income_stability),
            'late_payments': int(late_payments),
            'debt_to_income': float(debt_to_income)
        }

        if st.button("🎯 Get Credit Recommendation", use_container_width=True, key="btn_recommend"):
            # discretize state if env supports it
            env = st.session_state.environment
            agent = st.session_state.trained_agent or st.session_state.agent
            try:
                if hasattr(env, "_discretize_state"):
                    state = env._discretize_state(customer_data)
                else:
                    # try to produce a state vector if env has a mapping
                    state = np.array(list(customer_data.values()))
            except Exception:
                state = np.array(list(customer_data.values()))

            # get action probabilities or best action
            action_probs = None
            best_action = None
            if hasattr(agent, "get_action_probabilities"):
                try:
                    action_probs = agent.get_action_probabilities(state)
                except Exception:
                    action_probs = None
            if hasattr(agent, "get_best_action"):
                try:
                    best_action = agent.get_best_action(state)
                except Exception:
                    best_action = None
            if best_action is None:
                if hasattr(agent, "choose_action"):
                    best_action = agent.choose_action(state)
                elif hasattr(agent, "act"):
                    best_action = agent.act(state)
                else:
                    st.error("❌ Agent does not implement a method to choose an action.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return

            action_names = {0: "Decrease Credit", 1: "Maintain Credit", 2: "Increase Credit"}
            action_colors = {"Decrease Credit": "#ff6b6b", "Maintain Credit": "#ffd93d", "Increase Credit": "#6bcf7f"}
            prob_list = list(action_probs) if action_probs else [0.0] * 3
            prob_list = prob_list[:3] + [0.0] * max(0, 3 - len(prob_list))

            # Show gauges and barplots
            st.subheader("📊 Recommendation Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                conf = float(prob_list[best_action] * 100) if prob_list else 0.0
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=conf,
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1e3c72"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "lightblue"},
                            {'range': [80, 100], 'color': "#2a5298"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    },
                    title={'text': f"Confidence<br>{action_names.get(best_action, best_action)}", 'font': {'size': 16}}
                ))
                fig_g.update_layout(height=300, font={'size': 12})
                st.plotly_chart(fig_g, use_container_width=True, key="sim_gauge_conf")

            with col2:
                risk = calculate_risk_score(customer_data)
                fig_r = go.Figure(go.Indicator(
                    mode="gauge+number", 
                    value=float(risk*100),
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#d4af37"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ]
                    },
                    title={'text': "Risk Assessment<br>Score", 'font': {'size': 16}}
                ))
                fig_r.update_layout(height=300, font={'size': 12})
                st.plotly_chart(fig_r, use_container_width=True, key="sim_gauge_risk")

            with col3:
                fig_bar = px.bar(
                    x=list(action_names.values()), 
                    y=prob_list[:3],
                    labels={'x': 'Recommended Action', 'y': 'Probability'}, 
                    title="Action Probability Distribution",
                    color=list(action_names.values()),
                    color_discrete_map=action_colors
                )
                fig_bar.update_layout(
                    height=300, 
                    yaxis=dict(range=[0, 1]),
                    plot_bgcolor='rgba(248,249,250,1)',
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True, key="sim_bar_probs")

            st.markdown("---")
            st.subheader("📋 Customer Risk Analysis")
            explanations = create_feature_explanation(customer_data)
            for k, v in explanations.items():
                st.write(f"**{k.replace('_', ' ').title()}:** {v}")

            st.success(f"🎯 **Recommended Action:** {action_names.get(best_action, best_action)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------
    # Dashboard & Main
    # --------------------------
    def render_dashboard(self):
        st.markdown('<div class="professional-card">', unsafe_allow_html=True)
        st.header("System Dashboard")
        
        st.subheader("System Status Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            status = "✅ Initialized" if st.session_state.system_initialized else "❌ Not initialized"
            st.metric("System Status", status)
        with col2:
            status = "✅ Trained" if st.session_state.trained_agent else "❌ Not trained"
            st.metric("Model Status", status)
        with col3:
            status = "✅ Available" if st.session_state.evaluation_results else "❌ Not available"
            st.metric("Evaluation Data", status)

        if st.session_state.system_initialized:
            st.markdown("---")
            st.subheader("⚙️ Configuration Details")
            env = st.session_state.environment
            agent = st.session_state.agent
            try:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Environment Configuration**")
                    st.write(f"- 📊 Data Mode: {getattr(env, 'mode', 'n/a')}")
                    st.write(f"- 🏠 State Space: {env.get_state_space_size() if hasattr(env,'get_state_space_size') else 'n/a'}")
                    st.write(f"- 🎯 Action Space: {env.get_action_space_size() if hasattr(env,'get_action_space_size') else 'n/a'}")
                with col2:
                    st.write("**Agent Configuration**")
                    st.write(f"- 📈 Learning Rate: {getattr(agent, 'learning_rate', 'n/a')}")
                    st.write(f"- 💰 Discount Factor: {getattr(agent, 'discount_factor', 'n/a')}")
                    st.write(f"- 🔍 Exploration Rate: {getattr(agent, 'exploration_rate', 'n/a'):.4f}")
            except Exception:
                pass
                
        st.markdown('</div>', unsafe_allow_html=True)

    def render_main_interface(self):
        self._render_header()
        
        config = self.render_sidebar()

        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Dashboard", "⚙️ Training", "📈 Evaluation", "👥 Simulation"])
        with tab1:
            self.render_dashboard()
        with tab2:
            self.render_training_section()
        with tab3:
            self.render_evaluation_section()
        with tab4:
            self.render_simulation_section()
            
        self._render_footer()

# --------------------------
# Entrypoint
# --------------------------
def main():
    interface = CreditLineInterface()
    interface.render_main_interface()

if __name__ == "__main__":
    main()