"""
Evaluation module for assessing trained RL agent performance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from environment import CreditEnvironment
from agent import TDAgent

class CreditLineEvaluator:
    """
    Evaluates the performance of the trained credit line adjustment agent.
    """
    
    def __init__(self, environment: CreditEnvironment, agent: TDAgent):
        """
        Initialize the evaluator.
        
        Args:
            environment: CreditEnvironment instance
            agent: TDAgent instance
        """
        self.env = environment
        self.agent = agent
    
    def evaluate_policy(self, num_test_episodes: int = 100) -> Dict[str, Any]:
        """
        Evaluate the trained policy on test episodes.
        
        Args:
            num_test_episodes: Number of test episodes
            
        Returns:
            Evaluation results dictionary
        """
        print(f"Evaluating policy on {num_test_episodes} test episodes...")
        
        test_rewards = []
        action_distribution = {0: 0, 1: 0, 2: 0}  # Count of each action
        state_action_pairs = []
        
        for episode in range(num_test_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Use greedy policy (no exploration)
                action = self.agent.get_best_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                action_distribution[action] += 1
                state_action_pairs.append((state, action, reward))
                
                state = next_state
            
            test_rewards.append(episode_reward)
        
        # Calculate metrics
        results = {
            'average_reward': np.mean(test_rewards),
            'std_reward': np.std(test_rewards),
            'min_reward': np.min(test_rewards),
            'max_reward': np.max(test_rewards),
            'action_distribution': action_distribution,
            'action_percentages': {
                action: count / sum(action_distribution.values()) * 100 
                for action, count in action_distribution.items()
            },
            'test_rewards': test_rewards,
            'state_action_pairs': state_action_pairs
        }
        
        print(f"Evaluation completed. Average Reward: {results['average_reward']:.3f}")
        return results
    
    def plot_training_metrics(self, training_metrics: Dict[str, List], save_path: str = None):
        """
        Plot training metrics including rewards and exploration rate.
        
        Args:
            training_metrics: Dictionary from training
            save_path: Optional path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(1, len(training_metrics['episode_rewards']) + 1)
        
        # Plot episode rewards
        ax1.plot(episodes, training_metrics['episode_rewards'], 'b-', alpha=0.6, linewidth=0.8)
        
        # Add moving average
        window = min(50, len(episodes) // 10)
        if window > 0:
            moving_avg = np.convolve(training_metrics['episode_rewards'], 
                                   np.ones(window)/window, mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, 
                    label=f'Moving Avg (window={window})')
            ax1.legend()
        
        ax1.set_title('Episode Rewards During Training')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot exploration rate
        ax2.plot(episodes, training_metrics['exploration_rates'], 'g-')
        ax2.set_title('Exploration Rate Decay')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Exploration Rate (ε)')
        ax2.grid(True, alpha=0.3)
        
        # Plot Q-value changes
        ax3.plot(episodes, training_metrics['q_value_changes'], 'orange')
        ax3.set_title('Average Q-value Changes')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Q-value Change')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Plot reward distribution
        ax4.hist(training_metrics['episode_rewards'], bins=50, alpha=0.7, edgecolor='black')
        ax4.set_title('Distribution of Episode Rewards')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_q_value_heatmap(self, save_path: str = None):
        """
        Create a heatmap visualization of Q-values.
        
        Args:
            save_path: Optional path to save the plot
        """
        # Sample some states for visualization
        sample_states = []
        for util in range(3):  # credit utilization
            for payment in range(3):  # payment history
                state = (util, payment, 1, 1, 1)  # Fixed other dimensions for simplicity
                sample_states.append(state)
        
        q_values_sample = []
        for state in sample_states:
            state_idx = self.agent.get_state_index(state)
            q_values_sample.append(self.agent.q_table[state_idx])
        
        q_values_sample = np.array(q_values_sample)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(q_values_sample, cmap='RdYlGn', aspect='auto')
        
        # Set labels
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Decrease', 'Maintain', 'Increase'])
        ax.set_yticks(range(len(sample_states)))
        
        state_labels = []
        for state in sample_states:
            label = f"U{state[0]}P{state[1]}"
            state_labels.append(label)
        
        ax.set_yticklabels(state_labels)
        ax.set_ylabel('States (U=Utilization, P=Payment History)')
        ax.set_xlabel('Actions')
        ax.set_title('Q-values for Sample States')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Q-value')
        
        # Add text annotations
        for i in range(len(sample_states)):
            for j in range(3):
                text = ax.text(j, i, f'{q_values_sample[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_evaluation_report(self, training_results: Dict[str, Any], 
                                 evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            training_results: Results from training
            evaluation_results: Results from policy evaluation
            
        Returns:
            Comprehensive report dictionary
        """
        report = {
            'training_summary': {
                'total_episodes': training_results['total_episodes'],
                'total_training_time': training_results['total_training_time'],
                'final_average_reward': training_results['average_reward'],
                'convergence_episode': training_results['convergence_episode']
            },
            'evaluation_summary': {
                'test_episodes': len(evaluation_results['test_rewards']),
                'average_test_reward': evaluation_results['average_reward'],
                'reward_std': evaluation_results['std_reward'],
                'action_distribution': evaluation_results['action_percentages']
            },
            'performance_metrics': {
                'reward_improvement': evaluation_results['average_reward'] - training_results['average_reward'],
                'training_efficiency': training_results['average_reward'] / training_results['total_training_time'] if training_results['total_training_time'] > 0 else 0,
                'policy_stability': 1 - (evaluation_results['std_reward'] / abs(evaluation_results['average_reward'])) if evaluation_results['average_reward'] != 0 else 0
            }
        }
        
        return report