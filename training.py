"""
Training module for the Credit Line Adjuster RL system
"""

import time
import numpy as np
from typing import Dict, Any, Tuple, List
from environment import CreditEnvironment
from agent import TDAgent

class CreditLineTrainer:
    """
    Handles training of the RL agent for credit line adjustments.
    """
    
    def __init__(self, environment: CreditEnvironment, agent: TDAgent):
        """
        Initialize the trainer.
        
        Args:
            environment: CreditEnvironment instance
            agent: TDAgent instance
        """
        self.env = environment
        self.agent = agent
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'q_value_changes': [],
            'training_times': [],
            'exploration_rates': []
        }
    
    def train(self, num_episodes: int = 1000, log_interval: int = 100) -> Dict[str, Any]:
        """
        Train the agent for specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
            log_interval: Interval for logging progress
            
        Returns:
            Training results dictionary
        """
        print(f"Starting training for {num_episodes} episodes...")
        start_time = time.time()
        
        total_reward = 0
        episode_rewards = []
        
        for episode in range(num_episodes):
            episode_start = time.time()
            state = self.env.reset()
            episode_reward = 0
            episode_q_changes = []
            
            # Single episode (one customer decision)
            done = False
            while not done:
                action = self.agent.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Update Q-value
                q_change = self.agent.update_q_value(state, action, reward, next_state)
                episode_q_changes.append(q_change)
                
                state = next_state
                episode_reward += reward
            
            # Decay exploration rate
            self.agent.decay_exploration()
            
            # Store metrics
            total_reward += episode_reward
            episode_rewards.append(episode_reward)
            
            self.training_metrics['episode_rewards'].append(episode_reward)
            self.training_metrics['episode_lengths'].append(len(episode_q_changes))
            self.training_metrics['q_value_changes'].append(np.mean(episode_q_changes) if episode_q_changes else 0)
            self.training_metrics['training_times'].append(time.time() - episode_start)
            self.training_metrics['exploration_rates'].append(self.agent.exploration_rate)
            
            # Log progress
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(episode_rewards[-log_interval:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Average Reward: {avg_reward:.3f}, "
                      f"Exploration Rate: {self.agent.exploration_rate:.3f}")
        
        training_time = time.time() - start_time
        
        # Calculate final metrics
        results = {
            'total_episodes': num_episodes,
            'total_training_time': training_time,
            'average_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'final_exploration_rate': self.agent.exploration_rate,
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'convergence_episode': self._find_convergence_episode(episode_rewards),
            'training_metrics': self.training_metrics
        }
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final Average Reward: {results['average_reward']:.3f}")
        
        return results
    
    def _find_convergence_episode(self, rewards: List[float], window: int = 50) -> int:
        """
        Find the episode where rewards converge (stabilize).
        
        Args:
            rewards: List of episode rewards
            window: Window size for moving average
            
        Returns:
            Episode number where convergence occurred
        """
        if len(rewards) < window * 2:
            return len(rewards) // 2
        
        # Calculate moving averages
        moving_avgs = []
        for i in range(len(rewards) - window + 1):
            moving_avgs.append(np.mean(rewards[i:i + window]))
        
        # Find where the moving average stabilizes
        convergence_threshold = 0.01
        for i in range(1, len(moving_avgs)):
            if abs(moving_avgs[i] - moving_avgs[i-1]) < convergence_threshold:
                return i + window
        
        return len(rewards) // 2
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get current training progress metrics.
        
        Returns:
            Dictionary with training progress information
        """
        if not self.training_metrics['episode_rewards']:
            return {}
        
        recent_rewards = self.training_metrics['episode_rewards'][-100:] if len(self.training_metrics['episode_rewards']) >= 100 else self.training_metrics['episode_rewards']
        
        return {
            'episodes_completed': len(self.training_metrics['episode_rewards']),
            'current_avg_reward': np.mean(recent_rewards),
            'current_exploration_rate': self.agent.exploration_rate,
            'total_q_value_change': np.sum(self.training_metrics['q_value_changes'])
        }