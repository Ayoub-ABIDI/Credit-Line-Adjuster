"""
Reinforcement Learning Agent using Temporal Difference Learning (Q-learning)
"""

import numpy as np
import random
from typing import Tuple, List, Dict, Any
import pickle

class TDAgent:
    """
    Temporal Difference Learning Agent for credit line adjustments.
    Implements Q-learning algorithm.
    """
    
    def __init__(self, state_space_size: int, action_space_size: int, 
                 learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 exploration_rate: float = 0.1, exploration_decay: float = 0.995):
        """
        Initialize the TD agent.
        
        Args:
            state_space_size: Number of possible states
            action_space_size: Number of possible actions
            learning_rate: Alpha - how much to update Q-values
            discount_factor: Gamma - importance of future rewards
            exploration_rate: Epsilon - probability of random exploration
            exploration_decay: Rate at which exploration decreases
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.initial_exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        # Training history
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'exploration_rates': [],
            'q_value_changes': []
        }
    
    def get_state_index(self, state: Tuple) -> int:
        """
        Convert state tuple to a single index for Q-table.
        
        Args:
            state: Tuple representing discrete state
            
        Returns:
            Integer index for Q-table
        """
        # Flatten multi-dimensional state to single index
        index = 0
        multiplier = 1
        
        for i, dimension in enumerate(reversed(state)):
            index += dimension * multiplier
            # Calculate multiplier for next dimension
            if i == 0:  # debt_income (max ~3)
                multiplier *= 4
            elif i == 1:  # late_payments (max ~3)
                multiplier *= 4
            elif i == 2:  # income_stability (max ~2)
                multiplier *= 3
            elif i == 3:  # payment_history (max ~3)
                multiplier *= 4
            # credit_utilization handled by final dimension
        
        return index % self.state_space_size  # Ensure within bounds
    
    def choose_action(self, state: Tuple) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state tuple
            
        Returns:
            Action index
        """
        state_index = self.get_state_index(state)
        
        # Exploration: random action
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_space_size)
        
        # Exploitation: best action from Q-table
        return np.argmax(self.q_table[state_index])
    
    def update_q_value(self, state: Tuple, action: int, reward: float, next_state: Tuple) -> float:
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            
        Returns:
            Q-value change for tracking
        """
        state_index = self.get_state_index(state)
        next_state_index = self.get_state_index(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_index, action]
        
        # Maximum Q-value for next state
        max_next_q = np.max(self.q_table[next_state_index])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Update Q-table
        q_change = abs(new_q - current_q)
        self.q_table[state_index, action] = new_q
        
        return q_change
    
    def decay_exploration(self):
        """Decay exploration rate over time."""
        self.exploration_rate = max(0.01, self.exploration_rate * self.exploration_decay)
    
    def get_action_probabilities(self, state: Tuple) -> List[float]:
        """
        Get probability distribution over actions for a given state.
        
        Args:
            state: State tuple
            
        Returns:
            List of probabilities for each action
        """
        state_index = self.get_state_index(state)
        q_values = self.q_table[state_index]
        
        # Convert Q-values to probabilities using softmax
        exp_q = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
        probabilities = exp_q / np.sum(exp_q)
        
        return probabilities.tolist()
    
    def get_best_action(self, state: Tuple) -> int:
        """
        Get the best action for a state (without exploration).
        
        Args:
            state: State tuple
            
        Returns:
            Best action index
        """
        state_index = self.get_state_index(state)
        return np.argmax(self.q_table[state_index])
    
    def save_model(self, filepath: str):
        """Save the trained model to a file."""
        model_data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'state_space_size': self.state_space_size,
            'action_space_size': self.action_space_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.learning_rate = model_data['learning_rate']
        self.discount_factor = model_data['discount_factor']
        self.exploration_rate = model_data['exploration_rate']
        self.state_space_size = model_data['state_space_size']
        self.action_space_size = model_data['action_space_size']
    
    def reset_training(self):
        """Reset the agent for new training."""
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        self.exploration_rate = self.initial_exploration_rate
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'exploration_rates': [],
            'q_value_changes': []
        }