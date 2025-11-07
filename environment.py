"""
Credit Environment for Reinforcement Learning
Defines the state space, actions, and rewards for credit line adjustments.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import random

class CreditEnvironment:
    """
    Environment simulating customer credit behavior and credit line adjustments.
    """
    
    def __init__(self, mode: str = "synthetic", data_path: Optional[str] = None):
        """
        Initialize the credit environment.
        
        Args:
            mode: "synthetic" or "dataset"
            data_path: Path to CSV file if using dataset mode
        """
        self.mode = mode
        self.data_path = data_path
        self.data = None
        self.current_customer_idx = 0
        
        # State dimensions (discretized)
        self.credit_util_bins = [0, 0.3, 0.7, 1.0]  # Low, Medium, High
        self.payment_score_bins = [0, 0.6, 0.8, 1.0]  # Poor, Fair, Good
        self.income_stability_bins = [0, 0.5, 1.0]  # Unstable, Stable
        self.late_payments_bins = [0, 1, 3, 10]  # None, Few, Many
        self.debt_income_bins = [0, 0.3, 0.5, 1.0]  # Low, Medium, High
        
        # Actions: 0=Decrease, 1=Maintain, 2=Increase
        self.actions = [0, 1, 2]
        self.action_names = {0: "Decrease", 1: "Maintain", 2: "Increase"}
        
        # Initialize state space
        self.state_space_size = (
            len(self.credit_util_bins) * 
            len(self.payment_score_bins) * 
            len(self.income_stability_bins) * 
            len(self.late_payments_bins) * 
            len(self.debt_income_bins)
        )
        
        self._load_data()
        
    def _load_data(self):
        """Load data based on mode."""
        if self.mode == "dataset" and self.data_path:
            try:
                self.data = pd.read_csv(self.data_path)
                print(f"Loaded dataset with {len(self.data)} records")
            except Exception as e:
                print(f"Error loading dataset: {e}. Switching to synthetic mode.")
                self.mode = "synthetic"
                self.data = None
        else:
            self.mode = "synthetic"
            self.data = None
    
    def _generate_synthetic_customer(self) -> Dict[str, float]:
        """Generate a synthetic customer profile."""
        return {
            'credit_utilization': random.uniform(0, 1),
            'payment_history_score': random.uniform(0, 1),
            'income_stability': random.uniform(0, 1),
            'late_payments': random.randint(0, 10),
            'debt_to_income': random.uniform(0, 1)
        }
    
    def _discretize_state(self, customer_data: Dict[str, float]) -> Tuple:
        """Convert continuous customer data to discrete state representation."""
        util = np.digitize(customer_data['credit_utilization'], self.credit_util_bins) - 1
        payment = np.digitize(customer_data['payment_history_score'], self.payment_score_bins) - 1
        income = np.digitize(customer_data['income_stability'], self.income_stability_bins) - 1
        late = np.digitize(customer_data['late_payments'], self.late_payments_bins) - 1
        debt = np.digitize(customer_data['debt_to_income'], self.debt_income_bins) - 1
        
        return (util, payment, income, late, debt)
    
    def reset(self) -> Tuple:
        """Reset environment and return initial state."""
        if self.mode == "synthetic":
            self.current_customer = self._generate_synthetic_customer()
        else:
            if self.data is not None and len(self.data) > 0:
                customer_row = self.data.iloc[self.current_customer_idx % len(self.data)]
                self.current_customer = {
                    'credit_utilization': customer_row.get('credit_utilization', random.uniform(0, 1)),
                    'payment_history_score': customer_row.get('payment_history_score', random.uniform(0, 1)),
                    'income_stability': customer_row.get('income_stability', random.uniform(0, 1)),
                    'late_payments': customer_row.get('late_payments', random.randint(0, 10)),
                    'debt_to_income': customer_row.get('debt_to_income', random.uniform(0, 1))
                }
                self.current_customer_idx += 1
            else:
                self.current_customer = self._generate_synthetic_customer()
        
        self.state = self._discretize_state(self.current_customer)
        return self.state
    
    def step(self, action: int) -> Tuple[Tuple, float, bool, Dict]:
        """
        Take an action and return next state, reward, done, and info.
        
        Args:
            action: 0=Decrease, 1=Maintain, 2=Increase
            
        Returns:
            next_state, reward, done, info
        """
        reward = self._calculate_reward(action)
        done = True  # Each customer is a single episode
        info = {
            'customer_data': self.current_customer,
            'action_taken': self.action_names[action],
            'state': self.state
        }
        
        # Move to next customer
        next_state = self.reset()
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward based on action and customer state.
        
        Reward logic:
        - Positive for good decisions that lead to profit
        - Negative for bad decisions that lead to risk
        """
        customer = self.current_customer
        util = customer['credit_utilization']
        payment_score = customer['payment_history_score']
        income_stability = customer['income_stability']
        late_payments = customer['late_payments']
        debt_income = customer['debt_to_income']
        
        base_reward = 0
        
        # Reward based on credit utilization and action
        if util > 0.7 and action == 2:  # High utilization, increasing credit = risky
            base_reward -= 1
        elif util > 0.7 and action == 0:  # High utilization, decreasing credit = good
            base_reward += 1
        elif util < 0.3 and action == 2:  # Low utilization, increasing credit = good for profit
            base_reward += 1
        elif util < 0.3 and action == 0:  # Low utilization, decreasing credit = missed opportunity
            base_reward -= 0.5
        
        # Reward based on payment history
        if payment_score > 0.8 and action == 2:  # Good payer, increasing credit = good
            base_reward += 1
        elif payment_score < 0.6 and action == 2:  # Poor payer, increasing credit = risky
            base_reward -= 1
        
        # Reward based on income stability
        if income_stability > 0.7 and action == 2:  # Stable income, increasing credit = good
            base_reward += 0.5
        elif income_stability < 0.4 and action == 2:  # Unstable income, increasing credit = risky
            base_reward -= 1
        
        # Reward based on late payments
        if late_payments > 3 and action == 2:  # Many late payments, increasing credit = very risky
            base_reward -= 2
        elif late_payments == 0 and action == 2:  # No late payments, increasing credit = good
            base_reward += 1
        
        # Reward based on debt-to-income
        if debt_income > 0.5 and action == 2:  # High debt, increasing credit = risky
            base_reward -= 1
        elif debt_income < 0.3 and action == 2:  # Low debt, increasing credit = good
            base_reward += 0.5
        
        # Small penalty for frequent changes (prefer stability)
        if action != 1:  # If not maintain
            base_reward -= 0.1
        
        return base_reward
    
    def get_state_space_size(self) -> int:
        """Get the size of the state space."""
        return self.state_space_size
    
    def get_action_space_size(self) -> int:
        """Get the size of the action space."""
        return len(self.actions)
    
    def get_current_customer_data(self) -> Dict[str, float]:
        """Get the current customer's raw data."""
        return self.current_customer.copy()