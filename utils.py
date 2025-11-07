"""
Utility functions for the Credit Line Adjuster system
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from typing import Dict, Any, List, Tuple, Optional
import json

def generate_sample_dataset(num_samples: int = 1000) -> pd.DataFrame:
    """
    Generate a sample dataset for credit line adjustments.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with sample customer data
    """
    np.random.seed(42)  # For reproducibility
    
    data = {
        'credit_utilization': np.random.beta(2, 5, num_samples),  # Most customers have low utilization
        'payment_history_score': np.random.beta(8, 2, num_samples),  # Most have good payment history
        'income_stability': np.random.beta(6, 2, num_samples),  # Most have stable income
        'late_payments': np.random.poisson(0.5, num_samples),  # Most have 0-1 late payments
        'debt_to_income': np.random.beta(3, 6, num_samples),  # Most have low debt-to-income
    }
    
    df = pd.DataFrame(data)
    
    # Clip values to reasonable ranges
    df['credit_utilization'] = df['credit_utilization'].clip(0, 1)
    df['payment_history_score'] = df['payment_history_score'].clip(0, 1)
    df['income_stability'] = df['income_stability'].clip(0, 1)
    df['late_payments'] = df['late_payments'].clip(0, 10)
    df['debt_to_income'] = df['debt_to_income'].clip(0, 1)
    
    return df

def save_model(agent, filepath: str):
    """
    Save trained agent to file.
    
    Args:
        agent: Trained TDAgent
        filepath: Path to save the model
    """
    agent.save_model(filepath)

def load_model(filepath: str, state_space_size: int, action_space_size: int):
    """
    Load trained agent from file.
    
    Args:
        filepath: Path to saved model
        state_space_size: Size of state space
        action_space_size: Size of action space
        
    Returns:
        Loaded TDAgent
    """
    from agent import TDAgent
    
    agent = TDAgent(state_space_size, action_space_size)
    agent.load_model(filepath)
    return agent

def normalize_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize customer features to [0, 1] range.
    
    Args:
        features: Dictionary of raw features
        
    Returns:
        Dictionary of normalized features
    """
    normalized = features.copy()
    
    # Normalize credit utilization (already 0-1)
    # Normalize payment history score (already 0-1)
    # Normalize income stability (already 0-1)
    
    # Normalize late payments (0-10 -> 0-1, inverted so lower is better)
    normalized['late_payments'] = 1 - (features['late_payments'] / 10)
    normalized['late_payments'] = max(0, min(1, normalized['late_payments']))
    
    # Normalize debt-to-income (already 0-1, inverted so lower is better)
    normalized['debt_to_income'] = 1 - features['debt_to_income']
    
    return normalized

def create_feature_explanation(features: Dict[str, float]) -> Dict[str, str]:
    """
    Create human-readable explanations for feature values.
    
    Args:
        features: Dictionary of feature values
        
    Returns:
        Dictionary of explanations
    """
    explanations = {}
    
    util = features['credit_utilization']
    if util < 0.3:
        explanations['credit_utilization'] = f"Low ({util:.1%}) - Underutilized credit"
    elif util < 0.7:
        explanations['credit_utilization'] = f"Moderate ({util:.1%}) - Healthy usage"
    else:
        explanations['credit_utilization'] = f"High ({util:.1%}) - Heavy usage, potential risk"
    
    payment = features['payment_history_score']
    if payment < 0.6:
        explanations['payment_history_score'] = f"Poor ({payment:.1%}) - Payment issues"
    elif payment < 0.8:
        explanations['payment_history_score'] = f"Fair ({payment:.1%}) - Some late payments"
    else:
        explanations['payment_history_score'] = f"Good ({payment:.1%}) - Consistent on-time payments"
    
    income = features['income_stability']
    if income < 0.5:
        explanations['income_stability'] = f"Unstable ({income:.1%}) - Income fluctuations"
    else:
        explanations['income_stability'] = f"Stable ({income:.1%}) - Steady income"
    
    late = features['late_payments']
    if late == 0:
        explanations['late_payments'] = f"Excellent ({late}) - No late payments"
    elif late <= 2:
        explanations['late_payments'] = f"Good ({late}) - Few late payments"
    elif late <= 5:
        explanations['late_payments'] = f"Concerning ({late}) - Several late payments"
    else:
        explanations['late_payments'] = f"Poor ({late}) - Many late payments"
    
    debt = features['debt_to_income']
    if debt < 0.3:
        explanations['debt_to_income'] = f"Low ({debt:.1%}) - Manageable debt"
    elif debt < 0.5:
        explanations['debt_to_income'] = f"Moderate ({debt:.1%}) - Some debt burden"
    else:
        explanations['debt_to_income'] = f"High ({debt:.1%}) - Significant debt burden"
    
    return explanations

def calculate_risk_score(features: Dict[str, float]) -> float:
    """
    Calculate a composite risk score for a customer.
    
    Args:
        features: Dictionary of customer features
        
    Returns:
        Risk score between 0 (low risk) and 1 (high risk)
    """
    weights = {
        'credit_utilization': 0.25,
        'payment_history_score': 0.30,  # Inverse - higher score = lower risk
        'income_stability': 0.15,  # Inverse - higher stability = lower risk
        'late_payments': 0.20,  # Higher late payments = higher risk
        'debt_to_income': 0.10,
    }
    
    # Normalize features for risk calculation
    risk_factors = {
        'credit_utilization': features['credit_utilization'],  # Higher = more risk
        'payment_history_score': 1 - features['payment_history_score'],  # Inverse
        'income_stability': 1 - features['income_stability'],  # Inverse
        'late_payments': features['late_payments'] / 10,  # Scale to 0-1
        'debt_to_income': features['debt_to_income'],  # Higher = more risk
    }
    
    risk_score = sum(risk_factors[factor] * weights[factor] for factor in weights)
    return min(1.0, max(0.0, risk_score))

def save_training_results(results: Dict[str, Any], filepath: str):
    """
    Save training results to JSON file.
    
    Args:
        results: Training results dictionary
        filepath: Path to save results
    """
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy_types(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def load_training_results(filepath: str) -> Dict[str, Any]:
    """
    Load training results from JSON file.
    
    Args:
        filepath: Path to saved results
        
    Returns:
        Training results dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)