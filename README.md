# 💳 Credit Line Adjuster - Reinforcement Learning System

An intelligent **Reinforcement Learning (RL)** system that dynamically optimizes **credit line decisions** for financial institutions.  
The system learns optimal policies to **balance profitability and risk** by adjusting credit limits based on customer behaviors.

---

## 📘 Overview

The **Credit Line Adjuster** leverages **Q-learning** (Temporal Difference RL) to make adaptive credit limit recommendations for customers.  
It continuously learns from synthetic or real financial data, simulating credit risk and profitability in dynamic market conditions.

---

## 🏗️ Project Structure


CreditLiner/
├── 📁 venv/ # Virtual environment (gitignored)
├── 📄 interface.py # Main Streamlit interface
├── 📄 environment.py # RL Environment definition
├── 📄 agent.py # TD Agent implementation
├── 📄 training.py # Training logic and procedures
├── 📄 evaluation.py # Model evaluation framework
├── 📄 utils.py # Utility functions and helpers
├── 📄 requirements.txt # Python dependencies
├── 📄 README.md # Project documentation


---

## Core Features

### Reinforcement Learning Engine
- **TD Learning Agent** implementing **Q-learning**
- **State Space:** customer features (utilization, payment history, income stability, etc.)
- **Action Space:**
  - `0` → Decrease credit limit  
  - `1` → Maintain limit  
  - `2` → Increase limit
- **Reward Function:** Balances short-term profit with long-term portfolio health

### Data Management
- Synthetic customer data generation
- CSV dataset upload support
- Automated feature scaling & normalization

### Training Pipeline
- Configurable hyperparameters (learning rate, discount factor, exploration)
- Real-time progress visualization
- Performance & convergence tracking

### Evaluation Framework
- Profit vs. risk analysis
- Default rate tracking
- Historical performance reports

### Interactive Simulation (Streamlit UI)
- Real-time scenario testing
- Adjustable customer parameters:
  - Credit utilization
  - Payment history score
  - Income stability
  - Late payment count
  - Debt-to-income ratio
- Instant RL-based decision output

---

##  Installation & Setup

### Prerequisites
- Python **3.8+**
- `pip` package manager

### Setup Steps
```bash
# Clone the repository
git clone <repository-url>
cd CreditLiner

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run interface.py
Technical Architecture
RL Components
state = [utilization, payment_history, income_stability, 
         late_payments_normalized, debt_ratio]


Actions:

0: Decrease credit limit

1: Maintain limit

2: Increase limit

Reward:

Positive for profitable, low-risk actions

Negative for defaults or excessive risk

⚙️ Algorithms

Q-Learning (Off-policy TD Control)

ε-Greedy Exploration for exploration-exploitation balance

Optional Experience Replay for stability

Performance Metrics
Category	Metrics
Training	Episode rewards, ε decay, convergence
Evaluation	Average profit, risk score, default rate
Business	Portfolio performance, customer satisfaction
🔧 Configuration Options
Hyperparameters
Parameter	Range	Description
Learning Rate (α)	0.01 – 0.5	Step size for updates
Discount Factor (γ)	0.5 – 0.99	Future reward weighting
Exploration Rate (ε)	0.1 – 1.0	Random action probability
Decay Rate	0.99 – 0.999	Exploration decay factor
Training Episodes	100 – 10,000	Total training iterations
Data Options

Synthetic generator

Custom CSV upload (required columns: customer features, performance, risk indicators)


