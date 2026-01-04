<br>

<p align="center">
  <img width="600" alt="MERIDIAN" src="https://github.com/user-attachments/assets/98acabc0-8f12-4cb4-9085-30cd78c7d521" /> 
<p align="center">
  <strong>Multi-Asset Portfolio Optimization System</strong>
  <br>
  <br>
  <strong>Production-grade portfolio optimization with regime detection and interactive decision support</strong>
</p>  
<br>



[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  Project Overview

MERIDIAN is a comprehensive investment portfolio optimization system that demonstrates institutional-quality quantitative finance capabilities. The system combines multi-source market data acquisition, regime-conditional optimization, realistic transaction cost modeling, and an interactive decision support dashboard.

**Key Innovation:** Unlike academic portfolio projects that optimize once and show a backtest, MERIDIAN implements production-level thinking with regime detection, transaction costs, turnover constraints, and explainable recommendations through an interactive dashboard.

### Quick Demo
```bash
# Clone and setup
git clone https://github.com/michael-gurule/meridian.git
cd meridian
pip install -r requirements.txt

# Download sample data
python scripts/download_sample_data.py

# Run dashboard
./scripts/run_dashboard.sh
```

Access dashboard at `http://localhost:8501`

---

##  System Architecture
```
MERIDIAN Portfolio Optimization System
│
├── Phase 1: Market Data Infrastructure
│   ├── Multi-source data acquisition (25 assets)
│   ├── Robust validation and error handling
│   └── Efficient Parquet-based storage
│
├── Phase 2: Risk & Return Modeling
│   ├── Covariance estimation (4 methods)
│   ├── GARCH volatility forecasting
│   ├── Hidden Markov regime detection
│   └── Expected return estimation
│
├── Phase 3: Portfolio Optimization Engine
│   ├── Multiple objectives (Sharpe, min variance, risk parity)
│   ├── Realistic constraints (turnover, position limits)
│   ├── Transaction cost modeling
│   └── Walk-forward backtesting
│
└── Phase 4: Interactive Decision Dashboard
    ├── Real-time optimization interface
    ├── Regime monitoring and alerts
    ├── Scenario analysis (stress tests, Monte Carlo)
    └── Actionable rebalancing recommendations
```

---

##  Features

### Market Data Acquisition 
[X] **25 assets** across equities, fixed income, commodities, alternatives
[X] **Multi-source integration**: Yahoo Finance, NASA, USGS, NOAA, FEMA
[X] **Robust validation**: Data quality checks, outlier detection
[X] **Efficient storage**: Parquet format with 60% compression
[X] **Batch downloads**: API rate limit handling

### Statistical Modeling 
[X]  **Covariance estimation**: Sample, Ledoit-Wolf, EWMA, constant correlation
[X]  **GARCH(1,1)**: Volatility clustering and forecasting
[X]  **Regime detection**: 2-3 state Hidden Markov Models
[X]  **Return forecasting**: Historical, momentum, factor models

### Portfolio Optimization 
[X]  **Multiple objectives**: Mean-variance, minimum variance, risk parity, max diversification
[X]  **Realistic constraints**: Long-only, position bounds, turnover limits, sector constraints
[X]  **Transaction costs**: Bid-ask spread + market impact + commissions
[X]  **Regime-conditional**: Adapts allocations to market state
[X]  **Backtesting**: Walk-forward validation with realistic execution

### Interactive Dashboard 
[X]  **Portfolio monitoring**: Real-time state and performance tracking
[X]  **Optimization interface**: Interactive controls for strategy parameters
[X]  **Regime dashboard**: Current market state and regime probabilities
[X]  **Performance analytics**: 20+ metrics with benchmark comparison
[X]  **Scenario analysis**: Stress tests, custom shocks, historical events, Monte Carlo
[X]  **Recommendations**: Cost-benefit analysis with exportable trade lists

---

##  Project Structure
```
meridian/
├── src/
│   ├── data/                  # Data acquisition and storage
│   ├── models/                # Statistical models (regime, volatility, covariance)
│   ├── optimization/          # Portfolio optimization engine
│   ├── backtesting/          # Backtesting framework
│   ├── allocation/           # Pre-built strategies
│   ├── dashboard/            # Streamlit dashboard
│   └── utils/                # Utilities and helpers
├── data/
│   ├── raw/                  # Raw downloaded data
│   └── processed/            # Processed Parquet files
├── tests/                    # Automated tests
├── notebooks/                # Analysis notebooks
├── scripts/                  # Executable scripts
├── docs/                     # Documentation
├── config/                   # Configuration files
└── requirements.txt          # Dependencies
```

---

##  Installation

### Prerequisites

- Python 3.10+
- pip package manager
- 2GB disk space for data
- Internet connection for data download

### Setup
```bash
# Clone repository
git clone https://github.com/michael-gurule/meridian.git
cd meridian

# Create virtual environment (recommended)
python -m venv venv 
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download sample data
python scripts/download_sample_data.py
```

---

##  Usage

### Dashboard (Recommended)

**Start interactive dashboard:**
```bash
./scripts/run_dashboard.sh
```

Or manually:
```bash
streamlit run src/dashboard/app.py
```

Access at `http://localhost:8501`

### Command Line Examples

**Run optimization example:**
```bash
python scripts/run_optimization_example.py
```

**Run regime detection:**
```bash
python scripts/run_regime_detection.py
```

**Run backtest:**
```bash
python scripts/run_backtest.py
```

### Jupyter Notebooks
```bash
jupyter lab notebooks/
```

**Available notebooks:**
- `01_data_exploration.ipynb`: Data quality and statistics
- `02_regime_analysis.ipynb`: Regime detection demonstration
- `03_optimization_analysis.ipynb`: Optimization strategies

### Python API
```python
from src.data.storage import DataStorage
from src.optimization.optimizer import PortfolioOptimizer
from src.models.covariance import CovarianceEstimator

# Load data
storage = DataStorage()
prices = storage.load('SPY', data_type='processed')

# Optimize portfolio
optimizer = PortfolioOptimizer(objective_type='mean_variance')
result = optimizer.optimize(expected_returns, cov_matrix)

print(f"Optimal weights: {result['weights']}")
print(f"Expected Sharpe: {result['sharpe_ratio']:.2f}")
```

---

## Example Results

### Portfolio Optimization

**Mean-Variance Strategy (2-year backtest):**
- Annual Return: 9.3%
- Volatility: 16.1%
- Sharpe Ratio: 0.58
- Max Drawdown: -19.7%
- Transaction Costs: 15 bps/year

### Regime Detection

**2-State Model (SPY):**
- **Regime 0 (Low Volatility)**: 68% frequency, 10.2% return, 12.5% vol
- **Regime 1 (High Volatility)**: 32% frequency, 1.8% return, 28.7% vol

**Impact:** Regime-conditional optimization reduces drawdown by 20-30% vs. static allocation.

---

## Technical Highlights

### Advanced Methodologies

**Regime Detection:** 
- Hidden Markov Models with Baum-Welch training
- Viterbi algorithm for state sequence
- Forward-backward for probabilities

**Optimization:**
- Convex programming with CVXPY
- Quadratic objective functions
- Linear and quadratic constraints

**Transaction Costs:**
- Bid-ask spread (1-10 bps)
- Market impact (square-root law)
- Commission fees

**Backtesting:**
- Walk-forward validation
- Out-of-sample testing
- Realistic execution simulation

### Performance

- **Optimization**: <1 second for 25 assets
- **Backtest**: 5-10 seconds per year
- **Dashboard load**: <3 seconds
- **Data download**: 2-5 minutes for 25 assets

---

## Documentation

Comprehensive documentation in `docs/` directory:

- **Phase 1 Methodology**: Data acquisition and storage
- **Phase 2 Methodology**: Statistical modeling
- **Phase 3 Methodology**: Portfolio optimization
- **Phase 4 Methodology**: Interactive dashboard


---

## Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_optimization/test_optimizer.py
```

**Test Coverage:**
- Data acquisition: 85%
- Statistical models: 80%
- Optimization: 90%
- Backtesting: 85%

---

## Connection to Professional Experience

This project mirrors systems built during 8+ years of Fortune 500 consulting:

### Chemical Manufacturing Optimization ($29M Impact)

| MERIDIAN Component           | Professional System                        |
| ---------------------------- | ------------------------------------------ |
| Multi-objective optimization | Production + inventory + cost optimization |
| Turnover constraints         | Changeover costs, minimum run lengths      |
| Transaction costs            | Freight costs, supplier MOQs               |
| Regime detection             | Supply disruption scenarios                |
| Walk-forward testing         | Rolling quarterly forecasts                |
| Interactive dashboard        | Executive planning tools                   |

**Key Parallel:** Both systems optimize operational decisions under constraints with transaction costs, balancing short-term actions against long-term goals.

### COVID Risk Platform (Fortune 500 Financial Services)

| MERIDIAN Component            | Risk Platform                |
| ----------------------------- | ---------------------------- |
| Multi-source data integration | 50+ government API sources   |
| Real-time updates             | Daily risk intelligence      |
| Regime monitoring             | Regulatory scenario tracking |
| Automated reporting           | Portfolio risk dashboards    |

---

## License

MIT License - see LICENSE file for details.

---
## Contributing

This is a portfolio project. For questions or collaboration:

**Michael Gurule**

- [![Email Me](https://img.shields.io/badge/EMAIL-8A2BE2)](michaelgurule1164@gmail.com)
- [![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff)](www.linkedin.com/in/michael-j-gurule-447aa2134)
- [![Medium](https://img.shields.io/badge/Medium-%23000000.svg?logo=medium&logoColor=white)](https://medium.com/@michaelgurule1164)

---



<p align="center">
  <img width="250" alt="MERIDIAN" src="https://github.com/user-attachments/assets/98acabc0-8f12-4cb4-9085-30cd78c7d521" /> 
  <br>
  <sub> Built by Michael Gurule </sub>  

<p align="center">
  <sub> Demonstrating institutional-quality quantitative finance capabilities through production-grade portfolio optimization </sub>
</p>

  
