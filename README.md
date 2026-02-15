<br>

<p align="center">
  <img width="500" alt="MERIDIAN" src="https://github.com/user-attachments/assets/98acabc0-8f12-4cb4-9085-30cd78c7d521" /> 
<p align="center">
  <strong>Multi-Asset Portfolio Optimization System</strong>
  <br>
  <br>
  <strong>Production-grade portfolio optimization with regime detection and interactive decision support</strong>
</p>  
<br>


---

##  Project Overview

MERIDIAN is a comprehensive investment portfolio optimization system that demonstrates institutional-quality quantitative finance capabilities. The system combines multi-source market data acquisition, regime-conditional optimization, realistic transaction cost modeling, and an interactive decision support dashboard.

**Key Innovation:** Unlike academic portfolio projects that optimize once and show a backtest, MERIDIAN implements production-level thinking with regime detection, transaction costs, turnover constraints, and explainable recommendations through an interactive dashboard.

---

##  Features

### Market Data Acquisition 
- [X] **25 assets** across equities, fixed income, commodities, alternatives
- [X] **Multi-source integration**: Yahoo Finance, NASA, USGS, NOAA, FEMA
- [X] **Robust validation**: Data quality checks, outlier detection
- [X] **Efficient storage**: Parquet format with 60% compression
- [X] **Batch downloads**: API rate limit handling

### Statistical Modeling 
- [X]  **Covariance estimation**: Sample, Ledoit-Wolf, EWMA, constant correlation
- [X]  **GARCH(1,1)**: Volatility clustering and forecasting
- [X]  **Regime detection**: 2-3 state Hidden Markov Models
- [X]  **Return forecasting**: Historical, momentum, factor models

### Portfolio Optimization 
- [X]  **Multiple objectives**: Mean-variance, minimum variance, risk parity, max diversification
- [X]  **Realistic constraints**: Long-only, position bounds, turnover limits, sector constraints
- [X]  **Transaction costs**: Bid-ask spread + market impact + commissions
- [X]  **Regime-conditional**: Adapts allocations to market state
- [X]  **Backtesting**: Walk-forward validation with realistic execution

### Interactive Dashboard 
- [X]  **Portfolio monitoring**: Real-time state and performance tracking
- [X]  **Optimization interface**: Interactive controls for strategy parameters
- [X]  **Regime dashboard**: Current market state and regime probabilities
- [X]  **Performance analytics**: 20+ metrics with benchmark comparison
- [X]  **Scenario analysis**: Stress tests, custom shocks, historical events, Monte Carlo
- [X]  **Recommendations**: Cost-benefit analysis with exportable trade lists


### Interactive Dashboard
<br>

<p align="center">
  <img align="center" width="800" alt="Overview" src="https://github.com/user-attachments/assets/816ed176-cf6f-43f0-b260-ef5df2d92257" />
  <br>
  <sub><b>Dashboard Overview:</b>Current Portfolio Distrubution</sub>
</p> 
<br>
<p align="center">
  <img width="800" alt="Perform Analytics" src="https://github.com/user-attachments/assets/e5ebf9d3-3670-4aa6-9fda-92a625bb2139" />
  <br>
  <sub><b>Performance analytics:</b>20+ metrics with benchmark comparison</sub>
</p>
<br>

<p align="center">
  <img width="800" alt="Optimization" src="https://github.com/user-attachments/assets/74be80e0-5b7a-4b6c-bbdb-094dfe32ee0e" />
  <br>
  <sub><b>Optimization interface:</b>Interactive controls for strategy parameters</sub>
</p>
<br>

<p align="center">
  <img width="800"  alt="Scenerio Analysis" src="https://github.com/user-attachments/assets/b417f676-9334-4381-b880-36e845dfc23f" />
  <br>
  <sub><b>Scenario analysis:</b>Stress tests, custom shocks, historical events, Monte Carlo</sub>
</p>
<br>
<br>

```
bash

Clone and setup
git clone https://github.com/michael-gurule/meridian.git
cd meridian
pip install -r requirements.txt

Download sample data
python scripts/download_sample_data.py

Run dashboard
./scripts/run_dashboard.sh
```
Access dashboard at `http://localhost:8501'

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

**Available notebooks:**
- `01_data_exploration.ipynb`: Data quality and statistics
- `02_regime_analysis.ipynb`: Regime detection demonstration
- `03_optimization_analysis.ipynb`: Optimization strategies

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
<br>

<h1 align="center">LET'S CONNECT!</h1>

<p align="center">
This project demonstrates production-grade ML engineering capabilities including distributed training infrastructure, experiment management, and systematic research methodology. All code and documentation available for technical review.
</p>


<h3 align="center">Michael Gurule</h3>

<p align="center">
  <strong>Data Science | ML Engineering</strong>
</p>
<br>

  
<div align="center">
  <a href="mailto:michaelgurule1164@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"></a>
  
  <a href="michaelgurule.com">
    <img src="https://custom-icon-badges.demolab.com/badge/MICHAELGURULE.COM-150458?style=for-the-badge&logo=browser&logoColor=white"></a>
  
  <a href="www.linkedin.com/in/michael-gurule-447aa2134">
    <img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin-white&logoColor=fff"></a>
  
  <a href="https://medium.com/@michaelgurule1164">
    <img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"></a>    
</div>
<br>

---

<p align="center"> 
<img  width="450" alt="Designed By" src="https://github.com/user-attachments/assets/12ddff9c-b9b6-4e69-ace0-5cbc94f1a3ad"> 
</p>
<p align="center">
  <sub> Demonstrating institutional-quality quantitative finance capabilities through production-grade portfolio optimization </sub>
</p>










  
