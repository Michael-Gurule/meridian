"""
Run portfolio optimization example.

Demonstrates different optimization strategies and compares results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.data.storage import DataStorage
from src.optimization.optimizer import PortfolioOptimizer
from src.models.covariance import CovarianceEstimator
from src.models.returns import ReturnEstimator
from src.allocation.strategies import (
    EqualWeightStrategy,
    MinVarianceStrategy,
    RiskParityStrategy,
    MeanVarianceStrategy
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run optimization examples."""
    logger.info("=" * 80)
    logger.info("MERIDIAN - Portfolio Optimization Examples")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\nStep 1: Loading market data")
    storage = DataStorage()
    assets = storage.list_available_assets(data_type='processed')
    
    # Use subset for clarity
    major_assets = ['SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD']
    available = [a for a in major_assets if a in assets]
    
    logger.info(f"Using assets: {', '.join(available)}")
    
    data_dict = storage.load_batch(available, data_type='processed')
    prices = pd.DataFrame({ticker: data['close'] for ticker, data in data_dict.items()})
    returns = prices.pct_change().dropna()
    
    # Use recent data
    lookback = 252 * 2  # 2 years
    prices_recent = prices.iloc[-lookback:]
    returns_recent = returns.iloc[-lookback:]
    
    logger.info(f"Data: {len(returns_recent)} days, {len(available)} assets")
    
    # Estimate parameters
    logger.info("\nStep 2: Estimating parameters")
    
    cov_estimator = CovarianceEstimator(method='ledoit_wolf')
    return_estimator = ReturnEstimator(method='historical')
    
    cov_matrix = cov_estimator.estimate(returns_recent)
    expected_returns = return_estimator.estimate(returns_recent, annualize=False)
    
    logger.info(f"Expected annual returns:\n{(expected_returns * 252 * 100).round(2)}")
    
    # Strategy 1: Equal Weight
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY 1: Equal Weight (1/N)")
    logger.info("=" * 80)
    
    ew_strategy = EqualWeightStrategy()
    ew_weights = ew_strategy.allocate(prices_recent)
    
    ew_return = expected_returns.values @ ew_weights * 252
    ew_vol = np.sqrt(ew_weights @ cov_matrix.values @ ew_weights) * np.sqrt(252)
    ew_sharpe = ew_return / ew_vol
    
    logger.info(f"Weights: {dict(zip(available, ew_weights.round(3)))}")
    logger.info(f"Expected Return: {ew_return:.2%}")
    logger.info(f"Volatility: {ew_vol:.2%}")
    logger.info(f"Sharpe Ratio: {ew_sharpe:.2f}")
    
    # Strategy 2: Minimum Variance
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY 2: Minimum Variance")
    logger.info("=" * 80)
    
    mv_strategy = MinVarianceStrategy(cov_method='ledoit_wolf')
    mv_weights = mv_strategy.allocate(prices_recent)
    
    mv_return = expected_returns.values @ mv_weights * 252
    mv_vol = np.sqrt(mv_weights @ cov_matrix.values @ mv_weights) * np.sqrt(252)
    mv_sharpe = mv_return / mv_vol
    
    logger.info(f"Weights: {dict(zip(available, mv_weights.round(3)))}")
    logger.info(f"Expected Return: {mv_return:.2%}")
    logger.info(f"Volatility: {mv_vol:.2%}")
    logger.info(f"Sharpe Ratio: {mv_sharpe:.2f}")
    
    # Strategy 3: Risk Parity
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY 3: Risk Parity")
    logger.info("=" * 80)
    
    rp_strategy = RiskParityStrategy(cov_method='sample')
    rp_weights = rp_strategy.allocate(prices_recent)
    
    rp_return = expected_returns.values @ rp_weights * 252
    rp_vol = np.sqrt(rp_weights @ cov_matrix.values @ rp_weights) * np.sqrt(252)
    rp_sharpe = rp_return / rp_vol
    
    logger.info(f"Weights: {dict(zip(available, rp_weights.round(3)))}")
    logger.info(f"Expected Return: {rp_return:.2%}")
    logger.info(f"Volatility: {rp_vol:.2%}")
    logger.info(f"Sharpe Ratio: {rp_sharpe:.2f}")
    
    # Strategy 4: Mean-Variance (Markowitz)
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY 4: Mean-Variance (Sharpe Maximization)")
    logger.info("=" * 80)
    
    mvopt_strategy = MeanVarianceStrategy(
        cov_method='ledoit_wolf',
        return_method='historical',
        risk_aversion=1.0
    )
    mvopt_weights = mvopt_strategy.allocate(prices_recent)
    
    mvopt_return = expected_returns.values @ mvopt_weights * 252
    mvopt_vol = np.sqrt(mvopt_weights @ cov_matrix.values @ mvopt_weights) * np.sqrt(252)
    mvopt_sharpe = mvopt_return / mvopt_vol
    
    logger.info(f"Weights: {dict(zip(available, mvopt_weights.round(3)))}")
    logger.info(f"Expected Return: {mvopt_return:.2%}")
    logger.info(f"Volatility: {mvopt_vol:.2%}")
    logger.info(f"Sharpe Ratio: {mvopt_sharpe:.2f}")
    
    # Comparison
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY COMPARISON")
    logger.info("=" * 80)
    
    comparison = pd.DataFrame({
        'Strategy': ['Equal Weight', 'Min Variance', 'Risk Parity', 'Mean-Variance'],
        'Return': [ew_return, mv_return, rp_return, mvopt_return],
        'Volatility': [ew_vol, mv_vol, rp_vol, mvopt_vol],
        'Sharpe': [ew_sharpe, mv_sharpe, rp_sharpe, mvopt_sharpe]
    })
    
    comparison['Return'] = (comparison['Return'] * 100).round(2)
    comparison['Volatility'] = (comparison['Volatility'] * 100).round(2)
    comparison['Sharpe'] = comparison['Sharpe'].round(2)
    
    logger.info(f"\n{comparison.to_string(index=False)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Optimization examples complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()