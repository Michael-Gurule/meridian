"""
Demonstrate regime detection on market data.

This script runs regime detection analysis and generates a report.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from src.data.storage import DataStorage
from src.models.regime import RegimeDetector
from src.models.volatility import VolatilityModel
from src.models.covariance import CovarianceEstimator
from src.models.returns import ReturnEstimator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Run regime detection analysis."""
    logger.info("=" * 80)
    logger.info("MERIDIAN - Regime Detection Analysis")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\nStep 1: Loading market data")
    storage = DataStorage()
    assets = storage.list_available_assets(data_type='processed')
    
    data_dict = storage.load_batch(assets, data_type='processed')
    prices = pd.DataFrame({ticker: data['close'] for ticker, data in data_dict.items()})
    returns = prices.pct_change().dropna()
    
    logger.info(f"Loaded {len(assets)} assets")
    logger.info(f"Date range: {returns.index.min()} to {returns.index.max()}")
    
    # Focus on market (SPY)
    market_returns = returns['SPY']
    logger.info(f"\nAnalyzing SPY returns: {len(market_returns)} observations")
    
    # Calculate volatility
    logger.info("\nStep 2: Estimating volatility")
    vol_model = VolatilityModel(method='ewm', ewm_halflife=30)
    volatility = vol_model.estimate(market_returns, annualize=True)
    
    logger.info(f"Current volatility: {volatility.iloc[-1]:.2f}%")
    logger.info(f"Average volatility: {volatility.mean():.2f}%")
    
    # Detect regimes
    logger.info("\nStep 3: Detecting market regimes")
    detector = RegimeDetector(n_regimes=2, n_iter=100, random_state=42)
    detector.fit(market_returns, volatility / 100)
    
    regimes = detector.predict_regimes(market_returns, volatility / 100)
    probs = detector.predict_probabilities(market_returns, volatility / 100)
    
    logger.info("Regime detection complete")
    
    # Regime statistics
    logger.info("\n" + "=" * 80)
    logger.info("REGIME STATISTICS")
    logger.info("=" * 80)
    
    stats = detector.get_regime_stats()
    for regime_idx in range(2):
        regime_stats = stats.loc[regime_idx]
        logger.info(f"\nRegime {regime_idx}:")
        logger.info(f"  Frequency: {regime_stats['frequency']:.1%}")
        logger.info(f"  Mean Return: {regime_stats['mean_return']:.2f}%")
        logger.info(f"  Volatility: {regime_stats['volatility']:.2f}%")
        logger.info(f"  Sharpe Ratio: {regime_stats['sharpe']:.2f}")
        logger.info(f"  Avg Duration: {regime_stats['avg_duration']:.1f} days")
    
    # Current state
    logger.info("\n" + "=" * 80)
    logger.info("CURRENT MARKET STATE")
    logger.info("=" * 80)
    
    current_prob = detector.get_current_regime_probability(market_returns, volatility / 100)
    for regime in range(2):
        prob = current_prob[f'regime_{regime}']
        logger.info(f"Regime {regime} probability: {prob:.1%}")
    
    most_likely = current_prob.idxmax()
    logger.info(f"\nMost likely current regime: {most_likely}")
    
    # Regime-conditional covariance
    logger.info("\n" + "=" * 80)
    logger.info("REGIME-CONDITIONAL ANALYSIS")
    logger.info("=" * 80)
    
    major_assets = ['SPY', 'QQQ', 'TLT', 'GLD']
    available_major = [a for a in major_assets if a in returns.columns]
    
    if len(available_major) >= 3:
        cov_estimator = CovarianceEstimator(method='sample')
        
        for regime in range(2):
            mask = regimes == regime
            # Align returns with regimes index
            aligned_returns = returns[available_major].loc[regimes.index]
            regime_returns = aligned_returns[mask]
            
            regime_cov = cov_estimator.estimate_correlation(regime_returns)
            avg_corr = (regime_cov.sum().sum() - len(regime_cov)) / (len(regime_cov) ** 2 - len(regime_cov))
            
            logger.info(f"\nRegime {regime} - Average correlation: {avg_corr:.3f}")
    
    # Expected returns by regime
    logger.info("\n" + "=" * 80)
    logger.info("EXPECTED RETURNS BY REGIME")
    logger.info("=" * 80)
    
    return_estimator = ReturnEstimator(method='historical')
    
    for regime in range(2):
        mask = regimes == regime
        # Align returns with regimes index
        aligned_returns = returns[available_major].loc[regimes.index]
        regime_returns_data = aligned_returns[mask]
        
        expected = return_estimator.estimate(regime_returns_data, annualize=True)
        
        logger.info(f"\nRegime {regime} Expected Returns (Annualized):")
        for asset in available_major:
            logger.info(f"  {asset}: {expected[asset]:.2f}%")
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()