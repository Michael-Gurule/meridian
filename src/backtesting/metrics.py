"""
Performance metrics for portfolio backtesting.

Calculates comprehensive performance statistics.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict

from src.utils.statistics import (
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_var,
    calculate_cvar,
    calculate_calmar_ratio
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PerformanceMetrics:
    """
    Calculates portfolio performance metrics.
    
    Provides comprehensive statistics for backtest evaluation.
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Initialize performance metrics calculator.
        
        Parameters
        ----------
        risk_free_rate : float
            Annual risk-free rate
        """
        self.risk_free_rate = risk_free_rate
        
        logger.info("Initialized PerformanceMetrics")
    
    def calculate_all_metrics(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns
        prices : pd.Series, optional
            Price series (for drawdown)
        benchmark_returns : pd.Series, optional
            Benchmark returns
        
        Returns
        -------
        Dict
            Performance metrics
        """
        if prices is None:
            prices = (1 + returns).cumprod()
        
        # Annualization factor
        trading_days = 252
        
        # Return metrics
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        ann_return = (1 + total_return) ** (trading_days / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(trading_days)
        downside_vol = returns[returns < 0].std() * np.sqrt(trading_days)
        
        # Risk-adjusted returns
        sharpe = (ann_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        sortino = calculate_sortino_ratio(returns, self.risk_free_rate)
        
        # Drawdown
        max_dd = calculate_max_drawdown(prices)
        calmar = calculate_calmar_ratio(returns, prices)
        
        # Risk measures
        var_95 = calculate_var(returns, 0.95)
        cvar_95 = calculate_cvar(returns, 0.95)
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Profit factor
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else np.inf
        
        metrics = {
            'total_return': total_return,
            'annual_return': ann_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'avg_daily_return': returns.mean(),
            'downside_volatility': downside_vol
        }
        
        # Benchmark comparison
        if benchmark_returns is not None:
            aligned_returns, aligned_bench = returns.align(benchmark_returns, join='inner')
            
            excess_returns = aligned_returns - aligned_bench
            tracking_error = excess_returns.std() * np.sqrt(trading_days)
            information_ratio = (excess_returns.mean() * trading_days) / tracking_error if tracking_error > 0 else 0
            
            # Beta
            covariance = np.cov(aligned_returns, aligned_bench)[0, 1]
            benchmark_var = aligned_bench.var()
            beta = covariance / benchmark_var if benchmark_var > 0 else 1.0
            
            # Alpha
            benchmark_return = aligned_bench.mean() * trading_days
            alpha = ann_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
            
            metrics['tracking_error'] = tracking_error
            metrics['information_ratio'] = information_ratio
            metrics['beta'] = beta
            metrics['alpha'] = alpha
        
        return metrics
    
    def performance_summary(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Generate performance summary table.
        
        Parameters
        ----------
        returns : pd.Series
            Strategy returns
        prices : pd.Series, optional
            Price series
        benchmark_returns : pd.Series, optional
            Benchmark returns
        
        Returns
        -------
        pd.DataFrame
            Performance summary
        """
        metrics = self.calculate_all_metrics(returns, prices, benchmark_returns)
        
        summary = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        
        return summary