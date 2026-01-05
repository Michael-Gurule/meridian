"""
Backtesting engine for portfolio strategies.

Runs walk-forward backtests with realistic execution and rebalancing.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Callable, List
from datetime import datetime

from src.optimization.optimizer import PortfolioOptimizer
from src.backtesting.execution import ExecutionSimulator
from src.backtesting.metrics import PerformanceMetrics
from src.models.covariance import CovarianceEstimator
from src.models.returns import ReturnEstimator
from src.models.regime import RegimeDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    Portfolio strategy backtesting engine.
    
    Runs walk-forward backtests with realistic constraints and execution.
    """
    
    def __init__(
        self,
        rebalance_frequency: str = 'M',
        lookback_window: int = 252,
        initial_capital: float = 1000000.0
    ):
        """
        Initialize backtest engine.
        
        Parameters
        ----------
        rebalance_frequency : str
            Rebalancing frequency ('D', 'W', 'M', 'Q')
        lookback_window : int
            Historical window for estimation (days)
        initial_capital : float
            Starting capital
        """
        self.rebalance_frequency = rebalance_frequency
        self.lookback_window = lookback_window
        self.initial_capital = initial_capital
        
        self.execution_sim = ExecutionSimulator()
        self.metrics_calc = PerformanceMetrics()
        
        logger.info(f"Initialized BacktestEngine: {rebalance_frequency} rebalancing")
    
    def run(
        self,
        prices: pd.DataFrame,
        optimizer: PortfolioOptimizer,
        cov_estimator: CovarianceEstimator,
        return_estimator: ReturnEstimator,
        regime_detector: Optional[RegimeDetector] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Run backtest.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data
        optimizer : PortfolioOptimizer
            Portfolio optimizer
        cov_estimator : CovarianceEstimator
            Covariance estimator
        return_estimator : ReturnEstimator
            Return estimator
        regime_detector : RegimeDetector, optional
            Regime detector for conditional optimization
        start_date : str, optional
            Backtest start date
        end_date : str, optional
            Backtest end date
        **kwargs : dict
            Additional parameters
        
        Returns
        -------
        Dict
            Backtest results
        """
        # Filter dates
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Rebalancing dates
        rebalance_dates = self._get_rebalance_dates(returns.index)
        
        logger.info(f"Starting backtest: {len(rebalance_dates)} rebalances")
        logger.info(f"Date range: {returns.index[0]} to {returns.index[-1]}")
        
        # Initialize tracking
        portfolio_values = []
        portfolio_returns = []
        weights_history = []
        turnover_history = []
        cost_history = []
        
        current_weights = None
        portfolio_value = self.initial_capital
        
        # Walk-forward backtest
        for i, rebal_date in enumerate(rebalance_dates):
            if rebal_date not in returns.index:
                continue
            
            # Historical data up to rebalance date
            hist_prices = prices.loc[:rebal_date].iloc[-self.lookback_window:]
            hist_returns = returns.loc[:rebal_date].iloc[-self.lookback_window:]
            
            if len(hist_returns) < self.lookback_window // 2:
                logger.warning(f"Insufficient history at {rebal_date}, skipping")
                continue
            
            # Estimate parameters
            try:
                cov_matrix = cov_estimator.estimate(hist_returns).values
                expected_returns = return_estimator.estimate(hist_returns, annualize=False).values
                
                # Regime-conditional optimization
                if regime_detector is not None:
                    # Fit regime detector
                    regime_detector.fit(hist_returns.iloc[:, 0])  # Use first asset as market proxy
                    
                    # Get current regime probabilities
                    regime_probs = regime_detector.get_current_regime_probability(
                        hist_returns.iloc[:, 0]
                    ).values
                    
                    # This is simplified; full implementation would estimate
                    # regime-specific parameters
                    logger.info(f"Regime probabilities: {regime_probs}")
                
                # Optimize
                if current_weights is None:
                    current_weights = np.zeros(len(hist_returns.columns))
                
                opt_result = optimizer.optimize(
                    expected_returns=expected_returns,
                    cov_matrix=cov_matrix,
                    current_weights=current_weights,
                    prices=hist_prices.iloc[-1].values,
                    portfolio_value=portfolio_value,
                    **kwargs
                )
                
                target_weights = opt_result['weights']
                
                # Execute rebalance
                execution = self.execution_sim.execute_rebalance(
                    current_weights=current_weights,
                    target_weights=target_weights,
                    prices=hist_prices.iloc[-1].values,
                    portfolio_value=portfolio_value
                )
                
                current_weights = execution['executed_weights']
                portfolio_value = execution['net_portfolio_value']
                
                # Track
                weights_history.append({
                    'date': rebal_date,
                    'weights': current_weights.copy()
                })
                
                turnover_history.append(opt_result['turnover'])
                cost_history.append(execution['cost_pct'])
                
                logger.info(
                    f"Rebalance {i+1}/{len(rebalance_dates)}: "
                    f"date={rebal_date.date()}, "
                    f"value=${portfolio_value:,.0f}, "
                    f"cost={execution['cost_pct']:.4%}"
                )
                
            except Exception as e:
                logger.error(f"Error at {rebal_date}: {str(e)}")
                continue
            
            # Daily returns until next rebalance
            if i < len(rebalance_dates) - 1:
                next_rebal = rebalance_dates[i + 1]
            else:
                next_rebal = returns.index[-1]
            
            period_returns = returns.loc[rebal_date:next_rebal]
            
            for date, daily_returns in period_returns.iterrows():
                if date == rebal_date:
                    continue
                
                # Portfolio return
                port_return = (current_weights * daily_returns.values).sum()
                portfolio_value *= (1 + port_return)
                
                portfolio_values.append({
                    'date': date,
                    'value': portfolio_value,
                    'weights': current_weights.copy()
                })
                
                portfolio_returns.append({
                    'date': date,
                    'return': port_return
                })
        
        # Compile results
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        returns_df = pd.DataFrame(portfolio_returns)
        returns_df.set_index('date', inplace=True)
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_all_metrics(
            returns=returns_df['return'],
            prices=portfolio_df['value']
        )
        
        results = {
            'portfolio_values': portfolio_df,
            'returns': returns_df,
            'weights_history': weights_history,
            'metrics': metrics,
            'turnover': np.mean(turnover_history) if turnover_history else 0,
            'avg_cost': np.mean(cost_history) if cost_history else 0,
            'n_rebalances': len(weights_history),
            'final_value': portfolio_value,
            'total_return': (portfolio_value / self.initial_capital) - 1
        }
        
        logger.info(f"Backtest complete: final value=${portfolio_value:,.0f}")
        logger.info(f"Total return: {results['total_return']:.2%}")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        
        return results
    
    def _get_rebalance_dates(self, date_index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        """
        Get rebalancing dates based on frequency.
        
        Parameters
        ----------
        date_index : pd.DatetimeIndex
            Full date range
        
        Returns
        -------
        List[pd.Timestamp]
            Rebalancing dates
        """
        if self.rebalance_frequency == 'D':
            return list(date_index)
        elif self.rebalance_frequency == 'W':
            return list(date_index[date_index.dayofweek == 0])  # Monday
        elif self.rebalance_frequency == 'M':
            return list(date_index.to_series().resample('ME').last().index)
        elif self.rebalance_frequency == 'Q':
            return list(date_index.to_series().resample('QE').last().index)
        else:
            raise ValueError(f"Unknown frequency: {self.rebalance_frequency}")
    
    def compare_strategies(
        self,
        prices: pd.DataFrame,
        strategies: Dict[str, PortfolioOptimizer],
        **kwargs
    ) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Historical prices
        strategies : Dict[str, PortfolioOptimizer]
            Dictionary of strategy name to optimizer
        **kwargs : dict
            Additional parameters
        
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        cov_estimator = CovarianceEstimator()
        return_estimator = ReturnEstimator()
        
        comparison_results = []
        
        for strategy_name, optimizer in strategies.items():
            logger.info(f"\nBacktesting strategy: {strategy_name}")
            
            result = self.run(
                prices=prices,
                optimizer=optimizer,
                cov_estimator=cov_estimator,
                return_estimator=return_estimator,
                **kwargs
            )
            
            metrics = result['metrics']
            metrics['strategy'] = strategy_name
            metrics['final_value'] = result['final_value']
            metrics['total_return'] = result['total_return']
            metrics['avg_turnover'] = result['turnover']
            metrics['avg_cost'] = result['avg_cost']
            
            comparison_results.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.set_index('strategy', inplace=True)
        
        return comparison_df