"""
Expected return estimation for portfolio optimization.

Implements multiple approaches:
- Historical mean
- Momentum-based signals
- Simple factor models
- Ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from sklearn.linear_model import LinearRegression

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReturnEstimator:
    """
    Estimates expected returns using multiple methods.
    
    Provides historical, momentum, and factor-based return forecasts.
    """
    
    def __init__(
        self,
        method: str = 'historical',
        lookback: int = 252,
        momentum_window: int = 60
    ):
        """
        Initialize return estimator.
        
        Parameters
        ----------
        method : str
            Estimation method ('historical', 'momentum', 'factor', 'ensemble')
        lookback : int
            Lookback period for historical mean
        momentum_window : int
            Window for momentum calculation
        """
        self.method = method
        self.lookback = lookback
        self.momentum_window = momentum_window
        
        logger.info(f"Initialized ReturnEstimator: method={method}")
    
    def estimate(
        self,
        returns: pd.DataFrame,
        market_returns: Optional[pd.Series] = None,
        annualize: bool = True
    ) -> pd.Series:
        """
        Estimate expected returns for assets.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical return series for assets
        market_returns : pd.Series, optional
            Market return series (for factor models)
        annualize : bool
            Return annualized estimates
        
        Returns
        -------
        pd.Series
            Expected return estimates
        """
        if self.method == 'historical':
            expected_returns = self._historical_mean(returns)
        elif self.method == 'momentum':
            expected_returns = self._momentum_based(returns)
        elif self.method == 'factor':
            if market_returns is None:
                logger.warning("Market returns not provided, using historical mean")
                expected_returns = self._historical_mean(returns)
            else:
                expected_returns = self._factor_model(returns, market_returns)
        elif self.method == 'ensemble':
            expected_returns = self._ensemble_estimate(returns, market_returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if annualize:
            expected_returns = expected_returns * 252
        
        return expected_returns
    
    def _historical_mean(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate historical mean returns.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series
        
        Returns
        -------
        pd.Series
            Historical mean returns
        """
        if len(returns) > self.lookback:
            recent_returns = returns.iloc[-self.lookback:]
        else:
            recent_returns = returns
        
        return recent_returns.mean()
    
    def _momentum_based(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum-based expected returns.
        
        Recent strong performers are expected to continue outperforming.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series
        
        Returns
        -------
        pd.Series
            Momentum-based expected returns
        """
        if len(returns) < self.momentum_window:
            logger.warning(f"Insufficient data for momentum ({len(returns)} < {self.momentum_window})")
            return self._historical_mean(returns)
        
        # Calculate cumulative returns over momentum window
        momentum_returns = returns.iloc[-self.momentum_window:].add(1).prod() - 1
        
        # Normalize to daily returns
        daily_momentum = (1 + momentum_returns) ** (1 / self.momentum_window) - 1
        
        return daily_momentum
    
    def _factor_model(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series
    ) -> pd.Series:
        """
        Estimate returns using single-factor (CAPM) model.
        
        E[R_i] = R_f + beta_i * (E[R_m] - R_f)
        Simplified: E[R_i] = alpha_i + beta_i * E[R_m]
        
        Parameters
        ----------
        returns : pd.DataFrame
            Asset return series
        market_returns : pd.Series
            Market return series
        
        Returns
        -------
        pd.Series
            Factor-based expected returns
        """
        # Align data
        aligned_returns, aligned_market = returns.align(market_returns, join='inner', axis=0)
        
        expected_returns = {}
        market_mean = aligned_market.mean()
        
        for asset in aligned_returns.columns:
            try:
                # Estimate beta via regression
                X = aligned_market.values.reshape(-1, 1)
                y = aligned_returns[asset].values
                
                # Remove NaN
                mask = ~(np.isnan(X.flatten()) | np.isnan(y))
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) < 30:
                    logger.warning(f"Insufficient data for {asset}, using historical mean")
                    expected_returns[asset] = aligned_returns[asset].mean()
                    continue
                
                model = LinearRegression()
                model.fit(X_clean, y_clean)
                
                alpha = model.intercept_
                beta = model.coef_[0]
                
                # Expected return = alpha + beta * E[market]
                expected_returns[asset] = alpha + beta * market_mean
                
            except Exception as e:
                logger.error(f"Factor model failed for {asset}: {str(e)}")
                expected_returns[asset] = aligned_returns[asset].mean()
        
        return pd.Series(expected_returns)
    
    def _ensemble_estimate(
        self,
        returns: pd.DataFrame,
        market_returns: Optional[pd.Series]
    ) -> pd.Series:
        """
        Ensemble of multiple estimation methods.
        
        Combines historical, momentum, and factor estimates with equal weights.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series
        market_returns : pd.Series, optional
            Market returns
        
        Returns
        -------
        pd.Series
            Ensemble expected returns
        """
        estimates = []
        weights = []
        
        # Historical mean
        hist_est = self._historical_mean(returns)
        estimates.append(hist_est)
        weights.append(0.4)
        
        # Momentum
        momentum_est = self._momentum_based(returns)
        estimates.append(momentum_est)
        weights.append(0.3)
        
        # Factor model (if available)
        if market_returns is not None:
            factor_est = self._factor_model(returns, market_returns)
            estimates.append(factor_est)
            weights.append(0.3)
        else:
            # Reweight without factor model
            weights = [0.6, 0.4]
        
        # Weighted average
        ensemble = sum(est * w for est, w in zip(estimates, weights))
        
        return ensemble
    
    def estimate_with_uncertainty(
        self,
        returns: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> pd.DataFrame:
        """
        Estimate expected returns with confidence intervals.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series
        confidence_level : float
            Confidence level for intervals
        
        Returns
        -------
        pd.DataFrame
            Expected returns with lower and upper bounds
        """
        from scipy import stats
        
        point_estimates = self.estimate(returns, annualize=False)
        
        results = []
        
        for asset in returns.columns:
            asset_returns = returns[asset].dropna()
            
            if len(asset_returns) < 30:
                logger.warning(f"Insufficient data for uncertainty estimation: {asset}")
                results.append({
                    'asset': asset,
                    'expected_return': point_estimates[asset],
                    'lower_bound': point_estimates[asset],
                    'upper_bound': point_estimates[asset]
                })
                continue
            
            mean = asset_returns.mean()
            std_error = asset_returns.std() / np.sqrt(len(asset_returns))
            
            # t-distribution confidence interval
            dof = len(asset_returns) - 1
            t_critical = stats.t.ppf((1 + confidence_level) / 2, dof)
            
            margin = t_critical * std_error
            
            results.append({
                'asset': asset,
                'expected_return': mean,
                'lower_bound': mean - margin,
                'upper_bound': mean + margin
            })
        
        return pd.DataFrame(results)
    
    def rank_assets(
        self,
        returns: pd.DataFrame,
        method: str = 'sharpe'
    ) -> pd.Series:
        """
        Rank assets by expected performance.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series
        method : str
            Ranking method ('return', 'sharpe', 'momentum')
        
        Returns
        -------
        pd.Series
            Asset ranks (1 = best)
        """
        if method == 'return':
            scores = self.estimate(returns, annualize=True)
        elif method == 'sharpe':
            expected_returns = self.estimate(returns, annualize=True)
            volatilities = returns.std() * np.sqrt(252)
            scores = expected_returns / volatilities
        elif method == 'momentum':
            scores = self._momentum_based(returns) * 252
        else:
            raise ValueError(f"Unknown ranking method: {method}")
        
        # Rank (higher score = better rank)
        ranks = scores.rank(ascending=False)
        
        return ranks