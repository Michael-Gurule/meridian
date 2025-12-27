"""
Covariance and correlation estimation for portfolio optimization.

Implements multiple estimation methods:
- Sample covariance (rolling window)
- Exponentially weighted covariance
- Ledoit-Wolf shrinkage
- Constant correlation model
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.covariance import LedoitWolf

from src.utils.logger import get_logger
from src.utils.statistics import ensure_positive_definite

logger = get_logger(__name__)


class CovarianceEstimator:
    """
    Estimates covariance matrices using multiple methods.
    
    Provides rolling, exponentially weighted, and shrinkage estimators
    for robust covariance forecasting.
    """
    
    def __init__(
        self,
        method: str = 'sample',
        window: int = 252,
        min_periods: int = 60,
        ewm_halflife: int = 60
    ):
        """
        Initialize covariance estimator.
        
        Parameters
        ----------
        method : str
            Estimation method ('sample', 'ewm', 'ledoit_wolf', 'constant_correlation')
        window : int
            Rolling window size for sample covariance
        min_periods : int
            Minimum periods required for estimation
        ewm_halflife : int
            Halflife for exponentially weighted method
        """
        self.method = method
        self.window = window
        self.min_periods = min_periods
        self.ewm_halflife = ewm_halflife
        
        logger.info(f"Initialized CovarianceEstimator: method={method}, window={window}")
    
    def estimate(
        self,
        returns: pd.DataFrame,
        ensure_pd: bool = True
    ) -> pd.DataFrame:
        """
        Estimate covariance matrix.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series for assets
        ensure_pd : bool
            Ensure positive definiteness
        
        Returns
        -------
        pd.DataFrame
            Covariance matrix
        """
        if self.method == 'sample':
            cov = self._sample_covariance(returns)
        elif self.method == 'ewm':
            cov = self._ewm_covariance(returns)
        elif self.method == 'ledoit_wolf':
            cov = self._ledoit_wolf_covariance(returns)
        elif self.method == 'constant_correlation':
            cov = self._constant_correlation_covariance(returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if ensure_pd:
            cov_values = ensure_positive_definite(cov.values)
            cov = pd.DataFrame(cov_values, index=cov.index, columns=cov.columns)
        
        return cov
    
    def _sample_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sample covariance matrix.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series
        
        Returns
        -------
        pd.DataFrame
            Sample covariance matrix
        """
        return returns.cov()
    
    def _ewm_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate exponentially weighted covariance matrix.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series
        
        Returns
        -------
        pd.DataFrame
            EWMA covariance matrix
        """
        return returns.ewm(halflife=self.ewm_halflife, adjust=False).cov().iloc[-len(returns.columns):]
    
    def _ledoit_wolf_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ledoit-Wolf shrinkage covariance.
        
        Shrinks sample covariance toward structured estimator to reduce
        estimation error.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series
        
        Returns
        -------
        pd.DataFrame
            Shrinkage covariance matrix
        """
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns.dropna()).covariance_
        
        return pd.DataFrame(
            cov_matrix,
            index=returns.columns,
            columns=returns.columns
        )
    
    def _constant_correlation_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate constant correlation covariance matrix.
        
        Assumes all pairwise correlations are equal to average correlation.
        Useful for large portfolios with unstable correlation estimates.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series
        
        Returns
        -------
        pd.DataFrame
            Constant correlation covariance matrix
        """
        # Calculate sample correlation matrix
        corr = returns.corr()
        
        # Average correlation (excluding diagonal)
        avg_corr = (corr.sum().sum() - len(corr)) / (len(corr) ** 2 - len(corr))
        
        # Create constant correlation matrix
        const_corr = pd.DataFrame(
            avg_corr,
            index=corr.index,
            columns=corr.columns
        )
        np.fill_diagonal(const_corr.values, 1.0)
        
        # Convert to covariance
        std = returns.std()
        cov = const_corr.multiply(std, axis=0).multiply(std, axis=1)
        
        return cov
    
    def rolling_covariance(
        self,
        returns: pd.DataFrame,
        ensure_pd: bool = True
    ) -> pd.DataFrame:
        """
        Calculate rolling covariance matrices.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series
        ensure_pd : bool
            Ensure positive definiteness
        
        Returns
        -------
        pd.DataFrame
            Rolling covariance (multi-index: date, asset pairs)
        """
        rolling_cov = returns.rolling(
            window=self.window,
            min_periods=self.min_periods
        ).cov()
        
        if ensure_pd:
            # This is computationally expensive for large matrices
            logger.warning("Positive definite enforcement on rolling cov is expensive")
        
        return rolling_cov
    
    def estimate_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate correlation matrix from covariance.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return series
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        cov = self.estimate(returns, ensure_pd=True)
        
        # Convert covariance to correlation
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        
        return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
    
    def forecast_covariance(
        self,
        returns: pd.DataFrame,
        horizon: int = 1
    ) -> pd.DataFrame:
        """
        Forecast covariance matrix for future horizon.
        
        Simple scaling approach: Cov(h) = h * Cov(1)
        More sophisticated approaches would use GARCH models.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical return series
        horizon : int
            Forecast horizon in days
        
        Returns
        -------
        pd.DataFrame
            Forecasted covariance matrix
        """
        one_period_cov = self.estimate(returns, ensure_pd=True)
        
        # Scale by horizon (assumes IID, oversimplification)
        forecasted_cov = one_period_cov * horizon
        
        logger.info(f"Forecasted covariance for {horizon}-day horizon")
        
        return forecasted_cov