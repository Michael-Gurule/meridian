"""
Pre-built allocation strategies.

Common portfolio strategies for benchmarking and deployment.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, List

from src.optimization.optimizer import PortfolioOptimizer
from src.models.covariance import CovarianceEstimator
from src.models.returns import ReturnEstimator
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AllocationStrategy(ABC):
    """
    Base class for allocation strategies.
    
    Defines interface for portfolio allocation.
    """
    
    def __init__(self, name: str):
        """
        Initialize strategy.
        
        Parameters
        ----------
        name : str
            Strategy name
        """
        self.name = name
        logger.info(f"Initialized {name} strategy")
    
    @abstractmethod
    def allocate(
        self,
        prices: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Generate portfolio allocation.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data
        **kwargs : dict
            Additional parameters
        
        Returns
        -------
        np.ndarray
            Portfolio weights
        """
        pass


class EqualWeightStrategy(AllocationStrategy):
    """
    Equal weight (1/N) allocation.
    
    Naive diversification baseline.
    """
    
    def __init__(self):
        """Initialize equal weight strategy."""
        super().__init__("Equal Weight")
    
    def allocate(
        self,
        prices: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Equal weight allocation.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        np.ndarray
            Equal weights
        """
        n_assets = len(prices.columns)
        return np.ones(n_assets) / n_assets


class MinVarianceStrategy(AllocationStrategy):
    """
    Minimum variance portfolio.
    
    Minimizes portfolio volatility without considering returns.
    """
    
    def __init__(
        self,
        cov_method: str = 'ledoit_wolf',
        constraints: Optional[Dict] = None
    ):
        """
        Initialize minimum variance strategy.
        
        Parameters
        ----------
        cov_method : str
            Covariance estimation method
        constraints : Dict, optional
            Portfolio constraints
        """
        super().__init__("Minimum Variance")
        
        self.cov_estimator = CovarianceEstimator(method=cov_method)
        self.optimizer = PortfolioOptimizer(
            objective_type='min_variance',
            constraints=constraints or {'long_only': True}
        )
    
    def allocate(
        self,
        prices: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Minimum variance allocation.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        np.ndarray
            Optimal weights
        """
        returns = prices.pct_change().dropna()
        cov_matrix = self.cov_estimator.estimate(returns).values
        
        result = self.optimizer.optimize(cov_matrix=cov_matrix, **kwargs)
        
        return result['weights']


class RiskParityStrategy(AllocationStrategy):
    """
    Risk parity allocation.
    
    Equalizes risk contribution across assets.
    """
    
    def __init__(
        self,
        cov_method: str = 'sample',
        constraints: Optional[Dict] = None
    ):
        """
        Initialize risk parity strategy.
        
        Parameters
        ----------
        cov_method : str
            Covariance estimation method
        constraints : Dict, optional
            Portfolio constraints
        """
        super().__init__("Risk Parity")
        
        self.cov_estimator = CovarianceEstimator(method=cov_method)
        self.optimizer = PortfolioOptimizer(
            objective_type='risk_parity',
            constraints=constraints or {'long_only': True}
        )
    
    def allocate(
        self,
        prices: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Risk parity allocation.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        np.ndarray
            Optimal weights
        """
        returns = prices.pct_change().dropna()
        cov_matrix = self.cov_estimator.estimate(returns).values
        
        result = self.optimizer.optimize(cov_matrix=cov_matrix, **kwargs)
        
        return result['weights']


class MeanVarianceStrategy(AllocationStrategy):
    """
    Mean-variance (Markowitz) optimization.
    
    Maximizes Sharpe ratio.
    """
    
    def __init__(
        self,
        cov_method: str = 'ledoit_wolf',
        return_method: str = 'historical',
        risk_aversion: float = 1.0,
        constraints: Optional[Dict] = None
    ):
        """
        Initialize mean-variance strategy.
        
        Parameters
        ----------
        cov_method : str
            Covariance estimation method
        return_method : str
            Return estimation method
        risk_aversion : float
            Risk aversion parameter
        constraints : Dict, optional
            Portfolio constraints
        """
        super().__init__("Mean-Variance")
        
        self.cov_estimator = CovarianceEstimator(method=cov_method)
        self.return_estimator = ReturnEstimator(method=return_method)
        self.risk_aversion = risk_aversion
        
        self.optimizer = PortfolioOptimizer(
            objective_type='mean_variance',
            constraints=constraints or {'long_only': True}
        )
    
    def allocate(
        self,
        prices: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        Mean-variance allocation.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        np.ndarray
            Optimal weights
        """
        returns = prices.pct_change().dropna()
        
        cov_matrix = self.cov_estimator.estimate(returns).values
        expected_returns = self.return_estimator.estimate(returns, annualize=False).values
        
        result = self.optimizer.optimize(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            risk_aversion=self.risk_aversion,
            **kwargs
        )
        
        return result['weights']


class SixtyFortyStrategy(AllocationStrategy):
    """
    Classic 60/40 stocks/bonds allocation.
    
    Traditional balanced portfolio benchmark.
    """
    
    def __init__(
        self,
        equity_tickers: Optional[List[str]] = None,
        bond_tickers: Optional[List[str]] = None
    ):
        """
        Initialize 60/40 strategy.
        
        Parameters
        ----------
        equity_tickers : List[str], optional
            List of equity tickers
        bond_tickers : List[str], optional
            List of bond tickers
        """
        super().__init__("60/40 Portfolio")
        self.equity_tickers = equity_tickers or ['SPY', 'QQQ', 'VTI']
        self.bond_tickers = bond_tickers or ['TLT', 'IEF', 'LQD']
    
    def allocate(
        self,
        prices: pd.DataFrame,
        **kwargs
    ) -> np.ndarray:
        """
        60/40 allocation.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        np.ndarray
            60/40 weights
        """
        n_assets = len(prices.columns)
        weights = np.zeros(n_assets)
        
        # Identify equity and bond assets
        equity_mask = prices.columns.isin(self.equity_tickers)
        bond_mask = prices.columns.isin(self.bond_tickers)
        
        n_equities = equity_mask.sum()
        n_bonds = bond_mask.sum()
        
        if n_equities > 0:
            weights[equity_mask] = 0.6 / n_equities
        
        if n_bonds > 0:
            weights[bond_mask] = 0.4 / n_bonds
        
        # Renormalize if some assets don't match
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            # Fallback to equal weight
            weights = np.ones(n_assets) / n_assets
        
        return weights