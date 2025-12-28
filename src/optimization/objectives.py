"""
Objective functions for portfolio optimization.

Implements various optimization objectives:
- Mean-variance (Sharpe ratio maximization)
- Minimum variance
- Risk parity
- Maximum diversification
"""

import numpy as np
import pandas as pd
from typing import Optional, Callable
import cvxpy as cp

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ObjectiveFunction:
    """
    Portfolio optimization objective functions.
    
    Provides different optimization objectives for portfolio construction.
    """
    
    def __init__(self, objective_type: str = 'sharpe'):
        """
        Initialize objective function.
        
        Parameters
        ----------
        objective_type : str
            Type of objective ('sharpe', 'min_variance', 'risk_parity', 'max_diversification')
        """
        self.objective_type = objective_type
        
        logger.info(f"Initialized ObjectiveFunction: {objective_type}")
    
    def mean_variance_objective(
        self,
        weights: cp.Variable,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 1.0
    ) -> cp.Expression:
        """
        Mean-variance objective: maximize return - risk_aversion * variance.
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights (CVXPY variable)
        expected_returns : np.ndarray
            Expected return vector
        cov_matrix : np.ndarray
            Covariance matrix
        risk_aversion : float
            Risk aversion parameter (λ)
        
        Returns
        -------
        cp.Expression
            CVXPY objective expression
        """
        portfolio_return = expected_returns @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Maximize: return - λ * variance
        return portfolio_return - (risk_aversion / 2) * portfolio_variance
    
    def minimum_variance_objective(
        self,
        weights: cp.Variable,
        cov_matrix: np.ndarray
    ) -> cp.Expression:
        """
        Minimum variance objective.
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights
        cov_matrix : np.ndarray
            Covariance matrix
        
        Returns
        -------
        cp.Expression
            Variance to minimize
        """
        return cp.quad_form(weights, cov_matrix)
    
    def risk_parity_objective(
        self,
        weights: cp.Variable,
        cov_matrix: np.ndarray
    ) -> cp.Expression:
        """
        Risk parity objective: equalize risk contributions.
        
        Approximation using sum of squares of risk contributions.
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights
        cov_matrix : np.ndarray
            Covariance matrix
        
        Returns
        -------
        cp.Expression
            Risk parity objective (sum of squared differences)
        """
        n_assets = len(cov_matrix)
        
        # Marginal risk contribution: (Σw)_i
        marginal_risk = cov_matrix @ weights
        
        # Risk contribution: w_i * (Σw)_i
        risk_contribution = cp.multiply(weights, marginal_risk)
        
        # Target: equal risk contribution
        target = cp.sum(risk_contribution) / n_assets
        
        # Minimize sum of squared deviations from target
        return cp.sum_squares(risk_contribution - target)
    
    def maximum_diversification_objective(
        self,
        weights: cp.Variable,
        volatilities: np.ndarray,
        cov_matrix: np.ndarray
    ) -> cp.Expression:
        """
        Maximum diversification ratio.
        
        Maximize: (w'σ) / sqrt(w'Σw)
        Equivalent to minimizing portfolio variance with weighted assets.
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights
        volatilities : np.ndarray
            Individual asset volatilities
        cov_matrix : np.ndarray
            Covariance matrix
        
        Returns
        -------
        cp.Expression
            Objective (minimize variance)
        """
        # This is an approximation; true max diversification requires
        # iterative optimization due to non-convexity
        weighted_vol = volatilities @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Minimize variance (higher diversification)
        return portfolio_variance
    
    def build_objective(
        self,
        weights: cp.Variable,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
        volatilities: Optional[np.ndarray] = None,
        **kwargs
    ) -> cp.Expression:
        """
        Build objective based on type.
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights
        expected_returns : np.ndarray, optional
            Expected returns
        cov_matrix : np.ndarray, optional
            Covariance matrix
        volatilities : np.ndarray, optional
            Asset volatilities
        **kwargs : dict
            Additional parameters
        
        Returns
        -------
        cp.Expression
            Objective expression
        """
        if self.objective_type == 'sharpe' or self.objective_type == 'mean_variance':
            if expected_returns is None or cov_matrix is None:
                raise ValueError("Mean-variance requires expected_returns and cov_matrix")
            
            risk_aversion = kwargs.get('risk_aversion', 1.0)
            return self.mean_variance_objective(weights, expected_returns, cov_matrix, risk_aversion)
        
        elif self.objective_type == 'min_variance':
            if cov_matrix is None:
                raise ValueError("Min variance requires cov_matrix")
            
            return self.minimum_variance_objective(weights, cov_matrix)
        
        elif self.objective_type == 'risk_parity':
            if cov_matrix is None:
                raise ValueError("Risk parity requires cov_matrix")
            
            return self.risk_parity_objective(weights, cov_matrix)
        
        elif self.objective_type == 'max_diversification':
            if cov_matrix is None or volatilities is None:
                raise ValueError("Max diversification requires cov_matrix and volatilities")
            
            return self.maximum_diversification_objective(weights, volatilities, cov_matrix)
        
        else:
            raise ValueError(f"Unknown objective type: {self.objective_type}")