"""
Portfolio constraint definitions.

Implements various constraints:
- Weight bounds (long-only, long-short)
- Budget constraint
- Turnover limits
- Sector/group constraints
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
import cvxpy as cp

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConstraintBuilder:
    """
    Builds portfolio constraints for optimization.
    
    Provides methods to construct various constraint types.
    """
    
    def __init__(self):
        """Initialize constraint builder."""
        logger.info("Initialized ConstraintBuilder")
    
    def budget_constraint(
        self,
        weights: cp.Variable
    ) -> cp.Constraint:
        """
        Budget constraint: weights sum to 1.
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights
        
        Returns
        -------
        cp.Constraint
            Budget constraint
        """
        return cp.sum(weights) == 1
    
    def long_only_constraint(
        self,
        weights: cp.Variable
    ) -> cp.Constraint:
        """
        Long-only constraint: all weights >= 0.
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights
        
        Returns
        -------
        cp.Constraint
            Non-negativity constraint
        """
        return weights >= 0
    
    def weight_bounds_constraint(
        self,
        weights: cp.Variable,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None
    ) -> List[cp.Constraint]:
        """
        Individual weight bounds.
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights
        lower_bounds : np.ndarray, optional
            Lower bounds for each asset
        upper_bounds : np.ndarray, optional
            Upper bounds for each asset
        
        Returns
        -------
        List[cp.Constraint]
            Bound constraints
        """
        constraints = []
        
        if lower_bounds is not None:
            constraints.append(weights >= lower_bounds)
        
        if upper_bounds is not None:
            constraints.append(weights <= upper_bounds)
        
        return constraints
    
    def turnover_constraint(
        self,
        weights: cp.Variable,
        current_weights: np.ndarray,
        max_turnover: float
    ) -> cp.Constraint:
        """
        Turnover constraint: limit trading.
        
        Turnover = sum(|w_new - w_old|) / 2
        
        Parameters
        ----------
        weights : cp.Variable
            New portfolio weights
        current_weights : np.ndarray
            Current portfolio weights
        max_turnover : float
            Maximum allowed turnover (0 to 1)
        
        Returns
        -------
        cp.Constraint
            Turnover constraint
        """
        # sum(|w_new - w_old|) <= 2 * max_turnover
        return cp.norm(weights - current_weights, 1) <= 2 * max_turnover
    
    def group_constraint(
        self,
        weights: cp.Variable,
        group_matrix: np.ndarray,
        group_lower: Optional[np.ndarray] = None,
        group_upper: Optional[np.ndarray] = None
    ) -> List[cp.Constraint]:
        """
        Group/sector constraints.
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights
        group_matrix : np.ndarray
            Group membership matrix (n_groups × n_assets)
            group_matrix[i,j] = 1 if asset j in group i
        group_lower : np.ndarray, optional
            Lower bounds for group weights
        group_upper : np.ndarray, optional
            Upper bounds for group weights
        
        Returns
        -------
        List[cp.Constraint]
            Group constraints
        """
        constraints = []
        
        # Group weights = group_matrix @ weights
        group_weights = group_matrix @ weights
        
        if group_lower is not None:
            constraints.append(group_weights >= group_lower)
        
        if group_upper is not None:
            constraints.append(group_weights <= group_upper)
        
        return constraints
    
    def leverage_constraint(
        self,
        weights: cp.Variable,
        max_leverage: float = 1.0
    ) -> cp.Constraint:
        """
        Leverage constraint: sum of absolute weights.
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights
        max_leverage : float
            Maximum leverage (1.0 = no leverage)
        
        Returns
        -------
        cp.Constraint
            Leverage constraint
        """
        return cp.norm(weights, 1) <= max_leverage
    
    def tracking_error_constraint(
        self,
        weights: cp.Variable,
        benchmark_weights: np.ndarray,
        cov_matrix: np.ndarray,
        max_tracking_error: float
    ) -> cp.Constraint:
        """
        Tracking error constraint vs. benchmark.
        
        TE = sqrt((w - w_b)' Σ (w - w_b))
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights
        benchmark_weights : np.ndarray
            Benchmark weights
        cov_matrix : np.ndarray
            Covariance matrix
        max_tracking_error : float
            Maximum tracking error (annualized)
        
        Returns
        -------
        cp.Constraint
            Tracking error constraint
        """
        active_weights = weights - benchmark_weights
        tracking_variance = cp.quad_form(active_weights, cov_matrix)
        
        # Tracking error squared
        return tracking_variance <= max_tracking_error ** 2
    
    def build_standard_constraints(
        self,
        weights: cp.Variable,
        long_only: bool = True,
        weight_bounds: Optional[Tuple[float, float]] = None,
        current_weights: Optional[np.ndarray] = None,
        max_turnover: Optional[float] = None
    ) -> List[cp.Constraint]:
        """
        Build standard constraint set.
        
        Parameters
        ----------
        weights : cp.Variable
            Portfolio weights
        long_only : bool
            Long-only constraint
        weight_bounds : Tuple[float, float], optional
            (min_weight, max_weight) for all assets
        current_weights : np.ndarray, optional
            Current weights (for turnover)
        max_turnover : float, optional
            Maximum turnover
        
        Returns
        -------
        List[cp.Constraint]
            List of constraints
        """
        constraints = []
        
        # Budget constraint (always included)
        constraints.append(self.budget_constraint(weights))
        
        # Long-only
        if long_only:
            constraints.append(self.long_only_constraint(weights))
        
        # Weight bounds
        if weight_bounds is not None:
            n_assets = weights.shape[0]
            lower = np.full(n_assets, weight_bounds[0])
            upper = np.full(n_assets, weight_bounds[1])
            constraints.extend(self.weight_bounds_constraint(weights, lower, upper))
        
        # Turnover
        if max_turnover is not None and current_weights is not None:
            constraints.append(
                self.turnover_constraint(weights, current_weights, max_turnover)
            )
        
        return constraints