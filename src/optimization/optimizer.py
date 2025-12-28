"""
Main portfolio optimizer.

Integrates objectives, constraints, and cost models to solve
portfolio optimization problems.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import cvxpy as cp

from src.optimization.objectives import ObjectiveFunction
from src.optimization.constraints import ConstraintBuilder
from src.optimization.costs import TransactionCostModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioOptimizer:
    """
    Main portfolio optimization engine.
    
    Solves portfolio optimization problems with multiple objectives,
    constraints, and transaction costs.
    """
    
    def __init__(
        self,
        objective_type: str = 'sharpe',
        constraints: Optional[Dict] = None,
        transaction_costs: bool = True
    ):
        """
        Initialize portfolio optimizer.
        
        Parameters
        ----------
        objective_type : str
            Optimization objective
        constraints : Dict, optional
            Constraint specifications
        transaction_costs : bool
            Include transaction costs in optimization
        """
        self.objective_type = objective_type
        self.constraint_specs = constraints or {}
        self.use_transaction_costs = transaction_costs
        
        self.objective_fn = ObjectiveFunction(objective_type)
        self.constraint_builder = ConstraintBuilder()
        self.cost_model = TransactionCostModel() if transaction_costs else None
        
        self.last_solution = None
        
        logger.info(f"Initialized PortfolioOptimizer: {objective_type}")
    
    def optimize(
        self,
        expected_returns: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
        current_weights: Optional[np.ndarray] = None,
        risk_aversion: float = 1.0,
        **kwargs
    ) -> Dict:
        """
        Solve portfolio optimization problem.
        
        Parameters
        ----------
        expected_returns : np.ndarray, optional
            Expected return vector
        cov_matrix : np.ndarray, optional
            Covariance matrix
        current_weights : np.ndarray, optional
            Current portfolio weights
        risk_aversion : float
            Risk aversion parameter
        **kwargs : dict
            Additional parameters
        
        Returns
        -------
        Dict
            Optimization results with keys:
            - weights: optimal weights
            - objective_value: objective function value
            - status: solver status
            - expected_return: portfolio expected return
            - volatility: portfolio volatility
            - sharpe_ratio: Sharpe ratio
        """
        if cov_matrix is None:
            raise ValueError("Covariance matrix required")
        
        n_assets = len(cov_matrix)
        
        # Handle current weights
        if current_weights is None:
            current_weights = np.zeros(n_assets)
        
        # Define optimization variable
        weights = cp.Variable(n_assets)
        
        # Build objective
        if expected_returns is None:
            expected_returns = np.zeros(n_assets)
        
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        objective_expr = self.objective_fn.build_objective(
            weights=weights,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            volatilities=volatilities,
            risk_aversion=risk_aversion,
            **kwargs
        )
        
        # Build constraints
        constraints = self._build_constraints(weights, current_weights, n_assets)
        
        # Solve optimization problem
        if self.objective_type == 'min_variance' or self.objective_type == 'risk_parity':
            # Minimize
            problem = cp.Problem(cp.Minimize(objective_expr), constraints)
        else:
            # Maximize
            problem = cp.Problem(cp.Maximize(objective_expr), constraints)
        
        # Solve
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                logger.warning(f"Optimization status: {problem.status}")
                return self._failed_result(n_assets, problem.status)
            
            optimal_weights = weights.value
            
            # Calculate portfolio statistics
            portfolio_return = expected_returns @ optimal_weights
            portfolio_variance = optimal_weights @ cov_matrix @ optimal_weights
            portfolio_vol = np.sqrt(portfolio_variance)
            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0
            
            # Transaction costs
            transaction_cost = 0.0
            if self.use_transaction_costs and self.cost_model is not None:
                prices = kwargs.get('prices', np.ones(n_assets))
                portfolio_value = kwargs.get('portfolio_value', 1000000.0)
                avg_volumes = kwargs.get('avg_volumes', None)
                
                transaction_cost = self.cost_model.cost_as_percentage(
                    current_weights, optimal_weights, prices, portfolio_value, avg_volumes
                )
            
            result = {
                'weights': optimal_weights,
                'objective_value': problem.value,
                'status': problem.status,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe,
                'turnover': np.sum(np.abs(optimal_weights - current_weights)) / 2,
                'transaction_cost_pct': transaction_cost,
                'net_return': portfolio_return - transaction_cost
            }
            
            self.last_solution = result
            
            logger.info(f"Optimization successful: return={portfolio_return:.4f}, vol={portfolio_vol:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return self._failed_result(n_assets, 'error')
    
    def _build_constraints(
        self,
        weights: cp.Variable,
        current_weights: np.ndarray,
        n_assets: int
    ) -> List[cp.Constraint]:
        """
        Build constraint list from specifications.
        
        Parameters
        ----------
        weights : cp.Variable
            Optimization variable
        current_weights : np.ndarray
            Current weights
        n_assets : int
            Number of assets
        
        Returns
        -------
        List[cp.Constraint]
            Constraint list
        """
        # Default constraints
        long_only = self.constraint_specs.get('long_only', True)
        weight_bounds = self.constraint_specs.get('weight_bounds', None)
        max_turnover = self.constraint_specs.get('max_turnover', None)
        
        constraints = self.constraint_builder.build_standard_constraints(
            weights=weights,
            long_only=long_only,
            weight_bounds=weight_bounds,
            current_weights=current_weights if max_turnover is not None else None,
            max_turnover=max_turnover
        )
        
        # Additional custom constraints
        if 'max_position_size' in self.constraint_specs:
            max_size = self.constraint_specs['max_position_size']
            upper = np.full(n_assets, max_size)
            constraints.extend(
                self.constraint_builder.weight_bounds_constraint(weights, upper_bounds=upper)
            )
        
        if 'min_position_size' in self.constraint_specs:
            min_size = self.constraint_specs['min_position_size']
            # For non-zero positions only (handled via binary variables in advanced version)
            pass
        
        return constraints
    
    def _failed_result(self, n_assets: int, status: str) -> Dict:
        """
        Return result dict for failed optimization.
        
        Parameters
        ----------
        n_assets : int
            Number of assets
        status : str
            Solver status
        
        Returns
        -------
        Dict
            Failed result with equal weights
        """
        equal_weights = np.ones(n_assets) / n_assets
        
        return {
            'weights': equal_weights,
            'objective_value': np.nan,
            'status': status,
            'expected_return': np.nan,
            'volatility': np.nan,
            'sharpe_ratio': np.nan,
            'turnover': np.nan,
            'transaction_cost_pct': np.nan,
            'net_return': np.nan
        }
    
    def regime_conditional_optimize(
        self,
        regime_probabilities: np.ndarray,
        regime_returns: List[np.ndarray],
        regime_covariances: List[np.ndarray],
        current_weights: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """
        Optimize considering multiple regime scenarios.
        
        Weighted combination of regime-specific optimizations.
        
        Parameters
        ----------
        regime_probabilities : np.ndarray
            Probability of each regime
        regime_returns : List[np.ndarray]
            Expected returns for each regime
        regime_covariances : List[np.ndarray]
            Covariance matrices for each regime
        current_weights : np.ndarray, optional
            Current weights
        **kwargs : dict
            Additional parameters
        
        Returns
        -------
        Dict
            Optimization result
        """
        n_regimes = len(regime_probabilities)
        n_assets = len(regime_returns[0])
        
        # Probability-weighted expected returns and covariance
        weighted_returns = sum(
            p * r for p, r in zip(regime_probabilities, regime_returns)
        )
        
        weighted_cov = sum(
            p * c for p, c in zip(regime_probabilities, regime_covariances)
        )
        
        # Optimize using weighted statistics
        result = self.optimize(
            expected_returns=weighted_returns,
            cov_matrix=weighted_cov,
            current_weights=current_weights,
            **kwargs
        )
        
        # Store regime information
        result['regime_probabilities'] = regime_probabilities
        result['regime_specific'] = {}
        
        # Calculate performance in each regime
        for i in range(n_regimes):
            regime_return = regime_returns[i] @ result['weights']
            regime_vol = np.sqrt(result['weights'] @ regime_covariances[i] @ result['weights'])
            
            result['regime_specific'][i] = {
                'expected_return': regime_return,
                'volatility': regime_vol,
                'sharpe': regime_return / regime_vol if regime_vol > 0 else 0.0
            }
        
        logger.info(f"Regime-conditional optimization: {n_regimes} regimes")
        
        return result
    
    def efficient_frontier(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_points: int = 20,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns
        cov_matrix : np.ndarray
            Covariance matrix
        n_points : int
            Number of points on frontier
        **kwargs : dict
            Additional parameters
        
        Returns
        -------
        pd.DataFrame
            Efficient frontier with returns and volatilities
        """
        # Min variance portfolio
        min_var_result = PortfolioOptimizer(
            objective_type='min_variance',
            constraints=self.constraint_specs
        ).optimize(cov_matrix=cov_matrix, **kwargs)
        
        min_return = expected_returns @ min_var_result['weights']
        max_return = expected_returns.max()
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_results = []
        
        for target in target_returns:
            # Add target return constraint
            n_assets = len(expected_returns)
            weights = cp.Variable(n_assets)
            
            constraints = self._build_constraints(
                weights, np.zeros(n_assets), n_assets
            )
            constraints.append(expected_returns @ weights >= target)
            
            # Minimize variance
            objective = cp.quad_form(weights, cov_matrix)
            problem = cp.Problem(cp.Minimize(objective), constraints)
            
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
                
                if problem.status in ['optimal', 'optimal_inaccurate']:
                    opt_weights = weights.value
                    port_return = expected_returns @ opt_weights
                    port_vol = np.sqrt(opt_weights @ cov_matrix @ opt_weights)
                    
                    frontier_results.append({
                        'return': port_return,
                        'volatility': port_vol,
                        'sharpe': port_return / port_vol if port_vol > 0 else 0.0,
                        'weights': opt_weights
                    })
            except:
                continue
        
        return pd.DataFrame(frontier_results)