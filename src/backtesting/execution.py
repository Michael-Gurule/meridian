"""
Trade execution simulation.

Simulates realistic trade execution including slippage,
market impact, and partial fills.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict

from src.optimization.costs import TransactionCostModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionSimulator:
    """
    Simulates realistic trade execution.
    
    Models slippage, market impact, and transaction costs.
    """
    
    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        slippage_bps: float = 2.0
    ):
        """
        Initialize execution simulator.
        
        Parameters
        ----------
        cost_model : TransactionCostModel, optional
            Transaction cost model
        slippage_bps : float
            Additional slippage in basis points
        """
        self.cost_model = cost_model or TransactionCostModel()
        self.slippage_bps = slippage_bps
        
        logger.info("Initialized ExecutionSimulator")
    
    def execute_rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        prices: np.ndarray,
        portfolio_value: float,
        avg_volumes: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Execute portfolio rebalancing.
        
        Parameters
        ----------
        current_weights : np.ndarray
            Current portfolio weights
        target_weights : np.ndarray
            Target portfolio weights
        prices : np.ndarray
            Current asset prices
        portfolio_value : float
            Total portfolio value
        avg_volumes : np.ndarray, optional
            Average daily volumes
        
        Returns
        -------
        Dict
            Execution results with:
            - executed_weights: actual weights after execution
            - total_cost: total transaction cost
            - slippage_cost: slippage cost
            - impact_cost: market impact cost
        """
        n_assets = len(current_weights)
        
        # Calculate trades
        weight_changes = target_weights - current_weights
        trade_values = np.abs(weight_changes) * portfolio_value
        
        # Calculate costs
        total_cost = self.cost_model.total_cost(
            current_weights, target_weights, prices, portfolio_value, avg_volumes
        )
        
        # Slippage
        slippage_cost = np.sum(trade_values) * (self.slippage_bps / 10000)
        
        # Total execution cost
        total_execution_cost = total_cost + slippage_cost
        
        # Adjust portfolio value for costs
        net_portfolio_value = portfolio_value - total_execution_cost
        
        # Executed weights (accounting for cost reduction)
        executed_weights = target_weights * (net_portfolio_value / portfolio_value)
        
        # Renormalize
        executed_weights = executed_weights / executed_weights.sum()
        
        return {
            'executed_weights': executed_weights,
            'total_cost': total_execution_cost,
            'cost_pct': total_execution_cost / portfolio_value,
            'slippage_cost': slippage_cost,
            'net_portfolio_value': net_portfolio_value
        }
    
    def simulate_limit_order(
        self,
        order_size: float,
        limit_price: float,
        market_prices: np.ndarray,
        fill_probability: float = 0.8
    ) -> Dict:
        """
        Simulate limit order execution.
        
        Parameters
        ----------
        order_size : float
            Order size (shares or dollars)
        limit_price : float
            Limit price
        market_prices : np.ndarray
            Market price path
        fill_probability : float
            Probability of fill at limit
        
        Returns
        -------
        Dict
            Execution result
        """
        # Simple model: fill if market touches limit
        touched = (market_prices <= limit_price).any()
        filled = touched and (np.random.random() < fill_probability)
        
        if filled:
            fill_price = limit_price
            fill_size = order_size
        else:
            fill_price = np.nan
            fill_size = 0
        
        return {
            'filled': filled,
            'fill_price': fill_price,
            'fill_size': fill_size
        }