"""
Transaction cost modeling.

Models various transaction costs:
- Bid-ask spread
- Market impact
- Commission fees
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransactionCostModel:
    """
    Models transaction costs for portfolio rebalancing.
    
    Includes bid-ask spread, market impact, and commissions.
    """
    
    def __init__(
        self,
        spread_bps: float = 5.0,
        impact_coefficient: float = 0.1,
        commission_bps: float = 1.0,
        min_commission: float = 1.0
    ):
        """
        Initialize transaction cost model.
        
        Parameters
        ----------
        spread_bps : float
            Bid-ask spread in basis points
        impact_coefficient : float
            Market impact coefficient (percentage per sqrt(volume))
        commission_bps : float
            Commission in basis points
        min_commission : float
            Minimum commission per trade ($)
        """
        self.spread_bps = spread_bps
        self.impact_coefficient = impact_coefficient
        self.commission_bps = commission_bps
        self.min_commission = min_commission
        
        logger.info(f"Initialized TransactionCostModel: spread={spread_bps}bps")
    
    def bid_ask_cost(
        self,
        trade_value: float
    ) -> float:
        """
        Calculate bid-ask spread cost.
        
        Parameters
        ----------
        trade_value : float
            Absolute value of trade ($)
        
        Returns
        -------
        float
            Bid-ask cost ($)
        """
        return trade_value * (self.spread_bps / 10000)
    
    def market_impact_cost(
        self,
        trade_value: float,
        avg_daily_volume: float
    ) -> float:
        """
        Calculate market impact cost.
        
        Square-root law: cost ∝ sqrt(trade_size / volume)
        
        Parameters
        ----------
        trade_value : float
            Absolute value of trade ($)
        avg_daily_volume : float
            Average daily dollar volume
        
        Returns
        -------
        float
            Market impact cost ($)
        """
        if avg_daily_volume <= 0:
            return 0.0
        
        volume_fraction = trade_value / avg_daily_volume
        impact = self.impact_coefficient * np.sqrt(volume_fraction)
        
        return trade_value * impact
    
    def commission_cost(
        self,
        trade_value: float
    ) -> float:
        """
        Calculate commission cost.
        
        Parameters
        ----------
        trade_value : float
            Absolute value of trade ($)
        
        Returns
        -------
        float
            Commission cost ($)
        """
        commission = trade_value * (self.commission_bps / 10000)
        return max(commission, self.min_commission)
    
    def total_cost(
        self,
        current_weights: np.ndarray,
        new_weights: np.ndarray,
        prices: np.ndarray,
        portfolio_value: float,
        avg_volumes: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate total transaction cost for rebalancing.
        
        Parameters
        ----------
        current_weights : np.ndarray
            Current portfolio weights
        new_weights : np.ndarray
            Target portfolio weights
        prices : np.ndarray
            Current prices
        portfolio_value : float
            Total portfolio value ($)
        avg_volumes : np.ndarray, optional
            Average daily dollar volumes
        
        Returns
        -------
        float
            Total transaction cost ($)
        """
        # Trade values
        weight_changes = np.abs(new_weights - current_weights)
        trade_values = weight_changes * portfolio_value
        
        total_cost = 0.0
        
        for i, trade_val in enumerate(trade_values):
            if trade_val < 1.0:  # Skip tiny trades
                continue
            
            # Bid-ask spread
            total_cost += self.bid_ask_cost(trade_val)
            
            # Market impact
            if avg_volumes is not None and avg_volumes[i] > 0:
                total_cost += self.market_impact_cost(trade_val, avg_volumes[i])
            
            # Commission
            total_cost += self.commission_cost(trade_val)
        
        return total_cost
    
    def cost_as_percentage(
        self,
        current_weights: np.ndarray,
        new_weights: np.ndarray,
        prices: np.ndarray,
        portfolio_value: float,
        avg_volumes: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate transaction cost as percentage of portfolio.
        
        Parameters
        ----------
        current_weights : np.ndarray
            Current weights
        new_weights : np.ndarray
            Target weights
        prices : np.ndarray
            Prices
        portfolio_value : float
            Portfolio value
        avg_volumes : np.ndarray, optional
            Average volumes
        
        Returns
        -------
        float
            Cost as percentage (0.01 = 1%)
        """
        cost = self.total_cost(
            current_weights, new_weights, prices, portfolio_value, avg_volumes
        )
        
        return cost / portfolio_value
    
    def estimate_breakeven_horizon(
        self,
        expected_alpha: float,
        transaction_cost_pct: float
    ) -> float:
        """
        Estimate breakeven holding period.
        
        Time needed for expected alpha to exceed transaction costs.
        
        Parameters
        ----------
        expected_alpha : float
            Expected annual alpha (excess return)
        transaction_cost_pct : float
            Transaction cost as percentage
        
        Returns
        -------
        float
            Breakeven horizon in years
        """
        if expected_alpha <= 0:
            return np.inf
        
        # Years needed: cost / annual_alpha
        return transaction_cost_pct / expected_alpha