"""
Dashboard configuration settings.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    
    # Application settings
    app_title: str = "MERIDIAN Portfolio Optimization"
    app_icon: str = "📊"
    layout: str = "wide"
    
    # Portfolio settings
    default_portfolio_value: float = 1_000_000.0
    default_rebalance_frequency: str = "M"
    
    # Optimization settings
    default_objective: str = "mean_variance"
    default_risk_aversion: float = 1.0
    max_position_size: float = 0.4
    max_turnover: float = 0.5
    
    # Display settings
    show_regime_detection: bool = True
    show_transaction_costs: bool = True
    n_efficient_frontier_points: int = 30
    
    # Color scheme
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default colors."""
        if self.colors is None:
            self.colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'success': '#2ca02c',
                'danger': '#d62728',
                'warning': '#ffbb00',
                'info': '#17a2b8',
                'regime_0': '#2ecc71',
                'regime_1': '#e74c3c',
                'regime_2': '#f39c12'
            }


# Global config instance
CONFIG = DashboardConfig()