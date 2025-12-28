"""
Dashboard component modules.
"""

from .portfolio import render_portfolio_view
from .optimization import render_optimization_view
from .regime import render_regime_view
from .performance import render_performance_view
from .scenarios import render_scenario_view
from .recommendations import render_recommendations_view

__all__ = [
    'render_portfolio_view',
    'render_optimization_view',
    'render_regime_view',
    'render_performance_view',
    'render_scenario_view',
    'render_recommendations_view'
]