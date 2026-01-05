"""
Dashboard configuration settings.
MERIDIAN - Dark theme with emerald accents.
"""

from typing import Dict, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DashboardConfig:
    """Dashboard configuration."""

    # Application settings
    app_title: str = "MERIDIAN"
    app_subtitle: str = "Portfolio Optimization System"
    app_icon: str = "assets/logo-icon.svg"
    layout: str = "wide"

    # Logo paths
    logo_header: str = "assets/MERIDIAN_white_full.svg"
    logo_icon: str = "assets/logo-icon.svg"
    logo_large: str = "assets/MERIDIAN-1080x1080.png"

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

    # Demo mode assets
    demo_assets: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "TLT", "GLD", "BTC-USD"
    ])

    # Color scheme - Emerald Finance Theme (Dark Mode)
    colors: Dict[str, str] = field(default_factory=lambda: {
        # Primary colors
        'primary': '#10B981',        # Emerald 500
        'primary_light': '#34D399',  # Emerald 400
        'primary_dark': '#059669',   # Emerald 600

        # Secondary colors
        'secondary': '#3B82F6',      # Blue 500
        'accent': '#8B5CF6',         # Violet 500

        # Semantic colors
        'success': '#10B981',        # Emerald
        'warning': '#F59E0B',        # Amber
        'danger': '#EF4444',         # Red
        'info': '#3B82F6',           # Blue

        # Background colors
        'bg_dark': '#0F172A',        # Slate 900
        'bg_card': '#1E293B',        # Slate 800
        'bg_elevated': '#334155',    # Slate 700

        # Text colors
        'text_primary': '#F1F5F9',   # Slate 100
        'text_secondary': '#94A3B8', # Slate 400
        'text_muted': '#64748B',     # Slate 500

        # Regime colors
        'regime_0': '#10B981',       # Low vol - Emerald (bullish)
        'regime_1': '#EF4444',       # High vol - Red (bearish)
        'regime_2': '#F59E0B',       # Medium - Amber (neutral)

        # Chart colors sequence
        'chart_1': '#10B981',        # Emerald
        'chart_2': '#3B82F6',        # Blue
        'chart_3': '#8B5CF6',        # Violet
        'chart_4': '#F59E0B',        # Amber
        'chart_5': '#EC4899',        # Pink
        'chart_6': '#14B8A6',        # Teal
    })

    def get_logo_path(self, logo_type: str = "header") -> Path:
        """Get the full path to a logo file."""
        project_root = Path(__file__).parent.parent.parent.parent
        logo_map = {
            "header": self.logo_header,
            "icon": self.logo_icon,
            "large": self.logo_large
        }
        return project_root / logo_map.get(logo_type, self.logo_header)

    def get_chart_colors(self) -> List[str]:
        """Get the chart color sequence."""
        return [
            self.colors['chart_1'],
            self.colors['chart_2'],
            self.colors['chart_3'],
            self.colors['chart_4'],
            self.colors['chart_5'],
            self.colors['chart_6'],
        ]


# Global config instance
CONFIG = DashboardConfig()
