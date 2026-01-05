"""
Demo data module for MERIDIAN dashboard.
Provides quick-start sample portfolio data for demonstrations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.data.storage import DataStorage
from src.dashboard.config.dashboard_config import CONFIG


def load_demo_portfolio():
    """
    Load demo portfolio data from stored processed files.

    Returns
    -------
    dict
        Dictionary containing:
        - prices: DataFrame of asset prices
        - returns: DataFrame of asset returns
        - assets: List of asset tickers
        - current_weights: Series of equal weights
        - portfolio_value: Default portfolio value
    """
    storage = DataStorage()

    # Get available assets that match demo config
    available = storage.list_available_assets(data_type="processed")
    demo_assets = [a for a in CONFIG.demo_assets if a in available]

    if len(demo_assets) < 2:
        # Fallback to first 5 available assets
        demo_assets = available[:5]

    if len(demo_assets) < 2:
        return None

    # Load data
    data_dict = storage.load_batch(demo_assets, data_type="processed")

    # Build prices DataFrame
    prices = pd.DataFrame({
        ticker: data["close"] for ticker, data in data_dict.items()
    })

    # Use last 2 years of data
    end_date = prices.index.max()
    start_date = end_date - timedelta(days=2 * 365)
    prices = prices[prices.index >= start_date].dropna()

    # Calculate returns
    returns = prices.pct_change().dropna()

    # Equal weight starting portfolio
    n_assets = len(demo_assets)
    current_weights = pd.Series(
        np.ones(n_assets) / n_assets,
        index=demo_assets
    )

    return {
        'prices': prices,
        'returns': returns,
        'assets': demo_assets,
        'current_weights': current_weights,
        'portfolio_value': CONFIG.default_portfolio_value
    }


def get_demo_optimization_result():
    """
    Get a sample optimization result for demo purposes.

    Returns
    -------
    dict
        Optimization result dictionary
    """
    # Sample optimized weights (more interesting than equal weight)
    demo_weights = {
        'SPY': 0.35,
        'QQQ': 0.20,
        'TLT': 0.25,
        'GLD': 0.15,
        'BTC-USD': 0.05
    }

    return {
        'weights': demo_weights,
        'expected_return': 0.0004,  # ~10% annual
        'volatility': 0.01,         # ~16% annual
        'sharpe_ratio': 0.65,
        'turnover': 0.15,
        'objective_value': 0.0025
    }


def get_demo_summary_stats():
    """
    Get summary statistics for demo display.

    Returns
    -------
    dict
        Summary statistics
    """
    return {
        'total_assets': 25,
        'data_years': 10,
        'last_updated': datetime.now().strftime("%Y-%m-%d"),
        'data_points': 2520,
        'asset_classes': 4
    }


def get_feature_highlights():
    """
    Get feature highlights for the welcome screen.

    Returns
    -------
    list
        List of feature dictionaries
    """
    return [
        {
            'icon': '📊',
            'title': 'Multi-Asset Optimization',
            'description': '25 assets across equities, bonds, commodities, and crypto'
        },
        {
            'icon': '🎯',
            'title': 'Regime Detection',
            'description': 'Hidden Markov Models identify market states automatically'
        },
        {
            'icon': '📈',
            'title': 'Walk-Forward Backtesting',
            'description': 'Realistic testing with transaction costs and slippage'
        },
        {
            'icon': '🔮',
            'title': 'Scenario Analysis',
            'description': 'Stress tests, Monte Carlo, and historical event replay'
        }
    ]
