"""
MERIDIAN Portfolio Optimization Dashboard
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.storage import DataStorage
from src.dashboard.config.dashboard_config import CONFIG
from src.dashboard.utils.state import initialize_session_state, get_state, set_state
from src.dashboard.components import (
    render_portfolio_view,
    render_optimization_view,
    render_regime_view,
    render_performance_view,
    render_scenario_view,
    render_recommendations_view,
)
from src.models.covariance import CovarianceEstimator
from src.models.returns import ReturnEstimator


# Page configuration
st.set_page_config(
    page_title=CONFIG.app_title,
    page_icon=CONFIG.app_icon,
    layout=CONFIG.layout,
    initial_sidebar_state="expanded",
)

# Initialize session state
initialize_session_state()


def load_data():
    """Load market data."""
    storage = DataStorage()

    # Get available assets
    assets = storage.list_available_assets(data_type="processed")

    return storage, assets


def main():
    """Main dashboard application."""

    # Title and description
    st.title(f"{CONFIG.app_icon} {CONFIG.app_title}")
    st.markdown(
        "**Production-grade portfolio optimization with regime detection and scenario analysis**"
    )

    # Sidebar
    with st.sidebar:
        st.header(" Settings")

        # Load data
        storage, available_assets = load_data()

        # Asset selection
        st.subheader("Asset Universe")

        default_assets = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD"]
        default_selection = [a for a in default_assets if a in available_assets]

        selected_assets = st.multiselect(
            "Select Assets",
            options=available_assets,
            default=default_selection,
            help="Choose assets for portfolio optimization",
        )

        if len(selected_assets) < 2:
            st.error("Please select at least 2 assets")
            return

        # Portfolio settings
        st.subheader("Portfolio Settings")

        portfolio_value = st.number_input(
            "Portfolio Value ($)",
            min_value=10000,
            max_value=100000000,
            value=int(get_state("portfolio_value", CONFIG.default_portfolio_value)),
            step=10000,
            format="%d",
        )
        set_state("portfolio_value", portfolio_value)

        # Date range
        st.subheader("Data Range")

        lookback_years = st.slider(
            "Historical Data (years)",
            min_value=1,
            max_value=5,
            value=2,
            help="Years of historical data to load",
        )

        # Load data button
        if st.button(" Load Data", type="primary", use_container_width=True):
            with st.spinner("Loading market data..."):
                try:
                    data_dict = storage.load_batch(
                        selected_assets, data_type="processed"
                    )

                    prices = pd.DataFrame(
                        {ticker: data["close"] for ticker, data in data_dict.items()}
                    )

                    # Filter by date range
                    end_date = prices.index.max()
                    start_date = end_date - timedelta(days=lookback_years * 365)
                    prices = prices[prices.index >= start_date]

                    returns = prices.pct_change().dropna()

                    # Store in session state
                    set_state("prices", prices)
                    set_state("returns", returns)
                    set_state("selected_assets", selected_assets)

                    st.success(
                        f" Loaded {len(prices)} days of data for {len(selected_assets)} assets"
                    )

                except Exception as e:
                    st.error(f" Failed to load data: {str(e)}")

        # Data info
        if get_state("prices") is not None:
            prices = get_state("prices")
            st.info(
                f"**Data Loaded:** {len(prices)} days\n\n"
                f"**Date Range:** {prices.index.min().date()} to {prices.index.max().date()}"
            )

    # Main content
    prices = get_state("prices")
    returns = get_state("returns")

    if prices is None or returns is None:
        st.info("👈 Please load data using the sidebar to get started")

        # Show sample visualization
        st.subheader("Sample Dashboard Preview")
        st.image(
            "https://via.placeholder.com/1200x600.png?text=Portfolio+Dashboard+Preview",
            use_container_width=True,
        )

        return

    # Navigation tabs
    tabs = st.tabs(
        [
            " Portfolio",
            " Optimization",
            " Regime Detection",
            " Performance",
            " Scenarios",
            " Recommendations",
        ]
    )

    # Initialize current weights if not set OR if assets changed
    current_state_weights = get_state("current_weights")
    if current_state_weights is None or set(current_state_weights.index) != set(
        returns.columns
    ):
        # Equal weight as default for ALL assets in returns
        n_assets = len(returns.columns)
        current_weights = pd.Series(np.ones(n_assets) / n_assets, index=returns.columns)
        set_state("current_weights", current_weights)
    else:
        current_weights = current_state_weights

    # Estimate parameters for portfolio view
    cov_estimator = CovarianceEstimator(method="ledoit_wolf")
    return_estimator = ReturnEstimator(method="historical")

    recent_returns = returns.iloc[-252:]
    cov_matrix = cov_estimator.estimate(recent_returns)
    expected_returns = return_estimator.estimate(recent_returns, annualize=False)

    # Tab 1: Portfolio Overview
    with tabs[0]:
        render_portfolio_view(
            weights=current_weights,
            portfolio_value=portfolio_value,
            prices=prices,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
        )

    # Tab 2: Optimization
    with tabs[1]:
        render_optimization_view(
            prices=prices, returns=returns, current_weights=current_weights
        )

        # Update weights button
        opt_result = get_state("optimization_result")
        if opt_result is not None:
            if st.button(" Apply Optimal Weights", use_container_width=True):
                new_weights = pd.Series(opt_result["weights"], index=selected_assets)
                set_state("current_weights", new_weights)
                st.success("Portfolio weights updated!")
                st.rerun()

    # Tab 3: Regime Detection
    with tabs[2]:
        render_regime_view(returns, prices)

    # Tab 4: Performance
    with tabs[3]:
        # Create sample portfolio performance
        portfolio_returns = (returns * current_weights).sum(axis=1)
        portfolio_values = (1 + portfolio_returns).cumprod() * portfolio_value

        # Benchmark (equal weight)
        benchmark_weights = pd.Series(
            np.ones(len(selected_assets)) / len(selected_assets), index=selected_assets
        )
        benchmark_returns = (returns * benchmark_weights).sum(axis=1)
        benchmark_values = (1 + benchmark_returns).cumprod() * portfolio_value

        render_performance_view(
            portfolio_values=portfolio_values,
            returns=portfolio_returns,
            benchmark_values=benchmark_values,
            benchmark_returns=benchmark_returns,
        )

    # Tab 5: Scenarios
    with tabs[4]:
        render_scenario_view(
            returns=returns,
            current_weights=current_weights,
            portfolio_value=portfolio_value,
        )

    # Tab 6: Recommendations
    with tabs[5]:
        opt_result = get_state("optimization_result")

        if opt_result is not None:
            optimal_weights = pd.Series(opt_result["weights"], index=selected_assets)

            render_recommendations_view(
                current_weights=current_weights,
                optimal_weights=optimal_weights,
                portfolio_value=portfolio_value,
                prices=prices.iloc[-1],
                expected_benefit=opt_result.get("objective_value", None),
            )
        else:
            st.info("Run optimization first to get rebalancing recommendations")

    # Footer
    st.markdown("---")
    st.markdown(f"Last updated: 2025Michael Gurule Data Science Portfolio")


if __name__ == "__main__":
    main()
