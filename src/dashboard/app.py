"""
MERIDIAN Portfolio Optimization Dashboard
Modern dark theme with emerald accents.
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
from src.dashboard.styles import inject_custom_css, render_logo_header, COLORS
from src.dashboard.utils.state import initialize_session_state, get_state, set_state
from src.dashboard.demo_data import load_demo_portfolio, get_feature_highlights, get_demo_summary_stats
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
    page_title=f"{CONFIG.app_title} | {CONFIG.app_subtitle}",
    page_icon="📊",
    layout=CONFIG.layout,
    initial_sidebar_state="expanded",
)

# Inject custom CSS
inject_custom_css()

# Initialize session state
initialize_session_state()


def load_data():
    """Load market data."""
    storage = DataStorage()
    assets = storage.list_available_assets(data_type="processed")
    return storage, assets


def render_welcome_screen():
    """Render the welcome screen when no data is loaded."""

    # Hero section with logo
    logo_path = project_root / CONFIG.logo_header
    render_logo_header(str(logo_path), CONFIG.app_title)

    st.markdown(
        f"""
        <p style="text-align: center; color: {COLORS['text_secondary']}; font-size: 1.1rem; margin-bottom: 2rem;">
            Production-grade portfolio optimization with regime detection and scenario analysis
        </p>
        """,
        unsafe_allow_html=True
    )

    # Feature highlights
    features = get_feature_highlights()

    cols = st.columns(4)
    for i, feature in enumerate(features):
        with cols[i]:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="icon">{feature['icon']}</div>
                    <h4>{feature['title']}</h4>
                    <p>{feature['description']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Demo button - centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Load Demo Portfolio", type="primary", use_container_width=True):
            with st.spinner("Loading demo data..."):
                demo_data = load_demo_portfolio()
                if demo_data:
                    set_state("prices", demo_data['prices'])
                    set_state("returns", demo_data['returns'])
                    set_state("selected_assets", demo_data['assets'])
                    set_state("current_weights", demo_data['current_weights'])
                    set_state("portfolio_value", demo_data['portfolio_value'])
                    st.rerun()
                else:
                    st.error("Unable to load demo data. Please use the sidebar to load data manually.")

    # Summary stats
    st.markdown("<br>", unsafe_allow_html=True)
    stats = get_demo_summary_stats()

    st.markdown(
        f"""
        <div style="text-align: center; color: {COLORS['text_muted']}; font-size: 0.9rem;">
            <span style="margin: 0 1rem;">📊 {stats['total_assets']} Assets</span>
            <span style="margin: 0 1rem;">📅 {stats['data_years']} Years of Data</span>
            <span style="margin: 0 1rem;">🏷️ {stats['asset_classes']} Asset Classes</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():
    """Render the sidebar with settings."""

    with st.sidebar:
        # Sidebar header with mini logo
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem; margin-right: 0.5rem;">📊</span>
                <span style="font-weight: 600; color: {COLORS['text_primary']};">Settings</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Load data
        storage, available_assets = load_data()

        # Data status indicator
        prices = get_state("prices")
        if prices is not None:
            st.markdown(
                f"""
                <div style="background: {COLORS['bg_card']}; border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;">
                    <div style="display: flex; align-items: center;">
                        <span class="status-indicator status-active"></span>
                        <span style="color: {COLORS['success']}; font-weight: 500;">Data Loaded</span>
                    </div>
                    <div style="color: {COLORS['text_secondary']}; font-size: 0.85rem; margin-top: 0.5rem;">
                        {len(prices)} days | {len(prices.columns)} assets
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Asset selection
        st.subheader("Asset Universe")

        default_assets = CONFIG.demo_assets
        current_selection = get_state("selected_assets") or [a for a in default_assets if a in available_assets]

        selected_assets = st.multiselect(
            "Select Assets",
            options=available_assets,
            default=current_selection if all(a in available_assets for a in current_selection) else [],
            help="Choose assets for portfolio optimization",
        )

        if len(selected_assets) < 2:
            st.warning("Select at least 2 assets")
            return None, None

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
        if st.button("📥 Load Data", type="primary", use_container_width=True):
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

                    st.success(f"Loaded {len(prices)} days for {len(selected_assets)} assets")

                except Exception as e:
                    st.error(f"Failed to load data: {str(e)}")

        return storage, available_assets


def main():
    """Main dashboard application."""

    # Render sidebar
    render_sidebar()

    # Get data from session state
    prices = get_state("prices")
    returns = get_state("returns")
    selected_assets = get_state("selected_assets")
    portfolio_value = get_state("portfolio_value", CONFIG.default_portfolio_value)

    # Show welcome screen if no data loaded
    if prices is None or returns is None:
        render_welcome_screen()
        return

    # Header for data-loaded state
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
            <h1 style="margin: 0;">📊 {CONFIG.app_title}</h1>
            <div style="color: {COLORS['text_secondary']};">
                {prices.index.min().strftime('%b %Y')} - {prices.index.max().strftime('%b %Y')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Navigation tabs
    tabs = st.tabs([
        "📊 Portfolio",
        "⚙️ Optimization",
        "🔮 Regime",
        "📈 Performance",
        "🎯 Scenarios",
        "💡 Recommendations",
    ])

    # Initialize current weights if not set OR if assets changed
    current_state_weights = get_state("current_weights")
    if current_state_weights is None or set(current_state_weights.index) != set(returns.columns):
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
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✅ Apply Optimal Weights", use_container_width=True):
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
    st.markdown(
        f"""
        <div style="text-align: center; color: {COLORS['text_muted']}; font-size: 0.85rem;">
            <strong>MERIDIAN</strong> | Portfolio Optimization System |
            Built by <a href="https://www.linkedin.com/in/michael-j-gurule-447aa2134"
            style="color: {COLORS['primary']};">Michael Gurule</a>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
