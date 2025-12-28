"""
Scenario analysis component.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List

from src.optimization.optimizer import PortfolioOptimizer
from src.models.covariance import CovarianceEstimator
from src.dashboard.utils.formatting import format_percentage, format_currency


def render_scenario_view(
    returns: pd.DataFrame,
    current_weights: pd.Series,
    portfolio_value: float
):
    """
    Render scenario analysis interface.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns
    current_weights : pd.Series
        Current portfolio weights
    portfolio_value : float
        Portfolio value
    """
    st.header("🎯 Scenario Analysis")
    
    st.write(
        "Analyze portfolio performance under different market scenarios. "
        "Test stress scenarios, custom shocks, or historical events."
    )
    
    # Scenario type selection
    scenario_type = st.selectbox(
        "Scenario Type",
        options=['stress_test', 'custom_shock', 'historical_event', 'monte_carlo'],
        format_func=lambda x: {
            'stress_test': 'Stress Test (Market Crashes)',
            'custom_shock': 'Custom Asset Shocks',
            'historical_event': 'Historical Event Replay',
            'monte_carlo': 'Monte Carlo Simulation'
        }[x]
    )
    
    if scenario_type == 'stress_test':
        render_stress_test(returns, current_weights, portfolio_value)
    
    elif scenario_type == 'custom_shock':
        render_custom_shock(returns, current_weights, portfolio_value)
    
    elif scenario_type == 'historical_event':
        render_historical_event(returns, current_weights, portfolio_value)
    
    elif scenario_type == 'monte_carlo':
        render_monte_carlo(returns, current_weights, portfolio_value)


def render_stress_test(
    returns: pd.DataFrame,
    current_weights: pd.Series,
    portfolio_value: float
):
    """
    Render stress test scenarios.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data
    current_weights : pd.Series
        Portfolio weights
    portfolio_value : float
        Portfolio value
    """
    st.subheader("Stress Test Scenarios")
    
    # Pre-defined scenarios
    scenarios = {
        'Market Crash (-20%)': {'equities': -0.20, 'bonds': -0.05, 'gold': 0.10, 'crypto': -0.30},
        'Severe Crash (-40%)': {'equities': -0.40, 'bonds': -0.10, 'gold': 0.20, 'crypto': -0.50},
        'Bond Selloff': {'equities': -0.10, 'bonds': -0.15, 'gold': 0.05, 'crypto': -0.15},
        'Inflation Surge': {'equities': -0.15, 'bonds': -0.20, 'gold': 0.25, 'crypto': 0.10},
        'Flight to Safety': {'equities': -0.15, 'bonds': 0.10, 'gold': 0.15, 'crypto': -0.20}
    }
    
    selected_scenario = st.selectbox(
        "Select Stress Scenario",
        options=list(scenarios.keys())
    )
    
    if st.button("🔬 Run Stress Test", use_container_width=True):
        shock = scenarios[selected_scenario]
        
        # Apply shocks
        shocked_returns = apply_shock_to_returns(returns, current_weights, shock)
        
        # Calculate impact
        portfolio_impact = shocked_returns @ current_weights.values
        portfolio_change = portfolio_value * portfolio_impact
        new_value = portfolio_value + portfolio_change
        
        # Display results
        st.subheader("Stress Test Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Value",
                format_currency(portfolio_value)
            )
        
        with col2:
            st.metric(
                "Stressed Value",
                format_currency(new_value),
                delta=format_currency(portfolio_change)
            )
        
        with col3:
            impact_pct = portfolio_impact
            st.metric(
                "Impact",
                format_percentage(impact_pct),
                delta=None
            )
        
        # Asset-level impact
        st.subheader("Impact by Asset")
        
        asset_impacts = pd.DataFrame({
            'Asset': current_weights.index,
            'Weight': current_weights.values,
            'Shock': [shock.get(map_asset_to_class(asset), 0) for asset in current_weights.index],
            'Impact': current_weights.values * [shock.get(map_asset_to_class(asset), 0) for asset in current_weights.index]
        })
        
        asset_impacts = asset_impacts.sort_values('Impact', ascending=True)
        asset_impacts['Weight'] = asset_impacts['Weight'].apply(lambda x: f"{x*100:.2f}%")
        asset_impacts['Shock'] = asset_impacts['Shock'].apply(lambda x: f"{x*100:.1f}%")
        asset_impacts['Impact'] = asset_impacts['Impact'].apply(lambda x: f"{x*100:.2f}%")
        
        st.dataframe(asset_impacts, hide_index=True, use_container_width=True)
        
        # Interpretation
        if portfolio_impact < -0.15:
            st.error(f"⚠️ **High Risk**: Portfolio would lose {abs(portfolio_impact):.1%} in this scenario.")
        elif portfolio_impact < -0.05:
            st.warning(f"⚠️ **Moderate Risk**: Portfolio would lose {abs(portfolio_impact):.1%} in this scenario.")
        else:
            st.success(f"✅ **Low Risk**: Portfolio is well-positioned for this scenario.")


def render_custom_shock(
    returns: pd.DataFrame,
    current_weights: pd.Series,
    portfolio_value: float
):
    """
    Render custom shock builder.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data
    current_weights : pd.Series
        Portfolio weights
    portfolio_value : float
        Portfolio value
    """
    st.subheader("Custom Asset Shocks")
    
    st.write("Define custom return shocks for each asset:")
    
    # Create input fields for each asset
    shocks = {}
    
    cols = st.columns(3)
    for idx, asset in enumerate(current_weights.index):
        with cols[idx % 3]:
            shock = st.slider(
                f"{asset} Return",
                min_value=-50.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                format="%g%%",
                key=f"shock_{asset}"
            )
            shocks[asset] = shock / 100
    
    if st.button("📊 Calculate Impact", use_container_width=True):
        # Calculate portfolio impact
        shock_vector = np.array([shocks[asset] for asset in current_weights.index])
        portfolio_impact = (shock_vector * current_weights.values).sum()
        
        portfolio_change = portfolio_value * portfolio_impact
        new_value = portfolio_value + portfolio_change
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Value", format_currency(portfolio_value))
        
        with col2:
            st.metric(
                "New Value",
                format_currency(new_value),
                delta=format_currency(portfolio_change)
            )
        
        with col3:
            st.metric("Total Impact", format_percentage(portfolio_impact))


def render_historical_event(
    returns: pd.DataFrame,
    current_weights: pd.Series,
    portfolio_value: float
):
    """
    Render historical event replay.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data
    current_weights : pd.Series
        Portfolio weights
    portfolio_value : float
        Portfolio value
    """
    st.subheader("Historical Event Replay")
    
    # Define historical events (approximate dates)
    events = {
        '2020 COVID Crash (Feb-Mar)': ('2020-02-01', '2020-03-31'),
        '2018 Q4 Selloff': ('2018-10-01', '2018-12-31'),
        'Brexit Vote (2016)': ('2016-06-01', '2016-07-31'),
        '2015 China Devaluation': ('2015-08-01', '2015-09-30')
    }
    
    selected_event = st.selectbox(
        "Select Historical Event",
        options=list(events.keys())
    )
    
    if st.button("📜 Replay Event", use_container_width=True):
        start_date, end_date = events[selected_event]
        
        # Filter returns to event period
        event_returns = returns.loc[start_date:end_date]
        
        if len(event_returns) > 0:
            # Calculate cumulative returns during event
            cumulative_returns = (1 + event_returns).prod() - 1
            
            # Portfolio impact
            portfolio_impact = (cumulative_returns * current_weights).sum()
            portfolio_change = portfolio_value * portfolio_impact
            new_value = portfolio_value + portfolio_change
            
            # Display results
            st.subheader("Event Impact")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Event Duration", f"{len(event_returns)} days")
            
            with col2:
                st.metric(
                    "Portfolio Impact",
                    format_percentage(portfolio_impact)
                )
            
            with col3:
                st.metric(
                    "Value Change",
                    format_currency(portfolio_change)
                )
            
            # Asset performance
            st.subheader("Asset Performance During Event")
            
            asset_perf = pd.DataFrame({
                'Asset': cumulative_returns.index,
                'Return': cumulative_returns.values,
                'Weight': current_weights.values,
                'Contribution': cumulative_returns.values * current_weights.values
            })
            
            asset_perf = asset_perf.sort_values('Return', ascending=False)
            asset_perf['Return'] = asset_perf['Return'].apply(lambda x: f"{x*100:.2f}%")
            asset_perf['Weight'] = asset_perf['Weight'].apply(lambda x: f"{x*100:.2f}%")
            asset_perf['Contribution'] = asset_perf['Contribution'].apply(lambda x: f"{x*100:.2f}%")
            
            st.dataframe(asset_perf, hide_index=True, use_container_width=True)
        else:
            st.warning("No data available for selected event period.")


def render_monte_carlo(
    returns: pd.DataFrame,
    current_weights: pd.Series,
    portfolio_value: float
):
    """
    Render Monte Carlo simulation.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data
    current_weights : pd.Series
        Portfolio weights
    portfolio_value : float
        Portfolio value
    """
    st.subheader("Monte Carlo Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_simulations = st.slider(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
    
    with col2:
        time_horizon = st.slider(
            "Time Horizon (days)",
            min_value=20,
            max_value=252,
            value=60,
            step=10
        )
    
    if st.button("🎲 Run Simulation", use_container_width=True):
        with st.spinner("Running Monte Carlo simulation..."):
            # Estimate parameters
            mean_returns = returns.mean().values
            cov_matrix = returns.cov().values
            
            # Run simulation
            final_values = []
            
            for _ in range(n_simulations):
                # Generate random returns
                random_returns = np.random.multivariate_normal(
                    mean_returns,
                    cov_matrix,
                    size=time_horizon
                )
                
                # Calculate portfolio returns
                portfolio_returns = random_returns @ current_weights.values
                
                # Final value
                final_value = portfolio_value * (1 + portfolio_returns).prod()
                final_values.append(final_value)
            
            final_values = np.array(final_values)
            
            # Calculate statistics
            mean_value = final_values.mean()
            median_value = np.median(final_values)
            percentile_5 = np.percentile(final_values, 5)
            percentile_95 = np.percentile(final_values, 95)
            
            # Display results
            st.subheader("Simulation Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Mean Value",
                    format_currency(mean_value),
                    delta=format_currency(mean_value - portfolio_value)
                )
            
            with col2:
                st.metric(
                    "Median Value",
                    format_currency(median_value)
                )
            
            with col3:
                st.metric(
                    "5th Percentile",
                    format_currency(percentile_5),
                    help="95% chance of exceeding this value"
                )
            
            with col4:
                st.metric(
                    "95th Percentile",
                    format_currency(percentile_95),
                    help="5% chance of exceeding this value"
                )
            
            # Distribution chart
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=final_values,
                nbinsx=50,
                name='Simulated Values',
                marker_color='#1f77b4'
            ))
            
            fig.add_vline(
                x=portfolio_value,
                line=dict(color='red', width=2, dash='dash'),
                annotation_text="Current Value"
            )
            
            fig.update_layout(
                title=f"Distribution of Portfolio Values ({time_horizon} days)",
                xaxis_title="Portfolio Value ($)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics
            prob_loss = (final_values < portfolio_value).sum() / n_simulations
            avg_loss = final_values[final_values < portfolio_value].mean() if prob_loss > 0 else portfolio_value
            
            st.info(
                f"**Risk Assessment:** {prob_loss:.1%} probability of loss. "
                f"Average value in loss scenarios: {format_currency(avg_loss)}"
            )


def apply_shock_to_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
    shock: Dict[str, float]
) -> pd.Series:
    """
    Apply shock to asset returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns
    weights : pd.Series
        Portfolio weights
    shock : Dict[str, float]
        Shock by asset class
    
    Returns
    -------
    pd.Series
        Shocked returns
    """
    shocked_returns = pd.Series(index=returns.columns, dtype=float)
    
    for asset in returns.columns:
        asset_class = map_asset_to_class(asset)
        shocked_returns[asset] = shock.get(asset_class, 0.0)
    
    return shocked_returns


def map_asset_to_class(asset: str) -> str:
    """
    Map asset ticker to asset class.
    
    Parameters
    ----------
    asset : str
        Asset ticker
    
    Returns
    -------
    str
        Asset class
    """
    equity_tickers = ['SPY', 'QQQ', 'VTI', 'IWM', 'EFA', 'EEM']
    bond_tickers = ['TLT', 'IEF', 'LQD', 'AGG', 'BND']
    gold_tickers = ['GLD', 'IAU']
    crypto_tickers = ['BTC-USD', 'ETH-USD']
    
    if asset in equity_tickers:
        return 'equities'
    elif asset in bond_tickers:
        return 'bonds'
    elif asset in gold_tickers:
        return 'gold'
    elif asset in crypto_tickers:
        return 'crypto'
    else:
        return 'other'