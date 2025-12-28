"""
Portfolio overview component.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional

from src.dashboard.utils.plotting import create_portfolio_pie_chart, create_risk_decomposition_chart
from src.dashboard.utils.formatting import format_currency, format_percentage


def render_portfolio_view(
    weights: pd.Series,
    portfolio_value: float,
    prices: pd.DataFrame,
    expected_returns: Optional[pd.Series] = None,
    cov_matrix: Optional[pd.DataFrame] = None
):
    """
    Render portfolio overview.
    
    Parameters
    ----------
    weights : pd.Series
        Current portfolio weights
    portfolio_value : float
        Total portfolio value
    prices : pd.DataFrame
        Current prices
    expected_returns : pd.Series, optional
        Expected returns
    cov_matrix : pd.DataFrame, optional
        Covariance matrix
    """
    st.header("📊 Portfolio Overview")
    
    # Portfolio summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", format_currency(portfolio_value))
    
    with col2:
        n_positions = (weights > 0.001).sum()
        st.metric("Active Positions", f"{n_positions}")
    
    with col3:
        max_position = weights.max()
        st.metric("Largest Position", format_percentage(max_position))
    
    with col4:
        diversification = 1 / (weights ** 2).sum()  # Inverse HHI
        st.metric("Diversification", f"{diversification:.2f}")
    
    # Portfolio statistics
    if expected_returns is not None and cov_matrix is not None:
        st.subheader("Portfolio Statistics")
        
        portfolio_return = (expected_returns.values * weights.values).sum() * 252
        portfolio_variance = weights.values @ cov_matrix.values @ weights.values
        portfolio_vol = np.sqrt(portfolio_variance) * np.sqrt(252)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Expected Annual Return",
                format_percentage(portfolio_return),
                help="Projected annual return based on historical estimates"
            )
        
        with col2:
            st.metric(
                "Annual Volatility",
                format_percentage(portfolio_vol),
                help="Expected portfolio volatility (standard deviation)"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                help="Risk-adjusted return metric"
            )
    
    # Allocation visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Allocation by Asset")
        fig = create_portfolio_pie_chart(weights[weights > 0.001], "Current Allocation")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Position Details")
        
        position_data = pd.DataFrame({
            'Asset': weights.index,
            'Weight': weights.values,
            'Value': weights.values * portfolio_value,
            'Shares': (weights.values * portfolio_value / prices.iloc[-1].values)
        })
        
        position_data = position_data[position_data['Weight'] > 0.001].sort_values('Weight', ascending=False)
        position_data['Weight'] = position_data['Weight'].apply(lambda x: f"{x*100:.2f}%")
        position_data['Value'] = position_data['Value'].apply(lambda x: f"${x:,.0f}")
        position_data['Shares'] = position_data['Shares'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(position_data, hide_index=True, use_container_width=True)
    
    # Risk decomposition
    if cov_matrix is not None:
        st.subheader("Risk Decomposition")
        
        # Calculate marginal risk contribution
        marginal_risk = cov_matrix @ weights
        risk_contribution = weights * marginal_risk
        risk_contribution = risk_contribution / risk_contribution.sum()
        
        fig = create_risk_decomposition_chart(
            risk_contribution[risk_contribution > 0.001],
            "Risk Contribution by Asset"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(
            "**Risk Contribution** shows how much each asset contributes to total portfolio risk. "
            "A well-diversified portfolio has relatively balanced risk contributions."
        )


def render_holdings_table(weights: pd.Series, 
                          portfolio_value: float,
                          prices: pd.Series) -> pd.DataFrame:
    """
    Create detailed holdings table.
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights
    portfolio_value : float
        Total value
    prices : pd.Series
        Current prices
    
    Returns
    -------
    pd.DataFrame
        Holdings table
    """
    holdings = pd.DataFrame({
        'Asset': weights.index,
        'Weight (%)': weights.values * 100,
        'Value ($)': weights.values * portfolio_value,
        'Price ($)': prices.values,
        'Shares': (weights.values * portfolio_value) / prices.values
    })
    
    holdings = holdings[holdings['Weight (%)'] > 0.1].sort_values('Weight (%)', ascending=False)
    
    return holdings