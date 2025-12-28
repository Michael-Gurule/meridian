"""
Performance analytics component.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

from src.backtesting.metrics import PerformanceMetrics
from src.dashboard.utils.plotting import create_performance_chart, create_drawdown_chart
from src.dashboard.utils.formatting import format_percentage, format_currency


def render_performance_view(
    portfolio_values: pd.Series,
    returns: pd.Series,
    benchmark_values: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None
):
    """
    Render performance analytics.
    
    Parameters
    ----------
    portfolio_values : pd.Series
        Portfolio value time series
    returns : pd.Series
        Portfolio returns
    benchmark_values : pd.Series, optional
        Benchmark values
    benchmark_returns : pd.Series, optional
        Benchmark returns
    """
    st.header("📈 Performance Analytics")
    
    # Calculate metrics
    metrics_calc = PerformanceMetrics(risk_free_rate=0.0)
    metrics = metrics_calc.calculate_all_metrics(
        returns=returns,
        prices=portfolio_values,
        benchmark_returns=benchmark_returns
    )
    
    # Key metrics
    st.subheader("Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            format_percentage(metrics['total_return']),
            help="Cumulative return over period"
        )
    
    with col2:
        st.metric(
            "Annual Return",
            format_percentage(metrics['annual_return']),
            help="Annualized return"
        )
    
    with col3:
        st.metric(
            "Volatility",
            format_percentage(metrics['volatility']),
            help="Annualized volatility"
        )
    
    with col4:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            help="Risk-adjusted return"
        )
    
    # Risk metrics
    st.subheader("Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Max Drawdown",
            format_percentage(metrics['max_drawdown']),
            help="Largest peak-to-trough decline"
        )
    
    with col2:
        st.metric(
            "Sortino Ratio",
            f"{metrics['sortino_ratio']:.2f}",
            help="Downside risk-adjusted return"
        )
    
    with col3:
        st.metric(
            "VaR (95%)",
            format_percentage(metrics['var_95']),
            help="Value at Risk at 95% confidence"
        )
    
    with col4:
        st.metric(
            "CVaR (95%)",
            format_percentage(metrics['cvar_95']),
            help="Conditional Value at Risk"
        )
    
    # Performance chart
    st.subheader("Cumulative Performance")
    fig = create_performance_chart(
        portfolio_values,
        benchmark_values,
        "Portfolio vs Benchmark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown chart
    st.subheader("Drawdown Analysis")
    fig = create_drawdown_chart(portfolio_values, "Portfolio Drawdown")
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Return Statistics")
        
        return_stats = pd.DataFrame({
            'Metric': [
                'Best Day',
                'Worst Day',
                'Avg Daily Return',
                'Win Rate',
                'Profit Factor'
            ],
            'Value': [
                format_percentage(metrics['best_day']),
                format_percentage(metrics['worst_day']),
                format_percentage(metrics['avg_daily_return']),
                format_percentage(metrics['win_rate']),
                f"{metrics['profit_factor']:.2f}"
            ]
        })
        
        st.dataframe(return_stats, hide_index=True, use_container_width=True)
    
    with col2:
        if benchmark_returns is not None:
            st.subheader("Benchmark Comparison")
            
            comparison = pd.DataFrame({
                'Metric': [
                    'Tracking Error',
                    'Information Ratio',
                    'Beta',
                    'Alpha'
                ],
                'Value': [
                    format_percentage(metrics['tracking_error']),
                    f"{metrics['information_ratio']:.2f}",
                    f"{metrics['beta']:.2f}",
                    format_percentage(metrics['alpha'])
                ]
            })
            
            st.dataframe(comparison, hide_index=True, use_container_width=True)