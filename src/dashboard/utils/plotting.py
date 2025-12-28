"""
Plotting utilities for dashboard visualizations.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


def create_portfolio_pie_chart(weights: pd.Series, title: str = "Portfolio Allocation") -> go.Figure:
    """
    Create pie chart of portfolio weights.
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights by asset
    title : str
        Chart title
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure(data=[go.Pie(
        labels=weights.index,
        values=weights.values,
        hole=0.4,
        textinfo='label+percent',
        textposition='auto',
        marker=dict(line=dict(color='white', width=2))
    )])
    
    fig.update_layout(
        title=title,
        showlegend=True,
        height=400
    )
    
    return fig


def create_performance_chart(portfolio_values: pd.Series,
                             benchmark_values: Optional[pd.Series] = None,
                             title: str = "Portfolio Performance") -> go.Figure:
    """
    Create performance line chart.
    
    Parameters
    ----------
    portfolio_values : pd.Series
        Portfolio value time series
    benchmark_values : pd.Series, optional
        Benchmark values
    title : str
        Chart title
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    # Normalize to base 100
    normalized_portfolio = (portfolio_values / portfolio_values.iloc[0]) * 100
    
    fig.add_trace(go.Scatter(
        x=normalized_portfolio.index,
        y=normalized_portfolio.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#1f77b4', width=2)
    ))
    
    if benchmark_values is not None:
        normalized_benchmark = (benchmark_values / benchmark_values.iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=normalized_benchmark.index,
            y=normalized_benchmark.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value (Base 100)",
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_drawdown_chart(prices: pd.Series, title: str = "Drawdown") -> go.Figure:
    """
    Create drawdown chart.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    title : str
        Chart title
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.3)',
        line=dict(color='#d62728', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=400
    )
    
    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dash'))
    
    return fig


def create_efficient_frontier(frontier_df: pd.DataFrame, 
                              current_portfolio: Optional[Dict] = None,
                              title: str = "Efficient Frontier") -> go.Figure:
    """
    Create efficient frontier visualization.
    
    Parameters
    ----------
    frontier_df : pd.DataFrame
        Efficient frontier data (volatility, return, sharpe)
    current_portfolio : Dict, optional
        Current portfolio stats {'volatility': x, 'return': y}
    title : str
        Chart title
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    # Frontier line
    fig.add_trace(go.Scatter(
        x=frontier_df['volatility'] * 100,
        y=frontier_df['return'] * 100,
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color='#1f77b4', width=3),
        marker=dict(
            size=8,
            color=frontier_df['sharpe'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Sharpe Ratio")
        ),
        hovertemplate='<b>Return:</b> %{y:.2f}%<br>' +
                     '<b>Volatility:</b> %{x:.2f}%<extra></extra>'
    ))
    
    # Current portfolio
    if current_portfolio:
        fig.add_trace(go.Scatter(
            x=[current_portfolio['volatility'] * 100],
            y=[current_portfolio['return'] * 100],
            mode='markers',
            name='Current Portfolio',
            marker=dict(size=15, color='red', symbol='star')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        hovermode='closest',
        height=500
    )
    
    return fig


def create_regime_probability_chart(probabilities: pd.DataFrame,
                                    title: str = "Regime Probabilities") -> go.Figure:
    """
    Create regime probability time series chart.
    
    Parameters
    ----------
    probabilities : pd.DataFrame
        Regime probabilities over time
    title : str
        Chart title
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    
    for i, col in enumerate(probabilities.columns):
        fig.add_trace(go.Scatter(
            x=probabilities.index,
            y=probabilities[col],
            mode='lines',
            name=col,
            line=dict(color=colors[i % len(colors)], width=2),
            stackgroup='one',
            groupnorm='percent'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Probability",
        hovermode='x unified',
        height=400
    )
    
    return fig


def create_risk_decomposition_chart(risk_contributions: pd.Series,
                                    title: str = "Risk Contribution by Asset") -> go.Figure:
    """
    Create risk contribution bar chart.
    
    Parameters
    ----------
    risk_contributions : pd.Series
        Risk contribution by asset
    title : str
        Chart title
    
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure(data=[
        go.Bar(
            x=risk_contributions.index,
            y=risk_contributions.values * 100,
            marker_color='#1f77b4',
            text=[f'{v*100:.1f}%' for v in risk_contributions.values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title="Risk Contribution (%)",
        height=400
    )
    
    return fig