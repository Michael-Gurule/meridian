"""
Plotting utilities for dashboard visualizations.
Dark theme with emerald accents.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, List, Dict

from src.dashboard.styles import COLORS, get_chart_colors


def apply_dark_theme(fig: go.Figure, **kwargs) -> go.Figure:
    """
    Apply dark theme to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        The figure to style
    **kwargs
        Additional layout arguments (title, height, hovermode, etc.)

    Returns
    -------
    go.Figure
        Styled figure
    """
    # Base dark theme settings
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            color=COLORS['text_secondary'],
            family='Inter, -apple-system, sans-serif',
            size=12
        ),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text_secondary']),
            bordercolor='rgba(0,0,0,0)'
        ),
        hoverlabel=dict(
            bgcolor=COLORS['bg_card'],
            font=dict(color=COLORS['text_primary']),
            bordercolor=COLORS['primary']
        ),
        margin=dict(l=60, r=30, t=50, b=50),
        **kwargs
    )

    # Update axes styling (preserves existing title if set)
    fig.update_xaxes(
        gridcolor=COLORS['bg_elevated'],
        linecolor=COLORS['bg_elevated'],
        tickfont=dict(color=COLORS['text_secondary']),
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        title_font=dict(color=COLORS['text_secondary'])
    )

    fig.update_yaxes(
        gridcolor=COLORS['bg_elevated'],
        linecolor=COLORS['bg_elevated'],
        tickfont=dict(color=COLORS['text_secondary']),
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        title_font=dict(color=COLORS['text_secondary'])
    )

    return fig


def create_portfolio_pie_chart(weights: pd.Series, title: str = "Portfolio Allocation") -> go.Figure:
    """
    Create donut chart of portfolio weights.

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
    colors = get_chart_colors()

    fig = go.Figure(data=[go.Pie(
        labels=weights.index,
        values=weights.values,
        hole=0.55,
        textinfo='label+percent',
        textposition='outside',
        textfont={'color': COLORS['text_primary'], 'size': 11},
        marker=dict(
            colors=colors[:len(weights)],
            line=dict(color=COLORS['bg_dark'], width=2)
        ),
        hovertemplate='<b>%{label}</b><br>Weight: %{percent}<br>Value: %{value:.2%}<extra></extra>'
    )])

    # Add center text
    fig.add_annotation(
        text=f"<b>{len(weights)}</b><br>Assets",
        x=0.5, y=0.5,
        font=dict(size=16, color=COLORS['text_primary']),
        showarrow=False
    )

    fig.update_layout(
        title=title,
        showlegend=True,
        height=400,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text_secondary']}
    )

    return fig


def create_performance_chart(portfolio_values: pd.Series,
                             benchmark_values: Optional[pd.Series] = None,
                             title: str = "Portfolio Performance") -> go.Figure:
    """
    Create performance line chart with gradient fills.

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

    # Portfolio line with gradient fill
    fig.add_trace(go.Scatter(
        x=normalized_portfolio.index,
        y=normalized_portfolio.values,
        mode='lines',
        name='Portfolio',
        line=dict(color=COLORS['primary'], width=2.5),
        fill='tozeroy',
        fillcolor=f"rgba(16, 185, 129, 0.1)",
        hovertemplate='<b>Portfolio</b><br>Date: %{x}<br>Value: %{y:.1f}<extra></extra>'
    ))

    if benchmark_values is not None:
        normalized_benchmark = (benchmark_values / benchmark_values.iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=normalized_benchmark.index,
            y=normalized_benchmark.values,
            mode='lines',
            name='Benchmark',
            line=dict(color=COLORS['text_muted'], width=2, dash='dash'),
            hovertemplate='<b>Benchmark</b><br>Date: %{x}<br>Value: %{y:.1f}<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Value (Base 100)",
    )

    return apply_dark_theme(fig, title=title, height=500, hovermode='x unified')


def create_drawdown_chart(prices: pd.Series, title: str = "Drawdown") -> go.Figure:
    """
    Create drawdown chart with gradient fill.

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
        fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(color=COLORS['danger'], width=2),
        hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))

    fig.add_hline(y=0, line=dict(color=COLORS['text_muted'], width=1, dash='dash'))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
    )

    return apply_dark_theme(fig, title=title, height=400, hovermode='x unified')


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

    # Frontier line with Sharpe color scale
    fig.add_trace(go.Scatter(
        x=frontier_df['volatility'] * 100,
        y=frontier_df['return'] * 100,
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color=COLORS['primary'], width=3),
        marker=dict(
            size=10,
            color=frontier_df['sharpe'],
            colorscale=[
                [0, COLORS['danger']],
                [0.5, COLORS['warning']],
                [1, COLORS['success']]
            ],
            showscale=True,
            colorbar=dict(
                title="Sharpe",
                titlefont=dict(color=COLORS['text_secondary']),
                tickfont=dict(color=COLORS['text_secondary'])
            ),
            line=dict(color=COLORS['bg_dark'], width=1)
        ),
        hovertemplate='<b>Portfolio</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>'
    ))

    # Current portfolio
    if current_portfolio:
        fig.add_trace(go.Scatter(
            x=[current_portfolio['volatility'] * 100],
            y=[current_portfolio['return'] * 100],
            mode='markers',
            name='Current Portfolio',
            marker=dict(
                size=18,
                color=COLORS['warning'],
                symbol='star',
                line=dict(color=COLORS['bg_dark'], width=2)
            ),
            hovertemplate='<b>Current</b><br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
    )

    return apply_dark_theme(fig, title=title, height=500, hovermode='closest')


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

    regime_colors = [COLORS['success'], COLORS['danger'], COLORS['warning']]

    for i, col in enumerate(probabilities.columns):
        fig.add_trace(go.Scatter(
            x=probabilities.index,
            y=probabilities[col],
            mode='lines',
            name=col.replace('_', ' ').title(),
            line=dict(color=regime_colors[i % len(regime_colors)], width=2),
            stackgroup='one',
            groupnorm='percent',
            hovertemplate=f'<b>{col}</b><br>Date: %{{x}}<br>Probability: %{{y:.1%}}<extra></extra>'
        ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Probability (%)",
    )

    return apply_dark_theme(fig, title=title, height=400, hovermode='x unified')


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
    # Sort by contribution
    risk_contributions = risk_contributions.sort_values(ascending=True)

    # Color based on contribution level
    colors = [
        COLORS['success'] if v < 0.15 else
        COLORS['warning'] if v < 0.25 else
        COLORS['danger']
        for v in risk_contributions.values
    ]

    fig = go.Figure(data=[
        go.Bar(
            y=risk_contributions.index,
            x=risk_contributions.values * 100,
            orientation='h',
            marker_color=colors,
            text=[f'{v*100:.1f}%' for v in risk_contributions.values],
            textposition='outside',
            textfont=dict(color=COLORS['text_primary']),
            hovertemplate='<b>%{y}</b><br>Risk Contribution: %{x:.1f}%<extra></extra>'
        )
    ])

    fig.update_layout(
        xaxis_title="Risk Contribution (%)",
        yaxis_title="",
    )

    return apply_dark_theme(fig, title=title, height=max(300, len(risk_contributions) * 35))


def create_correlation_heatmap(correlation_matrix: pd.DataFrame,
                               title: str = "Correlation Matrix") -> go.Figure:
    """
    Create correlation heatmap.

    Parameters
    ----------
    correlation_matrix : pd.DataFrame
        Correlation matrix
    title : str
        Chart title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale=[
            [0, COLORS['danger']],
            [0.5, COLORS['bg_card']],
            [1, COLORS['success']]
        ],
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10, "color": COLORS['text_primary']},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>',
        colorbar=dict(
            title="Correlation",
            titlefont=dict(color=COLORS['text_secondary']),
            tickfont=dict(color=COLORS['text_secondary'])
        )
    ))

    return apply_dark_theme(fig, title=title, height=500)


def create_returns_distribution(returns: pd.Series,
                                title: str = "Returns Distribution") -> go.Figure:
    """
    Create returns histogram with normal overlay.

    Parameters
    ----------
    returns : pd.Series
        Return series
    title : str
        Chart title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=returns.values * 100,
        nbinsx=50,
        name='Returns',
        marker_color=COLORS['primary'],
        opacity=0.7,
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))

    # Add mean line
    mean_return = returns.mean() * 100
    fig.add_vline(
        x=mean_return,
        line=dict(color=COLORS['warning'], width=2, dash='dash'),
        annotation_text=f"Mean: {mean_return:.2f}%",
        annotation_font=dict(color=COLORS['text_primary'])
    )

    fig.update_layout(
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
    )

    return apply_dark_theme(fig, title=title, height=400)


def create_gauge_chart(value: float, title: str, min_val: float = 0, max_val: float = 1,
                       thresholds: Optional[List[float]] = None) -> go.Figure:
    """
    Create a gauge chart for metrics.

    Parameters
    ----------
    value : float
        Current value
    title : str
        Gauge title
    min_val : float
        Minimum value
    max_val : float
        Maximum value
    thresholds : list, optional
        Threshold values for color zones

    Returns
    -------
    go.Figure
        Plotly figure
    """
    if thresholds is None:
        thresholds = [(max_val - min_val) * 0.33 + min_val,
                      (max_val - min_val) * 0.66 + min_val]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': COLORS['text_primary']}},
        number={'font': {'color': COLORS['text_primary']}},
        gauge={
            'axis': {
                'range': [min_val, max_val],
                'tickcolor': COLORS['text_secondary'],
                'tickfont': {'color': COLORS['text_secondary']}
            },
            'bar': {'color': COLORS['primary']},
            'bgcolor': COLORS['bg_card'],
            'borderwidth': 0,
            'steps': [
                {'range': [min_val, thresholds[0]], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [thresholds[0], thresholds[1]], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [thresholds[1], max_val], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': COLORS['text_primary'], 'width': 2},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['text_secondary']}
    )

    return fig
