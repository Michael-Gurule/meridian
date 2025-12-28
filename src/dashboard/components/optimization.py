"""
Optimization interface component.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

from src.optimization.optimizer import PortfolioOptimizer
from src.models.covariance import CovarianceEstimator
from src.models.returns import ReturnEstimator
from src.dashboard.utils.plotting import create_efficient_frontier
from src.dashboard.utils.formatting import format_percentage, format_currency
from src.dashboard.utils.state import set_state


def render_optimization_view(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    current_weights: Optional[pd.Series] = None
):
    """
    Render optimization interface.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    returns : pd.DataFrame
        Returns data
    current_weights : pd.Series, optional
        Current portfolio weights
    """
    st.header("⚙️ Portfolio Optimization")
    
    # Optimization settings
    st.subheader("Optimization Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        objective = st.selectbox(
            "Optimization Objective",
            options=['mean_variance', 'min_variance', 'risk_parity', 'max_diversification'],
            format_func=lambda x: {
                'mean_variance': 'Mean-Variance (Sharpe)',
                'min_variance': 'Minimum Variance',
                'risk_parity': 'Risk Parity',
                'max_diversification': 'Maximum Diversification'
            }[x],
            help="Choose the portfolio optimization objective"
        )
        
        cov_method = st.selectbox(
            "Covariance Estimation",
            options=['sample', 'ledoit_wolf', 'ewma'],
            format_func=lambda x: {
                'sample': 'Sample Covariance',
                'ledoit_wolf': 'Ledoit-Wolf Shrinkage',
                'ewma': 'Exponentially Weighted'
            }[x],
            help="Method for estimating asset covariances"
        )
    
    with col2:
        risk_aversion = st.slider(
            "Risk Aversion (λ)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Higher values prioritize risk reduction over returns"
        )
        
        lookback_days = st.slider(
            "Lookback Window (days)",
            min_value=60,
            max_value=756,
            value=252,
            step=30,
            help="Historical window for parameter estimation"
        )
    
    # Constraints
    st.subheader("Constraints")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        long_only = st.checkbox("Long Only", value=True, help="No short positions")
        
    with col2:
        max_position = st.slider(
            "Max Position Size",
            min_value=0.05,
            max_value=1.0,
            value=0.40,
            step=0.05,
            help="Maximum weight per asset"
        )
    
    with col3:
        max_turnover = st.slider(
            "Max Turnover",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Maximum portfolio turnover allowed"
        ) if current_weights is not None else None
    
    # Run optimization
    if st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True):
        with st.spinner("Optimizing portfolio..."):
            try:
                # Prepare data
                recent_returns = returns.iloc[-lookback_days:]
                
                # Estimate parameters
                cov_estimator = CovarianceEstimator(method=cov_method)
                return_estimator = ReturnEstimator(method='historical')
                
                cov_matrix = cov_estimator.estimate(recent_returns)
                expected_returns = return_estimator.estimate(recent_returns, annualize=False)
                
                # Build optimizer
                constraints = {
                    'long_only': long_only,
                    'weight_bounds': (0.0, max_position) if long_only else (-0.3, max_position)
                }
                
                if max_turnover is not None and current_weights is not None:
                    constraints['max_turnover'] = max_turnover
                
                optimizer = PortfolioOptimizer(
                    objective_type=objective,
                    constraints=constraints
                )
                
                # Optimize
                result = optimizer.optimize(
                    expected_returns=expected_returns.values / 252,
                    cov_matrix=cov_matrix.values,
                    current_weights=current_weights.values if current_weights is not None else None,
                    risk_aversion=risk_aversion
                )
                
                # Store result
                set_state('optimization_result', result)
                set_state('optimization_params', {
                    'objective': objective,
                    'cov_method': cov_method,
                    'risk_aversion': risk_aversion
                })
                
                st.success("✅ Optimization complete!")
                
                # Display results
                display_optimization_results(result, expected_returns.index, cov_matrix)
                
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")


def display_optimization_results(
    result: dict,
    asset_names: pd.Index,
    cov_matrix: pd.DataFrame
):
    """
    Display optimization results.
    
    Parameters
    ----------
    result : dict
        Optimization result
    asset_names : pd.Index
        Asset names
    cov_matrix : pd.DataFrame
        Covariance matrix
    """
    st.subheader("Optimization Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Expected Return",
            format_percentage(result['expected_return'] * 252),
            help="Annualized expected return"
        )
    
    with col2:
        st.metric(
            "Volatility",
            format_percentage(result['volatility'] * np.sqrt(252)),
            help="Annualized volatility"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{result['sharpe_ratio']:.2f}",
            help="Risk-adjusted return"
        )
    
    with col4:
        st.metric(
            "Turnover",
            format_percentage(result['turnover']),
            help="Portfolio turnover"
        )
    
    # Optimal weights
    st.subheader("Optimal Allocation")
    
    optimal_weights = pd.Series(result['weights'], index=asset_names)
    optimal_weights = optimal_weights[optimal_weights > 0.001].sort_values(ascending=False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        weights_df = pd.DataFrame({
            'Asset': optimal_weights.index,
            'Weight': optimal_weights.values
        })
        weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
        st.dataframe(weights_df, hide_index=True, use_container_width=True)
    
    with col2:
        from src.dashboard.utils.plotting import create_portfolio_pie_chart
        fig = create_portfolio_pie_chart(optimal_weights, "Optimal Allocation")
        st.plotly_chart(fig, use_container_width=True)


def render_efficient_frontier_view(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    current_portfolio: Optional[dict] = None
):
    """
    Render efficient frontier visualization.
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns
    cov_matrix : pd.DataFrame
        Covariance matrix
    current_portfolio : dict, optional
        Current portfolio stats
    """
    st.subheader("Efficient Frontier")
    
    with st.spinner("Calculating efficient frontier..."):
        optimizer = PortfolioOptimizer(
            objective_type='mean_variance',
            constraints={'long_only': True}
        )
        
        frontier = optimizer.efficient_frontier(
            expected_returns=expected_returns.values / 252,
            cov_matrix=cov_matrix.values,
            n_points=30
        )
        
        fig = create_efficient_frontier(
            frontier,
            current_portfolio=current_portfolio,
            title="Efficient Frontier"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(
            "**Efficient Frontier** shows the optimal risk-return tradeoff. "
            "Each point represents a portfolio that maximizes return for a given level of risk."
        )