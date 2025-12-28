"""
Regime detection dashboard component.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.models.regime import RegimeDetector
from src.models.volatility import VolatilityModel
from src.dashboard.utils.plotting import create_regime_probability_chart
from src.dashboard.utils.formatting import format_percentage


def render_regime_view(returns: pd.DataFrame, prices: pd.DataFrame):
    """
    Render regime detection dashboard.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Returns data
    prices : pd.DataFrame
        Price data
    """
    st.header("🔮 Market Regime Detection")
    
    # Use market proxy (first asset or SPY)
    if 'SPY' in returns.columns:
        market_returns = returns['SPY']
    else:
        market_returns = returns.iloc[:, 0]
    
    # Volatility estimation
    vol_model = VolatilityModel(method='ewm', ewm_halflife=30)
    volatility = vol_model.estimate(market_returns, annualize=True)
    
    # Regime detection
    st.subheader("Regime Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_regimes = st.selectbox(
            "Number of Regimes",
            options=[2, 3],
            help="Number of distinct market states to detect"
        )
    
    with col2:
        n_iterations = st.slider(
            "Training Iterations",
            min_value=50,
            max_value=200,
            value=100,
            step=10,
            help="More iterations = better convergence"
        )
    
    if st.button("🔍 Detect Regimes", type="primary", use_container_width=True):
        with st.spinner("Detecting market regimes..."):
            try:
                # Fit regime detector
                detector = RegimeDetector(
                    n_regimes=n_regimes,
                    n_iter=n_iterations,
                    random_state=42
                )
                
                detector.fit(market_returns, volatility / 100)
                
                # Get predictions
                regimes = detector.predict_regimes(market_returns, volatility / 100)
                probabilities = detector.predict_probabilities(market_returns, volatility / 100)
                
                # Current regime
                current_prob = detector.get_current_regime_probability(market_returns, volatility / 100)
                
                # Display current state
                st.subheader("Current Market State")
                
                cols = st.columns(n_regimes)
                for i in range(n_regimes):
                    with cols[i]:
                        prob = current_prob[f'regime_{i}']
                        st.metric(
                            f"Regime {i}",
                            format_percentage(prob),
                            delta=None,
                            help=f"Probability of being in Regime {i}"
                        )
                
                most_likely = current_prob.idxmax()
                regime_num = int(most_likely.split('_')[1])
                st.success(f"**Most Likely Regime:** Regime {regime_num} ({current_prob[most_likely]:.1%} probability)")
                
                # Regime statistics
                st.subheader("Regime Characteristics")
                
                stats = detector.get_regime_stats()
                
                stats_display = stats.copy()
                stats_display['frequency'] = stats_display['frequency'].apply(lambda x: f"{x*100:.1f}%")
                stats_display['mean_return'] = stats_display['mean_return'].apply(lambda x: f"{x:.2f}%")
                stats_display['volatility'] = stats_display['volatility'].apply(lambda x: f"{x:.2f}%")
                stats_display['sharpe'] = stats_display['sharpe'].apply(lambda x: f"{x:.2f}")
                stats_display['avg_duration'] = stats_display['avg_duration'].apply(lambda x: f"{x:.0f} days")
                
                st.dataframe(stats_display, use_container_width=True)
                
                # Probability chart
                st.subheader("Regime Probabilities Over Time")
                
                fig = create_regime_probability_chart(
                    probabilities,
                    "Market Regime Probabilities"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Regime interpretation
                st.subheader("Regime Interpretation")
                
                for i in range(n_regimes):
                    regime_stats = stats.loc[i]
                    
                    with st.expander(f"📊 Regime {i} Details"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Statistical Characteristics:**")
                            st.write(f"- Frequency: {regime_stats['frequency']:.1%}")
                            st.write(f"- Mean Return: {regime_stats['mean_return']:.2f}% (annual)")
                            st.write(f"- Volatility: {regime_stats['volatility']:.2f}% (annual)")
                            st.write(f"- Sharpe Ratio: {regime_stats['sharpe']:.2f}")
                        
                        with col2:
                            st.write("**Temporal Characteristics:**")
                            st.write(f"- Average Duration: {regime_stats['avg_duration']:.0f} days")
                            
                            # Interpretation
                            if regime_stats['volatility'] < 15:
                                regime_type = "Low Volatility (Bull Market)"
                            elif regime_stats['volatility'] < 25:
                                regime_type = "Medium Volatility (Normal Market)"
                            else:
                                regime_type = "High Volatility (Bear/Crisis)"
                            
                            st.write(f"- **Type:** {regime_type}")
                
                # Investment implications
                st.info(
                    "**Investment Implications:** Portfolio allocations can be adjusted based on current regime. "
                    "High volatility regimes typically favor defensive positions (bonds, gold), while low volatility "
                    "regimes allow higher equity exposure."
                )
                
            except Exception as e:
                st.error(f"❌ Regime detection failed: {str(e)}")