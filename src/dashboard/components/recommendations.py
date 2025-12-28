"""
Rebalancing recommendations component.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

from src.optimization.costs import TransactionCostModel
from src.dashboard.utils.formatting import format_percentage, format_currency, format_basis_points


def render_recommendations_view(
    current_weights: pd.Series,
    optimal_weights: pd.Series,
    portfolio_value: float,
    prices: pd.Series,
    expected_benefit: Optional[float] = None
):
    """
    Render rebalancing recommendations.
    
    Parameters
    ----------
    current_weights : pd.Series
        Current portfolio weights
    optimal_weights : pd.Series
        Target optimal weights
    portfolio_value : float
        Total portfolio value
    prices : pd.Series
        Current asset prices
    expected_benefit : float, optional
        Expected benefit from rebalancing
    """
    st.header("💡 Rebalancing Recommendations")
    
    # Calculate changes
    weight_changes = optimal_weights - current_weights
    turnover = np.abs(weight_changes).sum() / 2
    
    # Transaction costs
    cost_model = TransactionCostModel(
        spread_bps=5.0,
        impact_coefficient=0.1,
        commission_bps=1.0
    )
    
    total_cost = cost_model.total_cost(
        current_weights.values,
        optimal_weights.values,
        prices.values,
        portfolio_value
    )
    
    cost_pct = total_cost / portfolio_value
    
    # Summary metrics
    st.subheader("Rebalancing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Portfolio Turnover",
            format_percentage(turnover),
            help="Percentage of portfolio being rebalanced"
        )
    
    with col2:
        st.metric(
            "Transaction Cost",
            format_currency(total_cost),
            help="Estimated total transaction costs"
        )
    
    with col3:
        st.metric(
            "Cost (% of Portfolio)",
            format_basis_points(cost_pct),
            help="Transaction cost as basis points"
        )
    
    with col4:
        if expected_benefit is not None:
            net_benefit = expected_benefit - cost_pct
            st.metric(
                "Net Benefit",
                format_basis_points(net_benefit),
                delta=format_basis_points(net_benefit),
                help="Expected benefit minus transaction costs"
            )
        else:
            st.metric(
                "Net Benefit",
                "N/A",
                help="Optimization benefit not calculated"
            )
    
    # Recommendation decision
    st.subheader("Recommendation")
    
    if turnover < 0.05:
        st.success(
            "✅ **No Action Required**: Portfolio is close to optimal allocation. "
            f"Transaction costs ({format_basis_points(cost_pct)}) would exceed benefits."
        )
        recommend_rebalance = False
    
    elif expected_benefit is not None and expected_benefit < cost_pct:
        st.warning(
            "⚠️ **Hold Current Positions**: Expected benefit "
            f"({format_basis_points(expected_benefit)}) is less than transaction costs "
            f"({format_basis_points(cost_pct)}). Wait for larger drift from optimal."
        )
        recommend_rebalance = False
    
    else:
        st.info(
            "💼 **Rebalance Recommended**: Portfolio has drifted from optimal allocation. "
            f"Expected net benefit: {format_basis_points(expected_benefit - cost_pct if expected_benefit else 0)}."
        )
        recommend_rebalance = True
    
    # Detailed trade list
    st.subheader("Trade List")
    
    trades = pd.DataFrame({
        'Asset': current_weights.index,
        'Current Weight': current_weights.values,
        'Target Weight': optimal_weights.values,
        'Change': weight_changes.values,
        'Current Shares': (current_weights.values * portfolio_value) / prices.values,
        'Target Shares': (optimal_weights.values * portfolio_value) / prices.values,
        'Shares to Trade': ((optimal_weights.values - current_weights.values) * portfolio_value) / prices.values,
        'Trade Value': (optimal_weights.values - current_weights.values) * portfolio_value
    })
    
    # Filter to significant trades
    trades = trades[np.abs(trades['Change']) > 0.005].copy()
    trades['Action'] = trades['Change'].apply(lambda x: 'BUY' if x > 0 else 'SELL')
    
    if len(trades) > 0:
        trades = trades.sort_values('Change', ascending=False)
        
        # Format for display
        trades_display = trades.copy()
        trades_display['Current Weight'] = trades_display['Current Weight'].apply(lambda x: f"{x*100:.2f}%")
        trades_display['Target Weight'] = trades_display['Target Weight'].apply(lambda x: f"{x*100:.2f}%")
        trades_display['Change'] = trades_display['Change'].apply(lambda x: f"{x*100:+.2f}%")
        trades_display['Current Shares'] = trades_display['Current Shares'].apply(lambda x: f"{x:.2f}")
        trades_display['Target Shares'] = trades_display['Target Shares'].apply(lambda x: f"{x:.2f}")
        trades_display['Shares to Trade'] = trades_display['Shares to Trade'].apply(lambda x: f"{x:+.2f}")
        trades_display['Trade Value'] = trades_display['Trade Value'].apply(lambda x: f"${x:+,.0f}")
        
        # Color code actions
        def color_action(val):
            if val == 'BUY':
                return 'background-color: #d4edda'
            elif val == 'SELL':
                return 'background-color: #f8d7da'
            return ''
        
        styled_trades = trades_display.style.applymap(
            color_action,
            subset=['Action']
        )
        
        st.dataframe(styled_trades, hide_index=True, use_container_width=True)
        
        # Trade execution guide
        if recommend_rebalance:
            with st.expander("📋 Execution Guide"):
                st.write("**Recommended Execution Order:**")
                st.write("1. Execute SELL orders first to raise cash")
                st.write("2. Wait for settlement (T+2 for stocks)")
                st.write("3. Execute BUY orders with settled cash")
                st.write("4. Use limit orders to minimize market impact")
                st.write("5. Split large orders across multiple days if needed")
                
                st.write("\n**Cost Breakdown:**")
                
                cost_breakdown = pd.DataFrame({
                    'Component': ['Bid-Ask Spread', 'Market Impact', 'Commission', 'Total'],
                    'Estimated Cost': [
                        format_currency(total_cost * 0.5),  # Approximate
                        format_currency(total_cost * 0.3),  # Approximate
                        format_currency(total_cost * 0.2),  # Approximate
                        format_currency(total_cost)
                    ]
                })
                
                st.table(cost_breakdown)
    
    else:
        st.info("No significant trades required. Portfolio is well-aligned with target allocation.")
    
    # Download trade list
    if len(trades) > 0 and recommend_rebalance:
        st.subheader("Export")
        
        csv = trades.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Trade List (CSV)",
            data=csv,
            file_name="rebalancing_trades.csv",
            mime="text/csv",
            use_container_width=True
        )


def calculate_execution_schedule(
    trades: pd.DataFrame,
    daily_volume_limit: float = 0.1
) -> pd.DataFrame:
    """
    Calculate multi-day execution schedule.
    
    Parameters
    ----------
    trades : pd.DataFrame
        Trade list
    daily_volume_limit : float
        Maximum percentage of daily volume to trade
    
    Returns
    -------
    pd.DataFrame
        Execution schedule
    """
    # Simple schedule: split large trades across days
    schedule = []
    
    for _, trade in trades.iterrows():
        trade_value = abs(trade['Trade Value'])
        
        # Assuming average daily volume of $10M per asset
        estimated_volume = 10_000_000
        
        max_daily_trade = estimated_volume * daily_volume_limit
        n_days = int(np.ceil(trade_value / max_daily_trade))
        
        daily_amount = trade_value / n_days
        
        for day in range(n_days):
            schedule.append({
                'Day': day + 1,
                'Asset': trade['Asset'],
                'Action': trade['Action'],
                'Amount': daily_amount
            })
    
    return pd.DataFrame(schedule)