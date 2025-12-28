"""
Formatting utilities for dashboard displays.
"""

import pandas as pd
import numpy as np
from typing import Union


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Parameters
    ----------
    value : float
        Value to format (0.05 = 5%)
    decimals : int
        Decimal places
    
    Returns
    -------
    str
        Formatted percentage
    """
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 0) -> str:
    """
    Format value as currency.
    
    Parameters
    ----------
    value : float
        Dollar amount
    decimals : int
        Decimal places
    
    Returns
    -------
    str
        Formatted currency
    """
    return f"${value:,.{decimals}f}"


def format_basis_points(value: float) -> str:
    """
    Format value as basis points.
    
    Parameters
    ----------
    value : float
        Value (0.01 = 100 bps)
    
    Returns
    -------
    str
        Formatted bps
    """
    return f"{value * 10000:.1f} bps"


def color_negative_red(val: Union[float, str]) -> str:
    """
    Color negative values red, positive green.
    
    Parameters
    ----------
    val : float or str
        Value to color
    
    Returns
    -------
    str
        CSS color string
    """
    if isinstance(val, str):
        return ''
    
    color = '#d62728' if val < 0 else '#2ca02c'
    return f'color: {color}'


def create_metric_card(title: str, value: str, delta: str = None) -> str:
    """
    Create HTML for metric card.
    
    Parameters
    ----------
    title : str
        Metric title
    value : str
        Main value
    delta : str, optional
        Change value
    
    Returns
    -------
    str
        HTML string
    """
    delta_html = f'<div class="delta">{delta}</div>' if delta else ''
    
    return f"""
    <div style="
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    ">
        <div style="color: #808495; font-size: 14px;">{title}</div>
        <div style="font-size: 32px; font-weight: bold; margin: 10px 0;">{value}</div>
        {delta_html}
    </div>
    """


def style_dataframe(df: pd.DataFrame, 
                    highlight_cols: list = None,
                    percent_cols: list = None) -> pd.DataFrame:
    """
    Apply styling to dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to style
    highlight_cols : list, optional
        Columns to highlight negatives
    percent_cols : list, optional
        Columns to format as percentage
    
    Returns
    -------
    pd.DataFrame.Styler
        Styled dataframe
    """
    styled = df.style
    
    if highlight_cols:
        styled = styled.applymap(color_negative_red, subset=highlight_cols)
    
    if percent_cols:
        styled = styled.format({col: '{:.2%}' for col in percent_cols})
    
    return styled