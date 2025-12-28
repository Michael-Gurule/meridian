"""
Statistical utility functions for MERIDIAN modeling.

Provides helper functions for common statistical operations used across
multiple modeling modules.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from scipy import stats
from src.utils.logger import get_logger

logger = get_logger(__name__)


def rolling_window(data: pd.DataFrame, window: int) -> np.ndarray:
    """
    Create rolling windows from DataFrame.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    window : int
        Window size
    
    Returns
    -------
    np.ndarray
        Array of rolling windows
    """
    shape = (data.shape[0] - window + 1, window, data.shape[1])
    strides = (data.values.strides[0],) + data.values.strides
    
    return np.lib.stride_tricks.as_strided(
        data.values,
        shape=shape,
        strides=strides,
        writeable=False
    )


def ensure_positive_definite(matrix: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
    """
    Ensure covariance matrix is positive definite.
    
    Adjusts eigenvalues to be above minimum threshold while preserving
    eigenvector structure.
    
    Parameters
    ----------
    matrix : np.ndarray
        Covariance or correlation matrix
    min_eigenvalue : float
        Minimum eigenvalue threshold
    
    Returns
    -------
    np.ndarray
        Positive definite matrix
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Clip negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    
    # Reconstruct matrix
    reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Ensure symmetry
    reconstructed = (reconstructed + reconstructed.T) / 2
    
    return reconstructed


def calculate_ewma(
    data: Union[pd.Series, pd.DataFrame],
    halflife: int = 60
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate exponentially weighted moving average.
    
    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data
    halflife : int
        Halflife in days (lambda = 0.5^(1/halflife))
    
    Returns
    -------
    pd.Series or pd.DataFrame
        EWMA values
    """
    return data.ewm(halflife=halflife, adjust=False).mean()


def calculate_rolling_zscore(
    data: pd.Series,
    window: int = 252
) -> pd.Series:
    """
    Calculate rolling z-scores.
    
    Parameters
    ----------
    data : pd.Series
        Input data
    window : int
        Rolling window size
    
    Returns
    -------
    pd.Series
        Z-scores
    """
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    
    return (data - rolling_mean) / rolling_std


def winsorize_series(
    data: pd.Series,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99
) -> pd.Series:
    """
    Winsorize extreme values in a series.
    
    Parameters
    ----------
    data : pd.Series
        Input data
    lower_percentile : float
        Lower percentile threshold
    upper_percentile : float
        Upper percentile threshold
    
    Returns
    -------
    pd.Series
        Winsorized series
    """
    lower_bound = data.quantile(lower_percentile)
    upper_bound = data.quantile(upper_percentile)
    
    return data.clip(lower=lower_bound, upper=upper_bound)


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Calculate information ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns
    benchmark_returns : pd.Series
        Benchmark returns
    
    Returns
    -------
    float
        Information ratio
    """
    active_returns = returns - benchmark_returns
    
    if active_returns.std() == 0:
        return 0.0
    
    return (active_returns.mean() / active_returns.std()) * np.sqrt(252)


def calculate_drawdown_series(prices: pd.Series) -> pd.Series:
    """
    Calculate drawdown time series.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    
    Returns
    -------
    pd.Series
        Drawdown series (negative values)
    """
    cumulative_max = prices.cummax()
    drawdown = (prices - cumulative_max) / cumulative_max
    
    return drawdown


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    prices : pd.Series
        Price series
    
    Returns
    -------
    float
        Maximum drawdown (negative value)
    """
    drawdown = calculate_drawdown_series(prices)
    return drawdown.min()


def calculate_calmar_ratio(
    returns: pd.Series,
    prices: pd.Series
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    prices : pd.Series
        Price series
    
    Returns
    -------
    float
        Calmar ratio
    """
    annual_return = returns.mean() * 252
    max_dd = abs(calculate_max_drawdown(prices))
    
    if max_dd == 0:
        return 0.0
    
    return annual_return / max_dd


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (penalizes only downside volatility).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    risk_free_rate : float
        Risk-free rate (annualized)
    target_return : float
        Target return threshold
    
    Returns
    -------
    float
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < target_return]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    
    return (excess_returns.mean() / downside_std) * np.sqrt(252)


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% VaR)
    
    Returns
    -------
    float
        VaR (positive value representing potential loss)
    """
    return -returns.quantile(1 - confidence_level)


def calculate_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    confidence_level : float
        Confidence level
    
    Returns
    -------
    float
        CVaR (positive value representing expected loss beyond VaR)
    """
    var = calculate_var(returns, confidence_level)
    return -returns[returns <= -var].mean()


def calculate_tail_ratio(returns: pd.Series) -> float:
    """
    Calculate tail ratio (95th percentile / 5th percentile).
    
    Higher values indicate positive skew (large gains vs losses).
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    
    Returns
    -------
    float
        Tail ratio
    """
    p95 = returns.quantile(0.95)
    p5 = returns.quantile(0.05)
    
    if p5 == 0:
        return np.inf if p95 > 0 else 0.0
    
    return abs(p95 / p5)


def calculate_stability(returns: pd.Series, window: int = 252) -> float:
    """
    Calculate R-squared of cumulative returns vs linear fit.
    
    Measures consistency of returns over time.
    
    Parameters
    ----------
    returns : pd.Series
        Return series
    window : int
        Minimum window for calculation
    
    Returns
    -------
    float
        R-squared value (0 to 1)
    """
    if len(returns) < window:
        return 0.0
    
    cumulative = (1 + returns).cumprod()
    x = np.arange(len(cumulative))
    
    slope, intercept, r_value, _, _ = stats.linregress(x, cumulative.values)
    
    return r_value ** 2