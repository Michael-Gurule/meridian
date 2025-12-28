"""
Session state management for Streamlit.
"""

import streamlit as st
from typing import Any, Dict


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'portfolio_value': 1_000_000.0,
        'current_weights': None,
        'optimization_result': None,
        'regime_state': None,
        'selected_assets': None,
        'backtest_results': None,
        'last_update': None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def get_state(key: str, default: Any = None) -> Any:
    """
    Get value from session state.
    
    Parameters
    ----------
    key : str
        State key
    default : Any
        Default value if not found
    
    Returns
    -------
    Any
        State value
    """
    return st.session_state.get(key, default)


def set_state(key: str, value: Any):
    """
    Set value in session state.
    
    Parameters
    ----------
    key : str
        State key
    value : Any
        Value to set
    """
    st.session_state[key] = value


def clear_state(*keys: str):
    """
    Clear specific keys from session state.
    
    Parameters
    ----------
    *keys : str
        Keys to clear
    """
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]