"""
Volatility estimation and forecasting models.

Implements:
- Historical volatility
- Exponentially weighted volatility
- GARCH(1,1) model
- Realized volatility
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from arch import arch_model

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VolatilityModel:
    """
    Estimates and forecasts volatility using multiple methods.
    
    Provides historical, EWMA, and GARCH-based volatility estimates.
    """
    
    def __init__(
        self,
        method: str = 'ewm',
        window: int = 30,
        ewm_halflife: int = 30
    ):
        """
        Initialize volatility model.
        
        Parameters
        ----------
        method : str
            Volatility estimation method ('historical', 'ewm', 'garch')
        window : int
            Rolling window for historical volatility
        ewm_halflife : int
            Halflife for EWMA volatility
        """
        self.method = method
        self.window = window
        self.ewm_halflife = ewm_halflife
        
        logger.info(f"Initialized VolatilityModel: method={method}")
    
    def estimate(
        self,
        returns: pd.Series,
        annualize: bool = True
    ) -> pd.Series:
        """
        Estimate volatility series.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        annualize : bool
            Annualize volatility (multiply by sqrt(252))
        
        Returns
        -------
        pd.Series
            Volatility estimates
        """
        if self.method == 'historical':
            vol = self._historical_volatility(returns)
        elif self.method == 'ewm':
            vol = self._ewm_volatility(returns)
        elif self.method == 'garch':
            vol = self._garch_volatility(returns)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if annualize:
            vol = vol * np.sqrt(252)
        
        return vol
    
    def _historical_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Calculate rolling historical volatility.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        
        Returns
        -------
        pd.Series
            Historical volatility
        """
        return returns.rolling(window=self.window).std()
    
    def _ewm_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Calculate exponentially weighted volatility.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        
        Returns
        -------
        pd.Series
            EWMA volatility
        """
        return returns.ewm(halflife=self.ewm_halflife, adjust=False).std()
    
    def _garch_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Estimate volatility using GARCH(1,1) model.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        
        Returns
        -------
        pd.Series
            GARCH conditional volatility
        """
        try:
            # Convert returns to percentage for better numerical stability
            returns_pct = returns * 100
            
            # Fit GARCH(1,1) model
            model = arch_model(
                returns_pct.dropna(),
                vol='Garch',
                p=1,
                q=1,
                rescale=False
            )
            
            result = model.fit(disp='off', show_warning=False)
            
            # Extract conditional volatility
            conditional_vol = result.conditional_volatility / 100  # Convert back
            
            # Align with original index
            vol_series = pd.Series(
                np.nan,
                index=returns.index,
                name='garch_volatility'
            )
            vol_series.loc[conditional_vol.index] = conditional_vol.values
            
            logger.info(f"GARCH model fitted: omega={result.params['omega']:.6f}")
            
            return vol_series
            
        except Exception as e:
            logger.error(f"GARCH fitting failed: {str(e)}, falling back to EWMA")
            return self._ewm_volatility(returns)
    
    def forecast_volatility(
        self,
        returns: pd.Series,
        horizon: int = 1,
        annualize: bool = True
    ) -> float:
        """
        Forecast volatility for future horizon.
        
        Parameters
        ----------
        returns : pd.Series
            Historical return series
        horizon : int
            Forecast horizon in days
        annualize : bool
            Return annualized volatility
        
        Returns
        -------
        float
            Forecasted volatility
        """
        if self.method == 'garch':
            return self._forecast_garch(returns, horizon, annualize)
        else:
            # For historical and EWMA, use current volatility
            current_vol = self.estimate(returns, annualize=False).iloc[-1]
            
            # Scale by sqrt(horizon) for multi-period
            forecasted_vol = current_vol * np.sqrt(horizon)
            
            if annualize:
                forecasted_vol *= np.sqrt(252)
            
            return forecasted_vol
    
    def _forecast_garch(
        self,
        returns: pd.Series,
        horizon: int,
        annualize: bool
    ) -> float:
        """
        Forecast volatility using GARCH model.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        horizon : int
            Forecast horizon
        annualize : bool
            Return annualized volatility
        
        Returns
        -------
        float
            GARCH forecasted volatility
        """
        try:
            returns_pct = returns * 100
            
            model = arch_model(
                returns_pct.dropna(),
                vol='Garch',
                p=1,
                q=1,
                rescale=False
            )
            
            result = model.fit(disp='off', show_warning=False)
            
            # Forecast variance
            forecast = result.forecast(horizon=horizon)
            forecasted_variance = forecast.variance.values[-1, -1]
            
            # Convert to volatility
            forecasted_vol = np.sqrt(forecasted_variance) / 100
            
            if annualize:
                forecasted_vol *= np.sqrt(252)
            
            return forecasted_vol
            
        except Exception as e:
            logger.error(f"GARCH forecast failed: {str(e)}")
            return self._ewm_volatility(returns).iloc[-1] * np.sqrt(252 if annualize else 1)
    
    def realized_volatility(
        self,
        returns: pd.Series,
        window: int = 30,
        annualize: bool = True
    ) -> pd.Series:
        """
        Calculate realized volatility (sum of squared returns).
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        window : int
            Rolling window
        annualize : bool
            Annualize volatility
        
        Returns
        -------
        pd.Series
            Realized volatility
        """
        squared_returns = returns ** 2
        realized_var = squared_returns.rolling(window=window).sum()
        realized_vol = np.sqrt(realized_var)
        
        if annualize:
            realized_vol *= np.sqrt(252)
        
        return realized_vol
    
    def volatility_ratio(
        self,
        returns: pd.Series,
        short_window: int = 10,
        long_window: int = 60
    ) -> pd.Series:
        """
        Calculate ratio of short-term to long-term volatility.
        
        High ratio indicates volatility spike (potential regime change).
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        short_window : int
            Short-term window
        long_window : int
            Long-term window
        
        Returns
        -------
        pd.Series
            Volatility ratio
        """
        short_vol = returns.rolling(window=short_window).std()
        long_vol = returns.rolling(window=long_window).std()
        
        return short_vol / long_vol