"""
Feature engineering for portfolio analysis and modeling.

Creates derived features from price and return data for use in
regime detection, signal generation, and risk modeling.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict

from src.utils.logger import get_logger
from src.utils.statistics import calculate_rolling_zscore, calculate_drawdown_series

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Engineers features from price and return data.
    
    Creates technical indicators, statistical features, and derived metrics.
    """
    
    def __init__(self):
        """Initialize feature engineer."""
        logger.info("Initialized FeatureEngineer")
    
    def create_all_features(
        self,
        prices: pd.DataFrame,
        returns: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive feature set.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        returns : pd.DataFrame, optional
            Return data (computed if not provided)
        
        Returns
        -------
        pd.DataFrame
            Feature matrix
        """
        if returns is None:
            returns = prices.pct_change()
        
        features = {}
        
        # Price-based features
        features.update(self.create_price_features(prices))
        
        # Return-based features
        features.update(self.create_return_features(returns))
        
        # Volatility features
        features.update(self.create_volatility_features(returns))
        
        # Momentum features
        features.update(self.create_momentum_features(prices))
        
        # Combine into DataFrame
        feature_df = pd.DataFrame(features)
        
        logger.info(f"Created {len(feature_df.columns)} features")
        
        return feature_df
    
    def create_price_features(self, prices: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create price-based features.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        Dict[str, pd.Series]
            Price features
        """
        features = {}
        
        for asset in prices.columns:
            price = prices[asset]
            
            # Moving averages
            features[f'{asset}_ma_20'] = price.rolling(window=20).mean()
            features[f'{asset}_ma_50'] = price.rolling(window=50).mean()
            features[f'{asset}_ma_200'] = price.rolling(window=200).mean()
            
            # Price relative to moving averages
            features[f'{asset}_price_to_ma20'] = price / features[f'{asset}_ma_20']
            features[f'{asset}_price_to_ma50'] = price / features[f'{asset}_ma_50']
            
            # Drawdown
            features[f'{asset}_drawdown'] = calculate_drawdown_series(price)
            
            # Distance from all-time high
            cummax = price.cummax()
            features[f'{asset}_dist_from_high'] = (price - cummax) / cummax
        
        return features
    
    def create_return_features(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create return-based features.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return data
        
        Returns
        -------
        Dict[str, pd.Series]
            Return features
        """
        features = {}
        
        for asset in returns.columns:
            ret = returns[asset]
            
            # Rolling returns
            features[f'{asset}_ret_5d'] = ret.rolling(window=5).sum()
            features[f'{asset}_ret_20d'] = ret.rolling(window=20).sum()
            features[f'{asset}_ret_60d'] = ret.rolling(window=60).sum()
            
            # Return z-scores
            features[f'{asset}_ret_zscore'] = calculate_rolling_zscore(ret, window=252)
            
            # Skewness and kurtosis
            features[f'{asset}_skew_60d'] = ret.rolling(window=60).skew()
            features[f'{asset}_kurt_60d'] = ret.rolling(window=60).kurt()
            
            # Positive/negative return ratio
            pos_ret = ret[ret > 0].rolling(window=60).count()
            total_ret = ret.rolling(window=60).count()
            features[f'{asset}_pos_ratio'] = pos_ret / total_ret
        
        return features
    
    def create_volatility_features(self, returns: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create volatility-based features.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return data
        
        Returns
        -------
        Dict[str, pd.Series]
            Volatility features
        """
        features = {}
        
        for asset in returns.columns:
            ret = returns[asset]
            
            # Rolling volatility (multiple windows)
            features[f'{asset}_vol_10d'] = ret.rolling(window=10).std()
            features[f'{asset}_vol_30d'] = ret.rolling(window=30).std()
            features[f'{asset}_vol_60d'] = ret.rolling(window=60).std()
            
            # Volatility ratios (regime detection)
            features[f'{asset}_vol_ratio_10_60'] = (
                features[f'{asset}_vol_10d'] / features[f'{asset}_vol_60d']
            )
            
            # Realized volatility
            features[f'{asset}_realized_vol'] = np.sqrt(
                (ret ** 2).rolling(window=20).sum()
            )
            
            # Volatility of volatility
            features[f'{asset}_volvol'] = features[f'{asset}_vol_30d'].rolling(window=30).std()
        
        return features
    
    def create_momentum_features(self, prices: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create momentum-based features.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Price data
        
        Returns
        -------
        Dict[str, pd.Series]
            Momentum features
        """
        features = {}
        
        for asset in prices.columns:
            price = prices[asset]
            
            # Simple momentum (returns over various periods)
            features[f'{asset}_mom_1m'] = price.pct_change(periods=21)
            features[f'{asset}_mom_3m'] = price.pct_change(periods=63)
            features[f'{asset}_mom_6m'] = price.pct_change(periods=126)
            features[f'{asset}_mom_12m'] = price.pct_change(periods=252)
            
            # Rate of change
            features[f'{asset}_roc_20'] = (
                (price - price.shift(20)) / price.shift(20)
            )
            
            # Relative strength index (RSI)
            features[f'{asset}_rsi_14'] = self._calculate_rsi(price, window=14)
            
            # Moving average convergence divergence (MACD)
            macd, signal = self._calculate_macd(price)
            features[f'{asset}_macd'] = macd
            features[f'{asset}_macd_signal'] = signal
            features[f'{asset}_macd_diff'] = macd - signal
        
        return features
    
    def _calculate_rsi(self, price: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Parameters
        ----------
        price : pd.Series
            Price series
        window : int
            RSI window
        
        Returns
        -------
        pd.Series
            RSI values (0-100)
        """
        delta = price.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(
        self,
        price: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """
        Calculate MACD indicator.
        
        Parameters
        ----------
        price : pd.Series
            Price series
        fast : int
            Fast EMA period
        slow : int
            Slow EMA period
        signal : int
            Signal line period
        
        Returns
        -------
        tuple
            (MACD line, Signal line)
        """
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        return macd_line, signal_line
    
    def create_cross_asset_features(
        self,
        returns: pd.DataFrame,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Create features based on cross-asset relationships.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return data for multiple assets
        window : int
            Rolling window for correlations
        
        Returns
        -------
        pd.DataFrame
            Cross-asset features
        """
        features = {}
        
        # Rolling correlations between assets
        for i, asset1 in enumerate(returns.columns):
            for asset2 in returns.columns[i+1:]:
                corr = returns[asset1].rolling(window=window).corr(returns[asset2])
                features[f'corr_{asset1}_{asset2}'] = corr
        
        # Average correlation for each asset
        for asset in returns.columns:
            asset_corrs = []
            for other_asset in returns.columns:
                if asset != other_asset:
                    corr = returns[asset].rolling(window=window).corr(returns[other_asset])
                    asset_corrs.append(corr)
            
            features[f'{asset}_avg_corr'] = pd.concat(asset_corrs, axis=1).mean(axis=1)
        
        # Market dispersion (cross-sectional volatility)
        features['market_dispersion'] = returns.std(axis=1)
        
        return pd.DataFrame(features)
    
    def create_regime_features(
        self,
        returns: pd.DataFrame,
        windows: List[int] = [20, 60, 252]
    ) -> pd.DataFrame:
        """
        Create features useful for regime detection.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Return data
        windows : List[int]
            Windows for feature calculation
        
        Returns
        -------
        pd.DataFrame
            Regime-relevant features
        """
        features = {}
        
        # Market-wide features (assuming first column is market index)
        market_returns = returns.iloc[:, 0]
        
        for window in windows:
            # Mean return
            features[f'market_mean_{window}d'] = market_returns.rolling(window=window).mean()
            
            # Volatility
            features[f'market_vol_{window}d'] = market_returns.rolling(window=window).std()
            
            # Skewness
            features[f'market_skew_{window}d'] = market_returns.rolling(window=window).skew()
            
            # Kurtosis
            features[f'market_kurt_{window}d'] = market_returns.rolling(window=window).kurt()
        
        # Volatility regime indicators
        vol_short = market_returns.rolling(window=20).std()
        vol_long = market_returns.rolling(window=252).std()
        features['vol_regime'] = vol_short / vol_long
        
        # Trend strength
        ma_short = market_returns.rolling(window=20).mean()
        ma_long = market_returns.rolling(window=60).mean()
        features['trend_strength'] = (ma_short - ma_long) / vol_short
        
        return pd.DataFrame(features)