"""
Market regime detection using Hidden Markov Models.

Identifies distinct market states (e.g., low volatility, high volatility, crisis)
and provides regime probabilities for regime-conditional optimization.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RegimeDetector:
    """
    Detects market regimes using Hidden Markov Models.
    
    Identifies 2 or 3 distinct market states based on returns and volatility.
    """
    
    def __init__(
        self,
        n_regimes: int = 2,
        covariance_type: str = 'full',
        n_iter: int = 100,
        random_state: int = 42
    ):
        """
        Initialize regime detector.
        
        Parameters
        ----------
        n_regimes : int
            Number of hidden states (2 or 3 recommended)
        covariance_type : str
            Type of covariance parameters ('full', 'diag', 'spherical')
        n_iter : int
            Maximum iterations for EM algorithm
        random_state : int
            Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.regime_stats = None
        
        logger.info(f"Initialized RegimeDetector: {n_regimes} regimes")
    
    def fit(self, returns: pd.Series, volatility: Optional[pd.Series] = None):
        """Fit HMM to data."""
        # Clean data first
        if volatility is not None:
            # Align and clean both series together
            aligned_returns, aligned_vol = returns.align(volatility, join='inner')
            mask = ~(aligned_returns.isna() | aligned_vol.isna())
            self.clean_returns = aligned_returns[mask]
            self.clean_volatility = aligned_vol[mask]
        
            # Prepare features
            features = np.column_stack([
                self.clean_returns.values,
                self.clean_volatility.values
            ])
        else:
            # Just clean returns
            self.clean_returns = returns.dropna()
            self.clean_volatility = None
            features = self.clean_returns.values.reshape(-1, 1)
    
         # Standardize features
        features_scaled = self.scaler.fit_transform(features)
    
        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state
        )
    
        self.model.fit(features_scaled)
    
        # Compute regime statistics using cleaned returns
        self._compute_regime_stats(self.clean_returns, features_scaled)
    
        logger.info(f"HMM fitted: {self.n_regimes} regimes, {len(features)} observations")
    
    def _prepare_features(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series]
    ) -> np.ndarray:
        """
        Prepare feature matrix for HMM.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        volatility : pd.Series, optional
            Volatility series
        
        Returns
        -------
        np.ndarray
            Feature matrix
        """
        if volatility is not None:
            # Align series
            aligned_returns, aligned_vol = returns.align(volatility, join='inner')
        
            # Drop NaN values
            mask = ~(aligned_returns.isna() | aligned_vol.isna())
            clean_returns = aligned_returns[mask].values
            clean_vol = aligned_vol[mask].values
        
            # Stack features
            features = np.column_stack([clean_returns, clean_vol])
        else:
            # Just returns, drop NaN
            clean_returns = returns.dropna().values
            features = clean_returns.reshape(-1, 1)
        
        return features
    
    def predict_regimes(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Predict regime sequence.
    
        Parameters
        ----------
        returns : pd.Series
            Returns series
        volatility : pd.Series, optional
            Volatility series
        
        Returns
        -------
        pd.Series
            Predicted regime sequence
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Clean data the same way as in fit()
        if volatility is not None:
            aligned_returns, aligned_vol = returns.align(volatility, join='inner')
            mask = ~(aligned_returns.isna() | aligned_vol.isna())
            clean_returns = aligned_returns[mask]
            clean_volatility = aligned_vol[mask]
            
            features = np.column_stack([
                clean_returns.values,
                clean_volatility.values
            ])
        else:
            clean_returns = returns.dropna()
            features = clean_returns.values.reshape(-1, 1)
        
        # Standardize
        features_scaled = self.scaler.transform(features)
        
        # Predict
        regimes = self.model.predict(features_scaled)
        
        # Return with CLEANED index
        return pd.Series(regimes, index=clean_returns.index, name='regime')



    def predict_probabilities(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Predict regime probabilities.
        
        Parameters
        ----------
        returns : pd.Series
            Returns series
        volatility : pd.Series, optional
            Volatility series
        
        Returns
        -------
        pd.DataFrame
            Regime probabilities over time
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Clean data the same way as in fit()
        if volatility is not None:
            aligned_returns, aligned_vol = returns.align(volatility, join='inner')
            mask = ~(aligned_returns.isna() | aligned_vol.isna())
            clean_returns = aligned_returns[mask]
            clean_volatility = aligned_vol[mask]
            
            features = np.column_stack([
                clean_returns.values,
                clean_volatility.values
            ])
        else:
            clean_returns = returns.dropna()
            features = clean_returns.values.reshape(-1, 1)
        
        # Standardize
        features_scaled = self.scaler.transform(features)
        
        # Predict probabilities
        probs = self.model.predict_proba(features_scaled)
        
        # Column names
        columns = [f'regime_{i}' for i in range(self.n_regimes)]
        
        # Return with CLEANED index
        return pd.DataFrame(probs, index=clean_returns.index, columns=columns)
    
    def _compute_regime_stats(
        self,
        returns: pd.Series,
        features_scaled: np.ndarray
    ):
        """
        Compute statistics for each regime.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        features_scaled : np.ndarray
            Scaled feature matrix
        """
        regimes = self.model.predict(features_scaled)
        
        stats = {}
        
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            stats[regime] = {
                'mean_return': regime_returns.mean() * 252,  # Annualized
                'volatility': regime_returns.std() * np.sqrt(252),  # Annualized
                'sharpe': (regime_returns.mean() / regime_returns.std()) * np.sqrt(252),
                'frequency': mask.sum() / len(regimes),
                'avg_duration': self._calculate_avg_duration(regimes, regime)
            }
        
        self.regime_stats = pd.DataFrame(stats).T
        
        logger.info(f"Regime statistics:\n{self.regime_stats}")
    
    def _calculate_avg_duration(
        self,
        regimes: np.ndarray,
        target_regime: int
    ) -> float:
        """
        Calculate average duration of regime.
        
        Parameters
        ----------
        regimes : np.ndarray
            Regime sequence
        target_regime : int
            Target regime to analyze
        
        Returns
        -------
        float
            Average duration in days
        """
        durations = []
        current_duration = 0
        
        for regime in regimes:
            if regime == target_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def get_regime_stats(self) -> pd.DataFrame:
        """
        Get statistics for each regime.
        
        Returns
        -------
        pd.DataFrame
            Regime statistics
        """
        if self.regime_stats is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.regime_stats
    
    def get_current_regime_probability(
        self,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Get current regime probabilities (most recent period).
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        volatility : pd.Series, optional
            Volatility series
        
        Returns
        -------
        pd.Series
            Current regime probabilities
        """
        probabilities = self.predict_probabilities(returns, volatility)
        return probabilities.iloc[-1]