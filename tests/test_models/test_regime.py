"""
Tests for RegimeDetector class.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.regime import RegimeDetector


class TestRegimeDetector:
    """Test suite for RegimeDetector."""
    
    @pytest.fixture
    def sample_returns_two_regimes(self):
        """Create return series with two clear regimes."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        
        # Low volatility regime
        regime1 = np.random.randn(250) * 0.005
        
        # High volatility regime
        regime2 = np.random.randn(250) * 0.03
        
        returns = np.concatenate([regime1, regime2])
        
        return pd.Series(returns, index=dates, name='RETURNS')
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = RegimeDetector(n_regimes=2)
        assert detector.n_regimes == 2
        assert detector.model is None
    
    def test_fit(self, sample_returns_two_regimes):
        """Test fitting HMM."""
        detector = RegimeDetector(n_regimes=2, n_iter=50)
        detector.fit(sample_returns_two_regimes)
        
        assert detector.model is not None
        assert detector.regime_stats is not None
    
    def test_predict_regimes(self, sample_returns_two_regimes):
        """Test regime prediction."""
        detector = RegimeDetector(n_regimes=2, n_iter=50)
        detector.fit(sample_returns_two_regimes)
        
        regimes = detector.predict_regimes(sample_returns_two_regimes)
        
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(sample_returns_two_regimes)
        assert set(regimes.unique()).issubset({0, 1})
    
    def test_predict_probabilities(self, sample_returns_two_regimes):
        """Test regime probability prediction."""
        detector = RegimeDetector(n_regimes=2, n_iter=50)
        detector.fit(sample_returns_two_regimes)
        
        probs = detector.predict_probabilities(sample_returns_two_regimes)
        
        assert isinstance(probs, pd.DataFrame)
        assert probs.shape == (len(sample_returns_two_regimes), 2)
        assert (probs >= 0).all().all()
        assert (probs <= 1).all().all()
        
        # Probabilities should sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)
    
    def test_regime_stats(self, sample_returns_two_regimes):
        """Test regime statistics computation."""
        detector = RegimeDetector(n_regimes=2, n_iter=50)
        detector.fit(sample_returns_two_regimes)
        
        stats = detector.get_regime_stats()
        
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 2
        assert 'mean_return' in stats.columns
        assert 'volatility' in stats.columns
        assert 'sharpe' in stats.columns
    
    def test_different_volatility_regimes(self, sample_returns_two_regimes):
        """Test that regimes have different volatilities."""
        detector = RegimeDetector(n_regimes=2, n_iter=50)
        detector.fit(sample_returns_two_regimes)
        
        stats = detector.get_regime_stats()
        
        # One regime should have much higher volatility
        vol_diff = abs(stats['volatility'].iloc[0] - stats['volatility'].iloc[1])
        assert vol_diff > 0.05  # Significant difference
    
    def test_current_regime_probability(self, sample_returns_two_regimes):
        """Test current regime probability."""
        detector = RegimeDetector(n_regimes=2, n_iter=50)
        detector.fit(sample_returns_two_regimes)
        
        current_prob = detector.get_current_regime_probability(sample_returns_two_regimes)
        
        assert isinstance(current_prob, pd.Series)
        assert len(current_prob) == 2
        assert np.allclose(current_prob.sum(), 1.0)


class TestRegimeDetectorThreeStates:
    """Test 3-state regime detection."""
    
    @pytest.fixture
    def sample_returns_three_regimes(self):
        """Create returns with three regimes."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=600, freq='D')
        
        # Low vol
        regime1 = np.random.randn(200) * 0.005
        # Medium vol
        regime2 = np.random.randn(200) * 0.015
        # High vol (crisis)
        regime3 = np.random.randn(200) * 0.04
        
        returns = np.concatenate([regime1, regime2, regime3])
        
        return pd.Series(returns, index=dates, name='RETURNS')
    
    def test_three_regime_detection(self, sample_returns_three_regimes):
        """Test 3-regime HMM."""
        detector = RegimeDetector(n_regimes=3, n_iter=50)
        detector.fit(sample_returns_three_regimes)
        
        regimes = detector.predict_regimes(sample_returns_three_regimes)
        
        # Should have 3 distinct regimes
        assert len(regimes.unique()) == 3
        
        stats = detector.get_regime_stats()
        assert len(stats) == 3