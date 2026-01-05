"""
Tests for ReturnEstimator class.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.returns import ReturnEstimator


class TestReturnEstimator:
    """Test suite for ReturnEstimator."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        
        # Create returns with slight positive drift
        returns = pd.DataFrame({
            'ASSET1': np.random.randn(500) * 0.01 + 0.0002,
            'ASSET2': np.random.randn(500) * 0.015 + 0.0003,
            'ASSET3': np.random.randn(500) * 0.02 + 0.0001,
        }, index=dates)
        
        return returns
    
    @pytest.fixture
    def market_returns(self):
        """Create market return series."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        returns = pd.Series(
            np.random.randn(500) * 0.012 + 0.0003,
            index=dates,
            name='MARKET'
        )
        return returns
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = ReturnEstimator(method='historical', lookback=252)
        assert estimator.method == 'historical'
        assert estimator.lookback == 252
    
    def test_historical_mean(self, sample_returns):
        """Test historical mean estimation."""
        estimator = ReturnEstimator(method='historical')
        expected_returns = estimator.estimate(sample_returns, annualize=False)
        
        assert isinstance(expected_returns, pd.Series)
        assert len(expected_returns) == 3
        assert all(expected_returns.index == sample_returns.columns)
    
    def test_momentum_based(self, sample_returns):
        """Test momentum-based estimation."""
        estimator = ReturnEstimator(method='momentum', momentum_window=60)
        expected_returns = estimator.estimate(sample_returns, annualize=False)
        
        assert isinstance(expected_returns, pd.Series)
        assert len(expected_returns) == 3
    
    def test_factor_model(self, sample_returns, market_returns):
        """Test factor model estimation."""
        estimator = ReturnEstimator(method='factor')
        expected_returns = estimator.estimate(
            sample_returns,
            market_returns=market_returns,
            annualize=False
        )
        
        assert isinstance(expected_returns, pd.Series)
        assert len(expected_returns) == 3
    
    def test_ensemble_estimate(self, sample_returns, market_returns):
        """Test ensemble estimation."""
        estimator = ReturnEstimator(method='ensemble')
        expected_returns = estimator.estimate(
            sample_returns,
            market_returns=market_returns,
            annualize=False
        )
        
        assert isinstance(expected_returns, pd.Series)
        assert len(expected_returns) == 3
    
    def test_annualization(self, sample_returns):
        """Test return annualization."""
        estimator = ReturnEstimator(method='historical')
        
        daily = estimator.estimate(sample_returns, annualize=False)
        annual = estimator.estimate(sample_returns, annualize=True)
        
        # Annual should be ~252 times daily
        ratio = annual / daily
        assert np.allclose(ratio, 252, rtol=0.01)
    
    def test_estimate_with_uncertainty(self, sample_returns):
        """Test uncertainty estimation."""
        estimator = ReturnEstimator(method='historical')
        results = estimator.estimate_with_uncertainty(sample_returns)
        
        assert isinstance(results, pd.DataFrame)
        assert 'expected_return' in results.columns
        assert 'lower_bound' in results.columns
        assert 'upper_bound' in results.columns
        
        # Lower bound should be less than upper bound
        assert (results['lower_bound'] <= results['upper_bound']).all()
    
    def test_rank_assets(self, sample_returns):
        """Test asset ranking."""
        estimator = ReturnEstimator(method='historical')
        ranks = estimator.rank_assets(sample_returns, method='return')
        
        assert isinstance(ranks, pd.Series)
        assert len(ranks) == 3
        assert set(ranks.values) == {1.0, 2.0, 3.0}
    
    def test_rank_by_sharpe(self, sample_returns):
        """Test ranking by Sharpe ratio."""
        estimator = ReturnEstimator(method='historical')
        ranks = estimator.rank_assets(sample_returns, method='sharpe')
        
        assert isinstance(ranks, pd.Series)
        assert len(ranks) == 3


class TestReturnEstimatorComparison:
    """Test comparison of return estimation methods."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        
        returns = pd.DataFrame({
            'A': np.random.randn(500) * 0.01 + 0.0002,
            'B': np.random.randn(500) * 0.015 + 0.0003,
        }, index=dates)
        
        market = pd.Series(
            np.random.randn(500) * 0.012 + 0.0003,
            index=dates
        )
        
        return returns, market
    
    def test_methods_produce_different_estimates(self, sample_data):
        """Test that different methods give different results."""
        returns, market = sample_data
        
        hist = ReturnEstimator(method='historical').estimate(returns)
        momentum = ReturnEstimator(method='momentum').estimate(returns)
        
        # Should not be identical
        assert not np.allclose(hist.values, momentum.values)
    
    def test_ensemble_within_range(self, sample_data):
        """Test ensemble is within range of individual methods."""
        returns, market = sample_data
        
        hist = ReturnEstimator(method='historical').estimate(returns)
        momentum = ReturnEstimator(method='momentum').estimate(returns)
        ensemble = ReturnEstimator(method='ensemble').estimate(returns, market)
        
        # Ensemble should be between min and max of components
        for asset in returns.columns:
            min_val = min(hist[asset], momentum[asset])
            max_val = max(hist[asset], momentum[asset])
            
            # Allow some tolerance
            assert ensemble[asset] >= min_val * 0.9
            assert ensemble[asset] <= max_val * 1.1