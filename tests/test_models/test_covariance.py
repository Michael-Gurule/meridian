"""
Tests for CovarianceEstimator class.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.covariance import CovarianceEstimator


class TestCovarianceEstimator:
    """Test suite for CovarianceEstimator."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        
        # Create correlated returns
        n_assets = 5
        mean = np.zeros(n_assets)
        cov = np.array([
            [0.0004, 0.0002, 0.0001, 0.0001, 0.0001],
            [0.0002, 0.0003, 0.0001, 0.0001, 0.0001],
            [0.0001, 0.0001, 0.0005, 0.0002, 0.0001],
            [0.0001, 0.0001, 0.0002, 0.0004, 0.0001],
            [0.0001, 0.0001, 0.0001, 0.0001, 0.0003]
        ])
        
        returns = np.random.multivariate_normal(mean, cov, len(dates))
        
        return pd.DataFrame(
            returns,
            index=dates,
            columns=['ASSET1', 'ASSET2', 'ASSET3', 'ASSET4', 'ASSET5']
        )
    
    def test_initialization(self):
        """Test estimator initialization."""
        estimator = CovarianceEstimator(method='sample', window=252)
        assert estimator.method == 'sample'
        assert estimator.window == 252
    
    def test_sample_covariance(self, sample_returns):
        """Test sample covariance estimation."""
        estimator = CovarianceEstimator(method='sample')
        cov = estimator.estimate(sample_returns)
        
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (5, 5)
        assert np.all(cov.index == sample_returns.columns)
        assert np.allclose(cov, cov.T)  # Symmetric
    
    def test_ewm_covariance(self, sample_returns):
        """Test exponentially weighted covariance."""
        estimator = CovarianceEstimator(method='ewm', ewm_halflife=60)
        cov = estimator.estimate(sample_returns)
        
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (5, 5)
    
    def test_ledoit_wolf_covariance(self, sample_returns):
        """Test Ledoit-Wolf shrinkage."""
        estimator = CovarianceEstimator(method='ledoit_wolf')
        cov = estimator.estimate(sample_returns)
        
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (5, 5)
    
    def test_constant_correlation(self, sample_returns):
        """Test constant correlation model."""
        estimator = CovarianceEstimator(method='constant_correlation')
        cov = estimator.estimate(sample_returns)
        
        assert isinstance(cov, pd.DataFrame)
        assert cov.shape == (5, 5)
        
        # Check that off-diagonal correlations are equal
        corr = cov / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
        off_diag = corr.values[np.triu_indices(5, k=1)]
        assert np.allclose(off_diag, off_diag[0], atol=1e-10)
    
    def test_positive_definite(self, sample_returns):
        """Test that covariance matrix is positive definite."""
        estimator = CovarianceEstimator(method='sample')
        cov = estimator.estimate(sample_returns, ensure_pd=True)
        
        # Check eigenvalues are positive
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues > 0)
    
    def test_estimate_correlation(self, sample_returns):
        """Test correlation matrix estimation."""
        estimator = CovarianceEstimator(method='sample')
        corr = estimator.estimate_correlation(sample_returns)
        
        assert isinstance(corr, pd.DataFrame)
        assert corr.shape == (5, 5)
        
        # Diagonal should be 1
        assert np.allclose(np.diag(corr), 1.0)
        
        # Correlations should be between -1 and 1
        assert (corr.values >= -1.0 - 1e-10).all()
        assert (corr.values <= 1.0 + 1e-10).all()
    
    def test_forecast_covariance(self, sample_returns):
        """Test covariance forecasting."""
        estimator = CovarianceEstimator(method='sample')
        forecast = estimator.forecast_covariance(sample_returns, horizon=5)
        
        assert isinstance(forecast, pd.DataFrame)
        assert forecast.shape == (5, 5)
        
        # Should be roughly 5x the 1-day covariance
        one_day = estimator.estimate(sample_returns)
        assert np.allclose(forecast.values, one_day.values * 5, rtol=0.01)


class TestCovarianceComparison:
    """Test comparison of different covariance methods."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        
        returns = np.random.multivariate_normal(
            mean=[0, 0, 0],
            cov=[[0.0004, 0.0002, 0.0001],
                 [0.0002, 0.0003, 0.0001],
                 [0.0001, 0.0001, 0.0005]],
            size=len(dates)
        )
        
        return pd.DataFrame(returns, index=dates, columns=['A', 'B', 'C'])
    
    def test_methods_produce_different_results(self, sample_returns):
        """Test that different methods give different results."""
        sample_cov = CovarianceEstimator(method='sample').estimate(sample_returns)
        ewm_cov = CovarianceEstimator(method='ewm').estimate(sample_returns)
        lw_cov = CovarianceEstimator(method='ledoit_wolf').estimate(sample_returns)
        
        # Results should differ
        assert not np.allclose(sample_cov.values, ewm_cov.values)
        assert not np.allclose(sample_cov.values, lw_cov.values)
    
    def test_all_methods_positive_definite(self, sample_returns):
        """Test all methods produce positive definite matrices."""
        methods = ['sample', 'ewm', 'ledoit_wolf', 'constant_correlation']
        
        for method in methods:
            estimator = CovarianceEstimator(method=method)
            cov = estimator.estimate(sample_returns, ensure_pd=True)
            
            eigenvalues = np.linalg.eigvalsh(cov.values)
            assert np.all(eigenvalues > 0), f"{method} not positive definite"