"""
Tests for PortfolioOptimizer class.
"""

import pytest
import numpy as np
import pandas as pd
from src.optimization.optimizer import PortfolioOptimizer


class TestPortfolioOptimizer:
    """Test suite for PortfolioOptimizer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample optimization data."""
        np.random.seed(42)
        
        n_assets = 5
        expected_returns = np.array([0.08, 0.10, 0.12, 0.06, 0.09]) / 252
        
        # Create positive definite covariance matrix
        A = np.random.randn(n_assets, n_assets)
        cov_matrix = (A @ A.T) / 1000
        
        return {
            'expected_returns': expected_returns,
            'cov_matrix': cov_matrix,
            'n_assets': n_assets
        }
    
    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = PortfolioOptimizer(objective_type='sharpe')
        assert optimizer.objective_type == 'sharpe'
    
    def test_optimize_mean_variance(self, sample_data):
        """Test mean-variance optimization."""
        optimizer = PortfolioOptimizer(
            objective_type='mean_variance',
            constraints={'long_only': True}
        )
        
        result = optimizer.optimize(
            expected_returns=sample_data['expected_returns'],
            cov_matrix=sample_data['cov_matrix']
        )
        
        assert 'weights' in result
        assert len(result['weights']) == sample_data['n_assets']
        assert np.isclose(result['weights'].sum(), 1.0, atol=1e-4)
        assert (result['weights'] >= -1e-6).all()  # Long-only
    
    def test_optimize_min_variance(self, sample_data):
        """Test minimum variance optimization."""
        optimizer = PortfolioOptimizer(
            objective_type='min_variance',
            constraints={'long_only': True}
        )
        
        result = optimizer.optimize(cov_matrix=sample_data['cov_matrix'])
        
        assert 'weights' in result
        assert np.isclose(result['weights'].sum(), 1.0, atol=1e-4)
        assert result['status'] in ['optimal', 'optimal_inaccurate']
    
    def test_optimize_with_turnover_constraint(self, sample_data):
        """Test optimization with turnover constraint."""
        current_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        optimizer = PortfolioOptimizer(
            objective_type='min_variance',
            constraints={'long_only': True, 'max_turnover': 0.3}
        )
        
        result = optimizer.optimize(
            cov_matrix=sample_data['cov_matrix'],
            current_weights=current_weights
        )
        
        assert result['turnover'] <= 0.3 + 1e-4
    
    def test_optimize_with_weight_bounds(self, sample_data):
        """Test optimization with weight bounds."""
        optimizer = PortfolioOptimizer(
            objective_type='min_variance',
            constraints={
                'long_only': True,
                'weight_bounds': (0.0, 0.3)
            }
        )
        
        result = optimizer.optimize(cov_matrix=sample_data['cov_matrix'])
        
        assert (result['weights'] >= -1e-6).all()
        assert (result['weights'] <= 0.3 + 1e-4).all()
    
    def test_result_contains_key_metrics(self, sample_data):
        """Test that result contains all expected metrics."""
        optimizer = PortfolioOptimizer(objective_type='min_variance')
        
        result = optimizer.optimize(cov_matrix=sample_data['cov_matrix'])
        
        required_keys = [
            'weights', 'objective_value', 'status',
            'expected_return', 'volatility', 'sharpe_ratio', 'turnover'
        ]
        
        for key in required_keys:
            assert key in result
    
    def test_efficient_frontier(self, sample_data):
        """Test efficient frontier calculation."""
        optimizer = PortfolioOptimizer(
            objective_type='mean_variance',
            constraints={'long_only': True}
        )
        
        frontier = optimizer.efficient_frontier(
            expected_returns=sample_data['expected_returns'],
            cov_matrix=sample_data['cov_matrix'],
            n_points=10
        )
        
        assert isinstance(frontier, pd.DataFrame)
        assert len(frontier) > 0
        assert 'return' in frontier.columns
        assert 'volatility' in frontier.columns


class TestRegimeConditionalOptimization:
    """Test regime-conditional optimization."""
    
    @pytest.fixture
    def regime_data(self):
        """Create regime-specific data."""
        n_assets = 3
        
        # Two regimes
        regime_0_returns = np.array([0.001, 0.0015, 0.0012])
        regime_1_returns = np.array([-0.001, 0.0005, 0.001])
        
        regime_0_cov = np.array([
            [0.0001, 0.00005, 0.00002],
            [0.00005, 0.00015, 0.00003],
            [0.00002, 0.00003, 0.00012]
        ])
        
        regime_1_cov = np.array([
            [0.0004, 0.0002, 0.0001],
            [0.0002, 0.0006, 0.0002],
            [0.0001, 0.0002, 0.0005]
        ])
        
        return {
            'probabilities': np.array([0.7, 0.3]),
            'returns': [regime_0_returns, regime_1_returns],
            'covariances': [regime_0_cov, regime_1_cov]
        }
    
    def test_regime_conditional_optimize(self, regime_data):
        """Test regime-conditional optimization."""
        optimizer = PortfolioOptimizer(
            objective_type='mean_variance',
            constraints={'long_only': True}
        )
        
        result = optimizer.regime_conditional_optimize(
            regime_probabilities=regime_data['probabilities'],
            regime_returns=regime_data['returns'],
            regime_covariances=regime_data['covariances']
        )
        
        assert 'weights' in result
        assert 'regime_probabilities' in result
        assert 'regime_specific' in result
        assert len(result['regime_specific']) == 2