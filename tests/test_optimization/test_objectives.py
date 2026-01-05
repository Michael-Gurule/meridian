"""
Tests for ObjectiveFunction class.
"""

import pytest
import numpy as np
import cvxpy as cp
from src.optimization.objectives import ObjectiveFunction


class TestObjectiveFunction:
    """Test suite for ObjectiveFunction."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample optimization data."""
        n_assets = 5
        
        expected_returns = np.array([0.08, 0.10, 0.12, 0.06, 0.09]) / 252
        cov_matrix = np.array([
            [0.04, 0.02, 0.01, 0.01, 0.01],
            [0.02, 0.05, 0.02, 0.01, 0.01],
            [0.01, 0.02, 0.06, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.03, 0.01],
            [0.01, 0.01, 0.01, 0.01, 0.04]
        ]) / 252
        
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        return {
            'n_assets': n_assets,
            'expected_returns': expected_returns,
            'cov_matrix': cov_matrix,
            'volatilities': volatilities
        }
    
    def test_initialization(self):
        """Test objective function initialization."""
        obj = ObjectiveFunction(objective_type='sharpe')
        assert obj.objective_type == 'sharpe'
    
    def test_mean_variance_objective(self, sample_data):
        """Test mean-variance objective."""
        obj = ObjectiveFunction(objective_type='mean_variance')
        
        weights = cp.Variable(sample_data['n_assets'])
        
        expr = obj.mean_variance_objective(
            weights,
            sample_data['expected_returns'],
            sample_data['cov_matrix'],
            risk_aversion=1.0
        )
        
        assert isinstance(expr, cp.Expression)
    
    def test_minimum_variance_objective(self, sample_data):
        """Test minimum variance objective."""
        obj = ObjectiveFunction(objective_type='min_variance')
        
        weights = cp.Variable(sample_data['n_assets'])
        
        expr = obj.minimum_variance_objective(
            weights,
            sample_data['cov_matrix']
        )
        
        assert isinstance(expr, cp.Expression)
    
    def test_risk_parity_objective(self, sample_data):
        """Test risk parity objective."""
        obj = ObjectiveFunction(objective_type='risk_parity')
        
        weights = cp.Variable(sample_data['n_assets'])
        
        expr = obj.risk_parity_objective(
            weights,
            sample_data['cov_matrix']
        )
        
        assert isinstance(expr, cp.Expression)
    
    def test_build_objective_sharpe(self, sample_data):
        """Test building Sharpe objective."""
        obj = ObjectiveFunction(objective_type='sharpe')
        
        weights = cp.Variable(sample_data['n_assets'])
        
        expr = obj.build_objective(
            weights=weights,
            expected_returns=sample_data['expected_returns'],
            cov_matrix=sample_data['cov_matrix']
        )
        
        assert expr is not None
    
    def test_build_objective_min_variance(self, sample_data):
        """Test building min variance objective."""
        obj = ObjectiveFunction(objective_type='min_variance')
        
        weights = cp.Variable(sample_data['n_assets'])
        
        expr = obj.build_objective(
            weights=weights,
            cov_matrix=sample_data['cov_matrix']
        )
        
        assert expr is not None


class TestObjectiveTypes:
    """Test different objective types."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data."""
        return {
            'expected_returns': np.array([0.001, 0.002, 0.0015]),
            'cov_matrix': np.array([
                [0.01, 0.005, 0.002],
                [0.005, 0.015, 0.003],
                [0.002, 0.003, 0.012]
            ]),
            'volatilities': np.array([0.1, 0.12, 0.11])
        }
    
    def test_all_objectives_buildable(self, sample_data):
        """Test that all objective types can be built."""
        objectives = ['sharpe', 'mean_variance', 'min_variance', 'risk_parity', 'max_diversification']
        
        weights = cp.Variable(3)
        
        for obj_type in objectives:
            obj = ObjectiveFunction(objective_type=obj_type)
            
            try:
                expr = obj.build_objective(
                    weights=weights,
                    **sample_data
                )
                assert expr is not None
            except Exception as e:
                pytest.fail(f"Failed to build {obj_type}: {str(e)}")