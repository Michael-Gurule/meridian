"""
Tests for backtesting components.
"""

import pytest
import numpy as np
import pandas as pd
from src.backtesting.engine import BacktestEngine
from src.backtesting.execution import ExecutionSimulator
from src.backtesting.metrics import PerformanceMetrics
from src.optimization.optimizer import PortfolioOptimizer
from src.models.covariance import CovarianceEstimator
from src.models.returns import ReturnEstimator


class TestExecutionSimulator:
    """Test suite for ExecutionSimulator."""
    
    @pytest.fixture
    def sample_execution_data(self):
        """Create sample execution data."""
        return {
            'current_weights': np.array([0.25, 0.25, 0.25, 0.25]),
            'target_weights': np.array([0.3, 0.3, 0.2, 0.2]),
            'prices': np.array([100, 150, 200, 50]),
            'portfolio_value': 1000000.0
        }
    
    def test_initialization(self):
        """Test simulator initialization."""
        sim = ExecutionSimulator()
        assert sim.cost_model is not None
    
    def test_execute_rebalance(self, sample_execution_data):
        """Test trade execution."""
        sim = ExecutionSimulator()
        
        result = sim.execute_rebalance(**sample_execution_data)
        
        assert 'executed_weights' in result
        assert 'total_cost' in result
        assert 'cost_pct' in result
        
        # Weights should sum to 1
        assert np.isclose(result['executed_weights'].sum(), 1.0)
        
        # Cost should be positive
        assert result['total_cost'] >= 0


class TestPerformanceMetrics:
    """Test suite for PerformanceMetrics."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return series."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=252, freq='D')
        returns = pd.Series(
            np.random.randn(252) * 0.01 + 0.0003,
            index=dates,
            name='returns'
        )
        return returns
    
    def test_initialization(self):
        """Test metrics calculator initialization."""
        metrics = PerformanceMetrics()
        assert metrics.risk_free_rate == 0.0
    
    def test_calculate_all_metrics(self, sample_returns):
        """Test comprehensive metrics calculation."""
        metrics = PerformanceMetrics()
        
        result = metrics.calculate_all_metrics(sample_returns)
        
        required_metrics = [
            'total_return', 'annual_return', 'volatility',
            'sharpe_ratio', 'max_drawdown', 'var_95'
        ]
        
        for metric in required_metrics:
            assert metric in result
            assert not np.isnan(result[metric])
    
    def test_performance_summary(self, sample_returns):
        """Test performance summary generation."""
        metrics = PerformanceMetrics()
        
        summary = metrics.performance_summary(sample_returns)
        
        assert isinstance(summary, pd.DataFrame)
        assert 'Metric' in summary.columns
        assert 'Value' in summary.columns


class TestBacktestEngine:
    """Test suite for BacktestEngine."""
    
    @pytest.fixture
    def sample_prices(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        
        n_assets = 3
        prices = pd.DataFrame(
            100 * (1 + np.random.randn(500, n_assets) * 0.01).cumprod(axis=0),
            index=dates,
            columns=['ASSET1', 'ASSET2', 'ASSET3']
        )
        
        return prices
    
    def test_initialization(self):
        """Test backtest engine initialization."""
        engine = BacktestEngine(rebalance_frequency='M')
        assert engine.rebalance_frequency == 'M'
    
    def test_run_backtest(self, sample_prices):
        """Test running a backtest."""
        engine = BacktestEngine(
            rebalance_frequency='M',
            lookback_window=60,
            initial_capital=100000.0
        )
        
        optimizer = PortfolioOptimizer(
            objective_type='min_variance',
            constraints={'long_only': True}
        )
        
        cov_estimator = CovarianceEstimator(method='sample')
        return_estimator = ReturnEstimator(method='historical')
        
        result = engine.run(
            prices=sample_prices,
            optimizer=optimizer,
            cov_estimator=cov_estimator,
            return_estimator=return_estimator
        )
        
        assert 'portfolio_values' in result
        assert 'returns' in result
        assert 'metrics' in result
        assert 'final_value' in result
        
        # Final value should be positive
        assert result['final_value'] > 0