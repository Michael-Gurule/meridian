"""
Tests for VolatilityModel class.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.volatility import VolatilityModel


class TestVolatilityModel:
    """Test suite for VolatilityModel."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return series."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        
        # Create returns with changing volatility
        vol1 = np.random.randn(250) * 0.01  # Low vol period
        vol2 = np.random.randn(250) * 0.03  # High vol period
        returns = np.concatenate([vol1, vol2])
        
        return pd.Series(returns, index=dates, name='ASSET')
    
    def test_initialization(self):
        """Test model initialization."""
        model = VolatilityModel(method='ewm', window=30)
        assert model.method == 'ewm'
        assert model.window == 30
    
    def test_historical_volatility(self, sample_returns):
        """Test historical volatility estimation."""
        model = VolatilityModel(method='historical', window=30)
        vol = model.estimate(sample_returns, annualize=False)
        
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(sample_returns)
        assert not vol.isna().all()
    
    def test_ewm_volatility(self, sample_returns):
        """Test EWMA volatility."""
        model = VolatilityModel(method='ewm', ewm_halflife=30)
        vol = model.estimate(sample_returns, annualize=False)
        
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(sample_returns)
    
    def test_garch_volatility(self, sample_returns):
        """Test GARCH volatility estimation."""
        model = VolatilityModel(method='garch')
        vol = model.estimate(sample_returns, annualize=False)
        
        assert isinstance(vol, pd.Series)
        # GARCH may have some NaN at start
        assert vol.notna().sum() > 0
    
    def test_annualization(self, sample_returns):
        """Test volatility annualization."""
        model = VolatilityModel(method='ewm')
        
        vol_daily = model.estimate(sample_returns, annualize=False)
        vol_annual = model.estimate(sample_returns, annualize=True)
        
        # Annual should be ~sqrt(252) times daily
        ratio = vol_annual / vol_daily
        expected_ratio = np.sqrt(252)
        
        assert np.allclose(ratio.dropna(), expected_ratio, rtol=0.01)
    
    def test_volatility_increases_in_high_vol_period(self, sample_returns):
        """Test that volatility increases in high volatility period."""
        model = VolatilityModel(method='ewm', ewm_halflife=20)
        vol = model.estimate(sample_returns, annualize=False)
        
        # Compare early vs late period
        early_vol = vol.iloc[100:150].mean()
        late_vol = vol.iloc[350:400].mean()
        
        # Late period should have higher volatility
        assert late_vol > early_vol
    
    def test_forecast_volatility(self, sample_returns):
        """Test volatility forecasting."""
        model = VolatilityModel(method='ewm')
        forecast = model.forecast_volatility(sample_returns, horizon=5, annualize=False)
        
        assert isinstance(forecast, float)
        assert forecast > 0
    
    def test_realized_volatility(self, sample_returns):
        """Test realized volatility calculation."""
        model = VolatilityModel(method='ewm')
        realized_vol = model.realized_volatility(sample_returns, window=30, annualize=False)
        
        assert isinstance(realized_vol, pd.Series)
        assert len(realized_vol) == len(sample_returns)
    
    def test_volatility_ratio(self, sample_returns):
        """Test volatility ratio calculation."""
        model = VolatilityModel(method='ewm')
        vol_ratio = model.volatility_ratio(sample_returns, short_window=10, long_window=60)
        
        assert isinstance(vol_ratio, pd.Series)
        assert len(vol_ratio) == len(sample_returns)
        
        # Ratio should be positive
        assert (vol_ratio.dropna() > 0).all()


class TestVolatilityComparison:
    """Test comparison of volatility methods."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample data."""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
        returns = np.random.randn(500) * 0.02
        return pd.Series(returns, index=dates)
    
    def test_methods_produce_different_estimates(self, sample_returns):
        """Test that different methods give different results."""
        hist_vol = VolatilityModel(method='historical').estimate(sample_returns)
        ewm_vol = VolatilityModel(method='ewm').estimate(sample_returns)
        
        # Align series to same index (different methods may drop different rows)
        hist_clean = hist_vol.dropna()
        ewm_clean = ewm_vol.dropna()
    
        # Align to common index
        aligned_hist, aligned_ewm = hist_clean.align(ewm_clean, join='inner')
        
        # Should not be identical
        assert not np.allclose(aligned_hist.values, aligned_ewm.values)
        
        
    
    def test_all_methods_positive(self, sample_returns):
        """Test that all methods produce positive volatility."""
        methods = ['historical', 'ewm', 'garch']
        
        for method in methods:
            model = VolatilityModel(method=method)
            vol = model.estimate(sample_returns, annualize=False)
            
            assert (vol.dropna() > 0).all(), f"{method} produced non-positive vol"