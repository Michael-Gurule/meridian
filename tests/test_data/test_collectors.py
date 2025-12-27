"""
Tests for MarketDataCollector class.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.data.collectors import MarketDataCollector


class TestMarketDataCollector:
    """Test suite for MarketDataCollector."""
    
    @pytest.fixture
    def collector(self):
        """Create collector instance for testing."""
        return MarketDataCollector(
            start_date="2023-01-01",
            end_date="2023-12-31",
            retry_attempts=2,
            retry_delay=1
        )
    
    def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector.start_date == "2023-01-01"
        assert collector.end_date == "2023-12-31"
        assert collector.retry_attempts == 2
        assert collector.retry_delay == 1
    
    def test_download_single_asset_success(self, collector):
        """Test successful download of a single asset."""
        data = collector.download_single_asset("SPY")
        
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'close' in data.columns
        assert 'volume' in data.columns
    
    def test_download_single_asset_invalid_ticker(self, collector):
        """Test download with invalid ticker."""
        data = collector.download_single_asset("INVALID_TICKER_XYZ")
        
        # Should return None for invalid ticker
        assert data is None
    
    def test_download_multiple_assets(self, collector):
        """Test batch download of multiple assets."""
        tickers = ["SPY", "QQQ", "GLD"]
        data_dict = collector.download_multiple_assets(tickers, delay_between=0.1)
        
        assert isinstance(data_dict, dict)
        assert len(data_dict) > 0
        
        for ticker in data_dict.keys():
            assert ticker in tickers
            assert isinstance(data_dict[ticker], pd.DataFrame)
    
    def test_data_columns(self, collector):
        """Test that downloaded data has required columns."""
        data = collector.download_single_asset("SPY")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in data.columns
    
    def test_data_index(self, collector):
        """Test that data has datetime index."""
        data = collector.download_single_asset("SPY")
        
        assert isinstance(data.index, pd.DatetimeIndex)
        assert data.index.is_monotonic_increasing
    
    def test_get_latest_price(self, collector):
        """Test retrieval of latest price."""
        price = collector.get_latest_price("SPY")
        
        assert price is not None
        assert isinstance(price, float)
        assert price > 0
    
    def test_update_single_asset(self, collector):
        """Test updating existing data."""
        # Download initial data
        initial_data = collector.download_single_asset("SPY")
        
        # Simulate older data by removing recent rows
        old_data = initial_data.iloc[:-5]
        
        # Update with recent data
        updated_data = collector.update_single_asset("SPY", old_data)
        
        assert len(updated_data) >= len(old_data)
        assert updated_data.index.max() >= old_data.index.max()


class TestDataQuality:
    """Test data quality after download."""
    
    @pytest.fixture
    def sample_data(self):
        """Get sample data for testing."""
        collector = MarketDataCollector(start_date="2023-01-01", end_date="2023-12-31")
        return collector.download_single_asset("SPY")
    
    def test_no_null_prices(self, sample_data):
        """Test that critical price columns have no nulls."""
        assert sample_data['close'].isnull().sum() == 0
    
    def test_positive_prices(self, sample_data):
        """Test that all prices are positive."""
        assert (sample_data['close'] > 0).all()
        assert (sample_data['open'] > 0).all()
        assert (sample_data['high'] > 0).all()
        assert (sample_data['low'] > 0).all()
    
    def test_ohlc_logic(self, sample_data):
        """Test OHLC logical consistency."""
        # High should be >= Low
        assert (sample_data['high'] >= sample_data['low']).all()
        
        # High should be >= Open and Close
        assert (sample_data['high'] >= sample_data['open']).all()
        assert (sample_data['high'] >= sample_data['close']).all()
        
        # Low should be <= Open and Close
        assert (sample_data['low'] <= sample_data['open']).all()
        assert (sample_data['low'] <= sample_data['close']).all()
    
    def test_volume_non_negative(self, sample_data):
        """Test that volume is non-negative."""
        assert (sample_data['volume'] >= 0).all()
    
    def test_no_duplicate_dates(self, sample_data):
        """Test that there are no duplicate dates."""
        assert not sample_data.index.duplicated().any()