"""
Tests for MarketDataCollector class.
Uses mocking to avoid rate limiting from Yahoo Finance API.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from src.data.collectors import MarketDataCollector


def create_mock_ohlcv_data(
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    base_price: float = 400.0
) -> pd.DataFrame:
    """Create realistic mock OHLCV data for testing."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n = len(dates)

    # Generate realistic price movements
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, n)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        'high': prices * (1 + np.random.uniform(0.001, 0.02, n)),
        'low': prices * (1 - np.random.uniform(0.001, 0.02, n)),
        'close': prices,
        'volume': np.random.uniform(50_000_000, 150_000_000, n).astype(int)
    }, index=dates)

    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)

    data.index.name = 'Date'
    return data


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

    @pytest.fixture
    def mock_data(self):
        """Create mock OHLCV data."""
        return create_mock_ohlcv_data()

    def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector.start_date == "2023-01-01"
        assert collector.end_date == "2023-12-31"
        assert collector.retry_attempts == 2
        assert collector.retry_delay == 1

    @patch('yfinance.download')
    def test_download_single_asset_success(self, mock_download, collector, mock_data):
        """Test successful download of a single asset."""
        mock_download.return_value = mock_data.copy()
        mock_download.return_value.columns = pd.MultiIndex.from_product(
            [['Open', 'High', 'Low', 'Close', 'Volume'], ['SPY']]
        )

        data = collector.download_single_asset("SPY")

        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'close' in data.columns
        assert 'volume' in data.columns

    def test_download_single_asset_invalid_ticker(self, collector):
        """Test download with invalid ticker."""
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = pd.DataFrame()
            data = collector.download_single_asset("INVALID_TICKER_XYZ")

            # Should return None for invalid ticker (empty data)
            assert data is None

    @patch('yfinance.download')
    def test_download_multiple_assets(self, mock_download, collector):
        """Test batch download of multiple assets."""
        # Create mock data for multiple tickers
        tickers = ["SPY", "QQQ", "GLD"]
        mock_dfs = {}
        for i, ticker in enumerate(tickers):
            mock_dfs[ticker] = create_mock_ohlcv_data(base_price=100 * (i + 1))

        def download_side_effect(ticker, *args, **kwargs):
            if isinstance(ticker, str):
                df = mock_dfs.get(ticker, pd.DataFrame())
                if not df.empty:
                    df.columns = pd.MultiIndex.from_product(
                        [['Open', 'High', 'Low', 'Close', 'Volume'], [ticker]]
                    )
                return df
            return pd.DataFrame()

        mock_download.side_effect = download_side_effect

        data_dict = collector.download_multiple_assets(tickers, delay_between=0)

        assert isinstance(data_dict, dict)
        assert len(data_dict) > 0

        for ticker in data_dict.keys():
            assert ticker in tickers
            assert isinstance(data_dict[ticker], pd.DataFrame)

    @patch('yfinance.download')
    def test_data_columns(self, mock_download, collector, mock_data):
        """Test that downloaded data has required columns."""
        mock_download.return_value = mock_data.copy()
        mock_download.return_value.columns = pd.MultiIndex.from_product(
            [['Open', 'High', 'Low', 'Close', 'Volume'], ['SPY']]
        )

        data = collector.download_single_asset("SPY")

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in data.columns

    @patch('yfinance.download')
    def test_data_index(self, mock_download, collector, mock_data):
        """Test that data has datetime index."""
        mock_download.return_value = mock_data.copy()
        mock_download.return_value.columns = pd.MultiIndex.from_product(
            [['Open', 'High', 'Low', 'Close', 'Volume'], ['SPY']]
        )

        data = collector.download_single_asset("SPY")

        assert isinstance(data.index, pd.DatetimeIndex)
        assert data.index.is_monotonic_increasing

    @patch('yfinance.Ticker')
    def test_get_latest_price(self, mock_ticker_class, collector):
        """Test retrieval of latest price."""
        mock_ticker = MagicMock()
        mock_ticker.info = {'regularMarketPrice': 450.50}
        mock_ticker_class.return_value = mock_ticker

        price = collector.get_latest_price("SPY")

        assert price is not None
        assert isinstance(price, float)
        assert price > 0

    @patch('yfinance.download')
    def test_update_single_asset(self, mock_download, collector, mock_data):
        """Test updating existing data."""
        mock_download.return_value = mock_data.copy()
        mock_download.return_value.columns = pd.MultiIndex.from_product(
            [['Open', 'High', 'Low', 'Close', 'Volume'], ['SPY']]
        )

        # Download initial data
        initial_data = collector.download_single_asset("SPY")

        # Simulate older data by removing recent rows
        old_data = initial_data.iloc[:-5]

        # Update with recent data
        updated_data = collector.update_single_asset("SPY", old_data)

        assert len(updated_data) >= len(old_data)
        assert updated_data.index.max() >= old_data.index.max()


class TestDataQuality:
    """Test data quality validation using mock data."""

    @pytest.fixture
    def sample_data(self):
        """Get sample mock data for testing."""
        return create_mock_ohlcv_data()

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
