"""
Tests for DataStorage class.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
from src.data.storage import DataStorage


class TestDataStorage:
    """Test suite for DataStorage."""
    
    @pytest.fixture
    def storage(self, tmp_path):
        """Create storage instance with temporary directories."""
        raw_dir = tmp_path / "raw"
        processed_dir = tmp_path / "processed"
        
        return DataStorage(
            raw_data_dir=str(raw_dir),
            processed_data_dir=str(processed_dir)
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        np.random.seed(42)
        n = len(dates)
        
        return pd.DataFrame({
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 102 + np.random.randn(n).cumsum(),
            'low': 98 + np.random.randn(n).cumsum(),
            'close': 100 + np.random.randn(n).cumsum(),
            'volume': np.random.randint(1000000, 10000000, n)
        }, index=dates)
    
    def test_initialization(self, storage):
        """Test storage initialization."""
        assert storage.raw_data_dir.exists()
        assert storage.processed_data_dir.exists()
    
    def test_save_and_load_raw_data(self, storage, sample_data):
        """Test saving and loading raw data."""
        # Save data
        path = storage.save_raw_data("TEST", sample_data, overwrite=True)
        assert path.exists()
        
        # Load data
        loaded_data = storage.load_raw_data("TEST")
        
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
        pd.testing.assert_frame_equal(loaded_data, sample_data, check_freq=False)
    
    def test_save_and_load_processed_data(self, storage, sample_data):
        """Test saving and loading processed data."""
        # Save data
        path = storage.save_processed_data("TEST", sample_data, overwrite=True)
        assert path.exists()
        
        # Load data
        loaded_data = storage.load_processed_data("TEST")
        
        assert loaded_data is not None
        assert len(loaded_data) == len(sample_data)
        pd.testing.assert_frame_equal(loaded_data, sample_data, check_freq=False)
    
    def test_load_nonexistent_data(self, storage):
        """Test loading data that doesn't exist."""
        data = storage.load_raw_data("NONEXISTENT")
        assert data is None
        
        data = storage.load_processed_data("NONEXISTENT")
        assert data is None
    
    def test_overwrite_protection(self, storage, sample_data):
        """Test that overwrite=False prevents data loss."""
        # Save initial data
        storage.save_raw_data("TEST", sample_data, overwrite=True)
        
        # Modify data
        modified_data = sample_data.copy()
        modified_data['close'] = modified_data['close'] * 2
        
        # Try to save without overwrite
        storage.save_raw_data("TEST", modified_data, overwrite=False)
        
        # Load and verify original data is preserved
        loaded_data = storage.load_raw_data("TEST")
        pd.testing.assert_frame_equal(loaded_data, sample_data, check_freq=False)
    
    def test_save_batch(self, storage, sample_data):
        """Test batch saving of multiple assets."""
        data_dict = {
            'ASSET1': sample_data,
            'ASSET2': sample_data.copy(),
            'ASSET3': sample_data.copy()
        }
        
        paths = storage.save_batch(data_dict, data_type='raw')
        
        assert len(paths) == 3
        assert all(p.exists() for p in paths)
    
    def test_load_batch(self, storage, sample_data):
        """Test batch loading of multiple assets."""
        # Save some data first
        data_dict = {
            'ASSET1': sample_data,
            'ASSET2': sample_data.copy()
        }
        storage.save_batch(data_dict, data_type='processed')
        
        # Load batch
        loaded_dict = storage.load_batch(['ASSET1', 'ASSET2', 'ASSET3'], data_type='processed')
        
        assert len(loaded_dict) == 2  # ASSET3 doesn't exist
        assert 'ASSET1' in loaded_dict
        assert 'ASSET2' in loaded_dict
    
    def test_list_available_assets(self, storage, sample_data):
        """Test listing available assets."""
        # Save some data
        data_dict = {
            'SPY': sample_data,
            'QQQ': sample_data.copy(),
            'GLD': sample_data.copy()
        }
        storage.save_batch(data_dict, data_type='raw')
        
        # List assets
        available = storage.list_available_assets(data_type='raw')
        
        assert len(available) == 3
        assert 'GLD' in available
        assert 'QQQ' in available
        assert 'SPY' in available
    
    def test_get_data_info(self, storage, sample_data):
        """Test retrieving data metadata."""
        storage.save_raw_data("TEST", sample_data, overwrite=True)
        
        info = storage.get_data_info("TEST", data_type='raw')
        
        assert info is not None
        assert info['ticker'] == "TEST"
        assert info['rows'] == len(sample_data)
        assert 'start_date' in info
        assert 'end_date' in info
        assert 'file_size_mb' in info
    
    def test_delete_asset(self, storage, sample_data):
        """Test deleting stored data."""
        # Save data
        storage.save_raw_data("TEST", sample_data, overwrite=True)
        assert storage.load_raw_data("TEST") is not None
        
        # Delete data
        success = storage.delete_asset("TEST", data_type='raw')
        assert success is True
        
        # Verify deletion
        assert storage.load_raw_data("TEST") is None
    
    def test_delete_nonexistent_asset(self, storage):
        """Test deleting asset that doesn't exist."""
        success = storage.delete_asset("NONEXISTENT", data_type='raw')
        assert success is False


class TestStoragePerformance:
    """Test storage performance and efficiency."""
    
    @pytest.fixture
    def storage(self, tmp_path):
        """Create storage instance."""
        return DataStorage(
            raw_data_dir=str(tmp_path / "raw"),
            processed_data_dir=str(tmp_path / "processed")
        )
    
    @pytest.fixture
    def large_data(self):
        """Create larger dataset for performance testing."""
        dates = pd.date_range(start='2010-01-01', end='2023-12-31', freq='D')
        
        np.random.seed(42)
        n = len(dates)
        
        return pd.DataFrame({
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 102 + np.random.randn(n).cumsum(),
            'low': 98 + np.random.randn(n).cumsum(),
            'close': 100 + np.random.randn(n).cumsum(),
            'volume': np.random.randint(1000000, 10000000, n)
        }, index=dates)
    
    def test_parquet_compression(self, storage, large_data):
        """Test that Parquet compression is effective."""
        # Save data
        path = storage.save_raw_data("TEST", large_data, overwrite=True)
        
        # Check file size (should be much smaller than uncompressed)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        
        # Parquet with compression should be < 5 MB for this data
        assert file_size_mb < 5.0
    
    def test_save_load_roundtrip(self, storage, large_data):
        """Test that data survives save/load roundtrip."""
        storage.save_raw_data("TEST", large_data, overwrite=True)
        loaded_data = storage.load_raw_data("TEST")
        
        pd.testing.assert_frame_equal(loaded_data, large_data, check_freq=False)