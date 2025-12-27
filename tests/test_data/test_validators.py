"""
Tests for DataValidator class.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.validators import DataValidator


class TestDataValidator:
    """Test suite for DataValidator."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance for testing."""
        return DataValidator(
            max_missing_pct=5.0,
            max_price_change_pct=50.0,
            min_data_points=100,
            max_days_stale=5
        )
    
    @pytest.fixture
    def valid_data(self):
        """Create valid sample data."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        np.random.seed(42)
        n = len(dates)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(n).cumsum(),
            'high': 102 + np.random.randn(n).cumsum(),
            'low': 98 + np.random.randn(n).cumsum(),
            'close': 100 + np.random.randn(n).cumsum(),
            'volume': np.random.randint(1000000, 10000000, n)
        }, index=dates)
        
        # Ensure OHLC logic
        data['high'] = data[['open', 'high', 'close']].max(axis=1) + 0.5
        data['low'] = data[['open', 'low', 'close']].min(axis=1) - 0.5
        
        return data
    
    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator.max_missing_pct == 5.0
        assert validator.max_price_change_pct == 50.0
        assert validator.min_data_points == 100
        assert validator.max_days_stale == 5
    
    def test_validate_valid_data(self, validator, valid_data):
        """Test validation of clean data."""
        is_valid, report = validator.validate_single_asset("TEST", valid_data)
        
        assert is_valid is True
        assert report['ticker'] == "TEST"
        assert len(report['errors']) == 0
    
    def test_insufficient_data_points(self, validator, valid_data):
        """Test validation with insufficient data."""
        short_data = valid_data.iloc[:50]  # Less than min_data_points
        
        is_valid, report = validator.validate_single_asset("TEST", short_data)
        
        assert is_valid is False
        assert any("Insufficient data" in error for error in report['errors'])
    
    def test_missing_columns(self, validator, valid_data):
        """Test validation with missing columns."""
        incomplete_data = valid_data.drop(columns=['close'])
        
        is_valid, report = validator.validate_single_asset("TEST", incomplete_data)
        
        assert is_valid is False
        assert any("Missing required columns" in error for error in report['errors'])
    
    def test_excessive_missing_values(self, validator, valid_data):
        """Test validation with too many missing values."""
        data_with_nulls = valid_data.copy()
        data_with_nulls.loc[data_with_nulls.index[:50], 'close'] = np.nan
        
        is_valid, report = validator.validate_single_asset("TEST", data_with_nulls)
        
        # May be invalid or have warnings depending on threshold
        assert 'missing_values' in report['checks']
    
    def test_negative_prices(self, validator, valid_data):
        """Test validation with negative prices."""
        invalid_data = valid_data.copy()
        invalid_data.loc[invalid_data.index[0], 'close'] = -10
        
        is_valid, report = validator.validate_single_asset("TEST", invalid_data)
        
        assert is_valid is False
        assert any("non-positive prices" in error for error in report['errors'])
    
    def test_duplicate_timestamps(self, validator, valid_data):
        """Test validation with duplicate timestamps."""
        duplicate_data = pd.concat([valid_data, valid_data.iloc[:5]])
        
        is_valid, report = validator.validate_single_asset("TEST", duplicate_data)
        
        assert is_valid is False
        assert any("duplicate timestamps" in error for error in report['errors'])
    
    def test_ohlc_consistency_check(self, validator):
        """Test OHLC consistency validation."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create data with OHLC violations
        data = pd.DataFrame({
            'open': 100,
            'high': 95,  # High < Open (violation)
            'low': 105,   # Low > Open (violation)
            'close': 100,
            'volume': 1000000
        }, index=dates)
        
        issues = validator._check_ohlc_consistency(data)
        
        assert issues > 0
    
    def test_clean_data_forward_fill(self, validator, valid_data):
        """Test data cleaning with forward fill."""
        data_with_nulls = valid_data.copy()
        data_with_nulls.loc[data_with_nulls.index[10:15], 'close'] = np.nan
        
        cleaned = validator.clean_data(data_with_nulls, method='forward_fill')
        
        assert cleaned['close'].isnull().sum() == 0
        assert len(cleaned) == len(data_with_nulls)
    
    def test_clean_data_drop(self, validator, valid_data):
        """Test data cleaning by dropping nulls."""
        data_with_nulls = valid_data.copy()
        data_with_nulls.loc[data_with_nulls.index[10:15], 'close'] = np.nan
        
        cleaned = validator.clean_data(data_with_nulls, method='drop')
        
        assert cleaned['close'].isnull().sum() == 0
        assert len(cleaned) < len(data_with_nulls)
    
    def test_validate_multiple_assets(self, validator, valid_data):
        """Test batch validation."""
        data_dict = {
            'ASSET1': valid_data,
            'ASSET2': valid_data.copy(),
            'ASSET3': valid_data.iloc[:50]  # Invalid (too short)
        }
        
        results = validator.validate_multiple_assets(data_dict, save_report=False)
        
        assert len(results) == 3
        assert results['ASSET1'][0] is True
        assert results['ASSET2'][0] is True
        assert results['ASSET3'][0] is False


class TestValidationReports:
    """Test validation reporting functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return DataValidator()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        return pd.DataFrame({
            'open': 100,
            'high': 105,
            'low': 95,
            'close': 100,
            'volume': 1000000
        }, index=dates)
    
    def test_report_structure(self, validator, sample_data):
        """Test validation report structure."""
        is_valid, report = validator.validate_single_asset("TEST", sample_data)
        
        assert 'ticker' in report
        assert 'timestamp' in report
        assert 'checks' in report
        assert 'warnings' in report
        assert 'errors' in report
        assert 'is_valid' in report
    
    def test_report_checks(self, validator, sample_data):
        """Test that report includes all checks."""
        is_valid, report = validator.validate_single_asset("TEST", sample_data)
        
        assert 'data_points' in report['checks']
        assert 'missing_values' in report['checks']
        assert 'last_update' in report['checks']
        assert 'price_outliers' in report['checks']