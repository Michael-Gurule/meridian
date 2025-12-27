"""
Data validation and quality checks for market data.

Performs comprehensive validation to ensure data integrity before use in
portfolio optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Validates market data quality and completeness.
    
    Checks for missing values, outliers, stale data, and other quality issues.
    """
    
    def __init__(
        self,
        max_missing_pct: float = 5.0,
        max_price_change_pct: float = 50.0,
        min_data_points: int = 252,
        max_days_stale: int = 5
    ):
        """
        Initialize data validator with quality thresholds.
        
        Parameters
        ----------
        max_missing_pct : float
            Maximum allowed percentage of missing values
        max_price_change_pct : float
            Maximum allowed single-day price change (percentage)
        min_data_points : int
            Minimum required number of data points
        max_days_stale : int
            Maximum days since last update before flagging as stale
        """
        self.max_missing_pct = max_missing_pct
        self.max_price_change_pct = max_price_change_pct
        self.min_data_points = min_data_points
        self.max_days_stale = max_days_stale
        
        logger.info("Initialized DataValidator with quality thresholds")
    
    def validate_single_asset(
        self,
        ticker: str,
        data: pd.DataFrame,
        verbose: bool = True
    ) -> Tuple[bool, Dict]:
        """
        Validate data for a single asset.
        
        Parameters
        ----------
        ticker : str
            Asset ticker symbol
        data : pd.DataFrame
            OHLCV data to validate
        verbose : bool
            Whether to log detailed validation results
        
        Returns
        -------
        Tuple[bool, Dict]
            (is_valid, validation_report)
        """
        report = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'warnings': [],
            'errors': [],
            'is_valid': True
        }
        
        # Check 1: Minimum data points
        n_points = len(data)
        report['checks']['data_points'] = n_points
        
        if n_points < self.min_data_points:
            report['errors'].append(
                f"Insufficient data: {n_points} points (minimum: {self.min_data_points})"
            )
            report['is_valid'] = False
        
        # Check 2: Required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            report['errors'].append(f"Missing required columns: {missing_cols}")
            report['is_valid'] = False
            return report['is_valid'], report
        
        # Check 3: Missing values
        missing_counts = data[required_cols].isnull().sum()
        total_missing = missing_counts.sum()
        missing_pct = (total_missing / (len(data) * len(required_cols))) * 100
        
        report['checks']['missing_values'] = {
            'total': int(total_missing),
            'percentage': round(missing_pct, 2),
            'by_column': missing_counts.to_dict()
        }
        
        if missing_pct > self.max_missing_pct:
            report['errors'].append(
                f"Excessive missing values: {missing_pct:.2f}% (max: {self.max_missing_pct}%)"
            )
            report['is_valid'] = False
        elif missing_pct > 0:
            report['warnings'].append(
                f"Contains {missing_pct:.2f}% missing values"
            )
        
        # Check 4: Data freshness
        last_date = data.index.max()
        days_since_update = (datetime.now() - last_date).days
        
        report['checks']['last_update'] = {
            'date': last_date.isoformat(),
            'days_ago': days_since_update
        }
        
        if days_since_update > self.max_days_stale:
            report['warnings'].append(
                f"Stale data: last update {days_since_update} days ago"
            )
        
        # Check 5: Price sanity checks
        close_prices = data['close'].dropna()
        
        if len(close_prices) > 0:
            # Check for negative or zero prices
            invalid_prices = (close_prices <= 0).sum()
            if invalid_prices > 0:
                report['errors'].append(
                    f"Found {invalid_prices} non-positive prices"
                )
                report['is_valid'] = False
            
            # Check for extreme price changes
            returns = close_prices.pct_change().dropna()
            extreme_returns = returns[abs(returns * 100) > self.max_price_change_pct]
            
            report['checks']['price_outliers'] = {
                'count': len(extreme_returns),
                'max_change_pct': round(returns.abs().max() * 100, 2) if len(returns) > 0 else 0
            }
            
            if len(extreme_returns) > 0:
                report['warnings'].append(
                    f"Found {len(extreme_returns)} extreme price changes (>{self.max_price_change_pct}%)"
                )
        
        # Check 6: OHLC consistency
        ohlc_issues = self._check_ohlc_consistency(data)
        if ohlc_issues > 0:
            report['warnings'].append(
                f"Found {ohlc_issues} OHLC inconsistencies (high < low, close outside range)"
            )
            report['checks']['ohlc_inconsistencies'] = ohlc_issues
        
        # Check 7: Volume data
        if 'volume' in data.columns:
            zero_volume_days = (data['volume'] == 0).sum()
            zero_volume_pct = (zero_volume_days / len(data)) * 100
            
            report['checks']['zero_volume'] = {
                'count': int(zero_volume_days),
                'percentage': round(zero_volume_pct, 2)
            }
            
            if zero_volume_pct > 10:
                report['warnings'].append(
                    f"High proportion of zero volume days: {zero_volume_pct:.2f}%"
                )
        
        # Check 8: Duplicate timestamps
        duplicate_dates = data.index.duplicated().sum()
        if duplicate_dates > 0:
            report['errors'].append(f"Found {duplicate_dates} duplicate timestamps")
            report['is_valid'] = False
        
        # Check 9: Chronological order
        if not data.index.is_monotonic_increasing:
            report['errors'].append("Data is not in chronological order")
            report['is_valid'] = False
        
        if verbose:
            self._log_validation_report(report)
        
        return report['is_valid'], report
    
    def _check_ohlc_consistency(self, data: pd.DataFrame) -> int:
        """
        Check for OHLC logical consistency.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data
        
        Returns
        -------
        int
            Number of inconsistencies found
        """
        issues = 0
        
        # High should be >= Low
        issues += (data['high'] < data['low']).sum()
        
        # High should be >= Open and Close
        issues += (data['high'] < data['open']).sum()
        issues += (data['high'] < data['close']).sum()
        
        # Low should be <= Open and Close
        issues += (data['low'] > data['open']).sum()
        issues += (data['low'] > data['close']).sum()
        
        return int(issues)
    
    def validate_multiple_assets(
        self,
        data_dict: Dict[str, pd.DataFrame],
        save_report: bool = True,
        report_path: Optional[Path] = None
    ) -> Dict[str, Tuple[bool, Dict]]:
        """
        Validate data for multiple assets.
        
        Parameters
        ----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary mapping tickers to dataframes
        save_report : bool
            Whether to save validation report to file
        report_path : Path, optional
            Path to save validation report
        
        Returns
        -------
        Dict[str, Tuple[bool, Dict]]
            Validation results for each asset
        """
        logger.info(f"Validating {len(data_dict)} assets")
        
        results = {}
        valid_count = 0
        invalid_count = 0
        
        for ticker, data in data_dict.items():
            is_valid, report = self.validate_single_asset(ticker, data, verbose=False)
            results[ticker] = (is_valid, report)
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        
        logger.info(
            f"Validation complete: {valid_count} valid, {invalid_count} invalid"
        )
        
        # Save comprehensive report
        if save_report:
            self._save_validation_report(results, report_path)
        
        return results
    
    def _log_validation_report(self, report: Dict):
        """Log validation report details."""
        ticker = report['ticker']
        
        if report['is_valid']:
            logger.info(f"✓ {ticker} passed validation")
        else:
            logger.error(f"✗ {ticker} failed validation")
        
        for error in report['errors']:
            logger.error(f"  ERROR: {error}")
        
        for warning in report['warnings']:
            logger.warning(f"  WARNING: {warning}")
    
    def _save_validation_report(
        self,
        results: Dict[str, Tuple[bool, Dict]],
        report_path: Optional[Path] = None
    ):
        """
        Save validation results to JSON file.
        
        Parameters
        ----------
        results : Dict[str, Tuple[bool, Dict]]
            Validation results
        report_path : Path, optional
            Path to save report
        """
        if report_path is None:
            report_dir = Path("data/validation")
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"validation_report_{timestamp}.json"
        
        # Convert results to serializable format
        report_data = {
            'summary': {
                'timestamp': datetime.now().isoformat(),
                'total_assets': len(results),
                'valid_assets': sum(1 for v, _ in results.values() if v),
                'invalid_assets': sum(1 for v, _ in results.values() if not v)
            },
            'assets': {ticker: report for ticker, (_, report) in results.items()}
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")
    
    def clean_data(
        self,
        data: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw OHLCV data
        method : str
            Cleaning method ('forward_fill', 'interpolate', 'drop')
        
        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
        cleaned = data.copy()
        
        # Remove duplicates
        cleaned = cleaned[~cleaned.index.duplicated(keep='last')]
        
        # Sort chronologically
        cleaned = cleaned.sort_index()
        
        # Handle missing values
        if method == 'forward_fill':
            cleaned = cleaned.fillna(method='ffill')
            cleaned = cleaned.fillna(method='bfill')  # Handle leading NaNs
        elif method == 'interpolate':
            cleaned = cleaned.interpolate(method='linear')
        elif method == 'drop':
            cleaned = cleaned.dropna()
        
        logger.info(f"Cleaned data: {len(data)} -> {len(cleaned)} rows")
        
        return cleaned