"""
Download historical market data using batch requests to avoid rate limits.

This version downloads all assets in a single API call, which is more efficient
and less likely to trigger rate limiting.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd
from src.data.validators import DataValidator
from src.data.storage import DataStorage
from src.config.assets import get_all_tickers
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_batch(tickers, start_date="2015-01-01", end_date=None):
    """
    Download multiple tickers in a single batch request.
    
    Parameters
    ----------
    tickers : list
        List of ticker symbols
    start_date : str
        Start date for historical data
    end_date : str
        End date (defaults to today)
    
    Returns
    -------
    dict
        Dictionary mapping tickers to DataFrames
    """
    logger.info(f"Downloading {len(tickers)} assets in batch mode")
    logger.info(f"Date range: {start_date} to {end_date or 'today'}")
    
    try:
        # Download all tickers at once
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by='ticker',
            auto_adjust=True,
            progress=True
        )
        
        if data.empty:
            logger.error("No data returned from batch download")
            return {}
        
        # Process each ticker
        data_dict = {}
        
        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    # Single ticker case (no multi-index)
                    ticker_data = data.copy()
                else:
                    # Multiple tickers (multi-index)
                    ticker_data = data[ticker].copy()
                
                # Skip if empty
                if ticker_data.empty or len(ticker_data) < 100:
                    logger.warning(f"Insufficient data for {ticker}")
                    continue
                
                # Standardize column names
                ticker_data.columns = [col.lower() for col in ticker_data.columns]
                
                # Remove any NaN rows at start/end
                ticker_data = ticker_data.dropna(how='all')
                
                data_dict[ticker] = ticker_data
                logger.info(f"✓ {ticker}: {len(ticker_data)} rows")
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue
        
        logger.info(f"Batch download complete: {len(data_dict)}/{len(tickers)} successful")
        return data_dict
        
    except Exception as e:
        logger.error(f"Batch download failed: {str(e)}")
        return {}


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("MERIDIAN - Historical Data Download (Batch Mode)")
    logger.info("=" * 80)
    
    # Initialize components
    validator = DataValidator(
        max_missing_pct=5.0,
        max_price_change_pct=50.0,
        min_data_points=252
    )
    
    storage = DataStorage()
    
    # Get asset universe
    tickers = get_all_tickers()
    logger.info(f"Asset universe: {len(tickers)} assets")
    logger.info(f"Tickers: {', '.join(tickers)}")
    
    # Download in batch
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Downloading Historical Data (Batch Mode)")
    logger.info("=" * 80)
    
    data_dict = download_batch(tickers, start_date="2015-01-01")
    
    if not data_dict:
        logger.error("Batch download failed completely. Exiting.")
        return
    
    logger.info(f"\nDownload complete: {len(data_dict)}/{len(tickers)} successful")
    
    # Validate data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Validating Data Quality")
    logger.info("=" * 80)
    
    validation_results = validator.validate_multiple_assets(
        data_dict,
        save_report=True
    )
    
    valid_assets = {
        ticker: data for ticker, data in data_dict.items()
        if validation_results[ticker][0]
    }
    
    invalid_assets = [
        ticker for ticker, (is_valid, _) in validation_results.items()
        if not is_valid
    ]
    
    logger.info(f"\nValidation complete: {len(valid_assets)} valid, {len(invalid_assets)} invalid")
    
    if invalid_assets:
        logger.warning(f"Invalid assets: {', '.join(invalid_assets)}")
    
    # Save raw data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Saving Raw Data")
    logger.info("=" * 80)
    
    storage.save_batch(data_dict, data_type='raw')
    
    # Clean and save processed data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Cleaning and Saving Processed Data")
    logger.info("=" * 80)
    
    processed_dict = {}
    for ticker, data in valid_assets.items():
        cleaned = validator.clean_data(data, method='forward_fill')
        processed_dict[ticker] = cleaned
    
    storage.save_batch(processed_dict, data_type='processed')
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total assets attempted: {len(tickers)}")
    logger.info(f"Successfully downloaded: {len(data_dict)}")
    logger.info(f"Passed validation: {len(valid_assets)}")
    logger.info(f"Saved to storage: {len(processed_dict)}")
    
    if len(processed_dict) >= len(tickers) * 0.8:
        logger.info("\n✓ Data acquisition successful!")
    else:
        logger.warning("\n⚠ Some assets failed. Check logs for details.")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()