"""
Download historical market data for all assets in the universe.

This script performs the initial data acquisition for the MERIDIAN project.
Run this once to populate the raw data directory with 10 years of historical data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.collectors import MarketDataCollector
from src.data.validators import DataValidator
from src.data.storage import DataStorage
from src.config.assets import get_all_tickers
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """
    Main execution function.
    
    Downloads historical data for all assets, validates quality,
    and saves to storage.
    """
    logger.info("=" * 80)
    logger.info("MERIDIAN - Historical Data Download")
    logger.info("=" * 80)
    
    # Initialize components
    collector = MarketDataCollector(
        start_date="2015-01-01",
        retry_attempts=3,
        retry_delay=5
    )
    
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
    
    # Download data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Downloading Historical Data")
    logger.info("=" * 80)
    
    data_dict = collector.download_multiple_assets(
        tickers=tickers,
        delay_between=1.0  # 1 second delay to be respectful to API
    )
    
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