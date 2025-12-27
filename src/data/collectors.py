"""
Market data collection from multiple sources.

Primary source: Yahoo Finance via yfinance
Fallback sources: Alpha Vantage, others as needed
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from pathlib import Path
import time

from src.utils.logger import get_logger
from src.config.assets import ASSET_UNIVERSE, get_all_tickers

logger = get_logger(__name__)


class MarketDataCollector:
    """
    Collects historical and real-time market data for portfolio assets.
    
    Handles API rate limiting, retries on failure, and data validation.
    """
    
    def __init__(
        self,
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        retry_attempts: int = 3,
        retry_delay: int = 5
    ):
        """
        Initialize market data collector.
        
        Parameters
        ----------
        start_date : str
            Start date for historical data (YYYY-MM-DD)
        end_date : str, optional
            End date for historical data (defaults to today)
        retry_attempts : int
            Number of retry attempts for failed downloads
        retry_delay : int
            Delay in seconds between retry attempts
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        logger.info(
            f"Initialized MarketDataCollector: {self.start_date} to {self.end_date}"
        )
    
    def download_single_asset(
        self,
        ticker: str,
        validate: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Download historical data for a single asset.
        
        Parameters
        ----------
        ticker : str
            Asset ticker symbol
        validate : bool
            Whether to perform basic validation
        
        Returns
        -------
        pd.DataFrame or None
            OHLCV data with datetime index, or None if download fails
        """
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Downloading {ticker} (attempt {attempt + 1}/{self.retry_attempts})")
                
                # Download data
                data = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True  # Adjust for splits and dividends
                )
                
                if data.empty:
                    logger.warning(f"No data returned for {ticker}")
                    return None
                
                # Clean column names (remove multi-index if present)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Standardize column names
                data.columns = [col.lower() for col in data.columns]
                
                # Basic validation
                if validate:
                    if len(data) < 100:
                        logger.warning(f"Insufficient data for {ticker}: {len(data)} rows")
                        return None
                    
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in data.columns]
                    if missing_cols:
                        logger.warning(f"Missing columns for {ticker}: {missing_cols}")
                        return None
                
                logger.info(f"Successfully downloaded {ticker}: {len(data)} rows")
                return data
                
            except Exception as e:
                logger.error(f"Error downloading {ticker} (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.retry_attempts - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to download {ticker} after {self.retry_attempts} attempts")
                    return None
        
        return None
    
    def download_multiple_assets(
        self,
        tickers: Optional[List[str]] = None,
        delay_between: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for multiple assets.
        
        Parameters
        ----------
        tickers : List[str], optional
            List of ticker symbols (defaults to all assets in universe)
        delay_between : float
            Delay in seconds between downloads to avoid rate limiting
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping tickers to their OHLCV dataframes
        """
        if tickers is None:
            tickers = get_all_tickers()
        
        logger.info(f"Starting batch download for {len(tickers)} assets")
        
        data_dict = {}
        successful = 0
        failed = []
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Processing {i}/{len(tickers)}: {ticker}")
            
            data = self.download_single_asset(ticker)
            
            if data is not None:
                data_dict[ticker] = data
                successful += 1
            else:
                failed.append(ticker)
            
            # Delay to avoid rate limiting
            if i < len(tickers):
                time.sleep(delay_between)
        
        logger.info(
            f"Batch download complete: {successful} successful, {len(failed)} failed"
        )
        
        if failed:
            logger.warning(f"Failed tickers: {', '.join(failed)}")
        
        return data_dict
    
    def update_single_asset(
        self,
        ticker: str,
        existing_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Update existing asset data with latest values.
        
        Parameters
        ----------
        ticker : str
            Asset ticker symbol
        existing_data : pd.DataFrame
            Existing OHLCV data
        
        Returns
        -------
        pd.DataFrame
            Updated OHLCV data
        """
        last_date = existing_data.index.max()
        today = pd.Timestamp.now().normalize()
        
        # No update needed if data is current
        if last_date >= today - timedelta(days=1):
            logger.info(f"{ticker} is up to date")
            return existing_data
        
        logger.info(f"Updating {ticker} from {last_date.date()} to {today.date()}")
        
        # Download new data
        new_data = yf.download(
            ticker,
            start=last_date + timedelta(days=1),
            end=today,
            progress=False,
            auto_adjust=True
        )
        
        if new_data.empty:
            logger.info(f"No new data for {ticker}")
            return existing_data
        
        # Clean and standardize
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.get_level_values(0)
        new_data.columns = [col.lower() for col in new_data.columns]
        
        # Combine with existing data
        updated_data = pd.concat([existing_data, new_data])
        updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
        updated_data = updated_data.sort_index()
        
        logger.info(f"Updated {ticker}: added {len(new_data)} new rows")
        
        return updated_data
    
    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Get the most recent closing price for an asset.
        
        Parameters
        ----------
        ticker : str
            Asset ticker symbol
        
        Returns
        -------
        float or None
            Latest closing price
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # Try different price fields
            price = info.get('regularMarketPrice') or \
                    info.get('previousClose') or \
                    info.get('currentPrice')
            
            if price:
                logger.info(f"Latest price for {ticker}: ${price:.2f}")
                return float(price)
            else:
                logger.warning(f"Could not retrieve latest price for {ticker}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest price for {ticker}: {str(e)}")
            return None