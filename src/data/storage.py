"""
Data storage and retrieval for market data.

Manages persistent storage of OHLCV data using Parquet format for efficiency.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataStorage:
    """
    Manages storage and retrieval of market data.
    
    Uses Parquet format for efficient compression and fast I/O.
    """
    
    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed"
    ):
        """
        Initialize data storage manager.
        
        Parameters
        ----------
        raw_data_dir : str
            Directory for raw OHLCV data
        processed_data_dir : str
            Directory for processed/cleaned data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        
        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DataStorage: raw={raw_data_dir}, processed={processed_data_dir}")
    
    def save_raw_data(
        self,
        ticker: str,
        data: pd.DataFrame,
        overwrite: bool = False
    ) -> Path:
        """
        Save raw OHLCV data for a single asset.
        
        Parameters
        ----------
        ticker : str
            Asset ticker symbol
        data : pd.DataFrame
            OHLCV data to save
        overwrite : bool
            Whether to overwrite existing file
        
        Returns
        -------
        Path
            Path to saved file
        """
        file_path = self.raw_data_dir / f"{ticker}.parquet"
        
        if file_path.exists() and not overwrite:
            logger.warning(f"{file_path} already exists. Set overwrite=True to replace.")
            return file_path
        
        data.to_parquet(file_path, compression='snappy')
        logger.info(f"Saved {ticker} to {file_path} ({len(data)} rows)")
        
        return file_path
    
    def load_raw_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load raw OHLCV data for a single asset.
        
        Parameters
        ----------
        ticker : str
            Asset ticker symbol
        
        Returns
        -------
        pd.DataFrame or None
            OHLCV data, or None if file doesn't exist
        """
        file_path = self.raw_data_dir / f"{ticker}.parquet"
        
        if not file_path.exists():
            logger.warning(f"No data file found for {ticker} at {file_path}")
            return None
        
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded {ticker} from {file_path} ({len(data)} rows)")
        
        return data
    
    def save_processed_data(
        self,
        ticker: str,
        data: pd.DataFrame,
        overwrite: bool = True
    ) -> Path:
        """
        Save processed/cleaned data for a single asset.
        
        Parameters
        ----------
        ticker : str
            Asset ticker symbol
        data : pd.DataFrame
            Processed data to save
        overwrite : bool
            Whether to overwrite existing file
        
        Returns
        -------
        Path
            Path to saved file
        """
        file_path = self.processed_data_dir / f"{ticker}.parquet"
        
        if file_path.exists() and not overwrite:
            logger.warning(f"{file_path} already exists. Set overwrite=True to replace.")
            return file_path
        
        data.to_parquet(file_path, compression='snappy')
        logger.info(f"Saved processed {ticker} to {file_path} ({len(data)} rows)")
        
        return file_path
    
    def load_processed_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Load processed data for a single asset.
        
        Parameters
        ----------
        ticker : str
            Asset ticker symbol
        
        Returns
        -------
        pd.DataFrame or None
            Processed data, or None if file doesn't exist
        """
        file_path = self.processed_data_dir / f"{ticker}.parquet"
        
        if not file_path.exists():
            logger.warning(f"No processed data found for {ticker}")
            return None
        
        data = pd.read_parquet(file_path)
        logger.info(f"Loaded processed {ticker} ({len(data)} rows)")
        
        return data
    
    def save_batch(
        self,
        data_dict: Dict[str, pd.DataFrame],
        data_type: str = 'raw'
    ) -> List[Path]:
        """
        Save multiple assets in batch.
        
        Parameters
        ----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary mapping tickers to dataframes
        data_type : str
            Type of data ('raw' or 'processed')
        
        Returns
        -------
        List[Path]
            List of saved file paths
        """
        logger.info(f"Saving batch of {len(data_dict)} assets ({data_type})")
        
        saved_paths = []
        
        for ticker, data in data_dict.items():
            if data_type == 'raw':
                path = self.save_raw_data(ticker, data, overwrite=True)
            else:
                path = self.save_processed_data(ticker, data, overwrite=True)
            
            saved_paths.append(path)
        
        logger.info(f"Batch save complete: {len(saved_paths)} files")
        
        return saved_paths
    
    def load_batch(
        self,
        tickers: List[str],
        data_type: str = 'processed'
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple assets in batch.
        
        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols to load
        data_type : str
            Type of data ('raw' or 'processed')
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping tickers to dataframes
        """
        logger.info(f"Loading batch of {len(tickers)} assets ({data_type})")
        
        data_dict = {}
        
        for ticker in tickers:
            if data_type == 'raw':
                data = self.load_raw_data(ticker)
            else:
                data = self.load_processed_data(ticker)
            
            if data is not None:
                data_dict[ticker] = data
        
        logger.info(f"Batch load complete: {len(data_dict)}/{len(tickers)} successful")
        
        return data_dict
    
    def list_available_assets(self, data_type: str = 'raw') -> List[str]:
        """
        List all available assets in storage.
        
        Parameters
        ----------
        data_type : str
            Type of data ('raw' or 'processed')
        
        Returns
        -------
        List[str]
            List of ticker symbols
        """
        directory = self.raw_data_dir if data_type == 'raw' else self.processed_data_dir
        
        parquet_files = list(directory.glob("*.parquet"))
        tickers = [f.stem for f in parquet_files]
        
        logger.info(f"Found {len(tickers)} {data_type} assets in storage")
        
        return sorted(tickers)
    
    def get_data_info(self, ticker: str, data_type: str = 'raw') -> Optional[Dict]:
        """
        Get metadata about stored data.
        
        Parameters
        ----------
        ticker : str
            Asset ticker symbol
        data_type : str
            Type of data ('raw' or 'processed')
        
        Returns
        -------
        Dict or None
            Metadata dictionary
        """
        if data_type == 'raw':
            data = self.load_raw_data(ticker)
        else:
            data = self.load_processed_data(ticker)
        
        if data is None:
            return None
        
        info = {
            'ticker': ticker,
            'data_type': data_type,
            'rows': len(data),
            'columns': list(data.columns),
            'start_date': data.index.min().isoformat(),
            'end_date': data.index.max().isoformat(),
            'missing_values': data.isnull().sum().to_dict(),
            'file_size_mb': self._get_file_size(ticker, data_type)
        }
        
        return info
    
    def _get_file_size(self, ticker: str, data_type: str) -> float:
        """Get file size in MB."""
        directory = self.raw_data_dir if data_type == 'raw' else self.processed_data_dir
        file_path = directory / f"{ticker}.parquet"
        
        if file_path.exists():
            size_bytes = file_path.stat().st_size
            return round(size_bytes / (1024 * 1024), 2)
        
        return 0.0
    
    def delete_asset(self, ticker: str, data_type: str = 'raw') -> bool:
        """
        Delete stored data for an asset.
        
        Parameters
        ----------
        ticker : str
            Asset ticker symbol
        data_type : str
            Type of data ('raw' or 'processed')
        
        Returns
        -------
        bool
            True if deleted successfully
        """
        directory = self.raw_data_dir if data_type == 'raw' else self.processed_data_dir
        file_path = directory / f"{ticker}.parquet"
        
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted {data_type} data for {ticker}")
            return True
        
        logger.warning(f"No {data_type} data found for {ticker}")
        return False