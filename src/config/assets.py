"""
Asset universe definition for MERIDIAN portfolio optimization.

Defines the complete set of tradeable assets across multiple asset classes,
their characteristics, and data source configurations.
"""

from enum import Enum
from typing import Dict, List
from dataclasses import dataclass


class AssetClass(Enum):
    """Asset class categorization."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    ALTERNATIVE = "alternative"


@dataclass
class AssetConfig:
    """
    Configuration for a single asset.
    
    Attributes
    ----------
    ticker : str
        Yahoo Finance ticker symbol
    name : str
        Full asset name
    asset_class : AssetClass
        Asset class categorization
    sector : str
        Sector or sub-category
    data_source : str
        Primary data source
    """
    ticker: str
    name: str
    asset_class: AssetClass
    sector: str
    data_source: str = "yfinance"


# Complete asset universe
ASSET_UNIVERSE: Dict[str, AssetConfig] = {
    # Equities
    "SPY": AssetConfig(
        ticker="SPY",
        name="SPDR S&P 500 ETF",
        asset_class=AssetClass.EQUITY,
        sector="US Large Cap"
    ),
    "QQQ": AssetConfig(
        ticker="QQQ",
        name="Invesco QQQ Trust",
        asset_class=AssetClass.EQUITY,
        sector="US Technology"
    ),
    "IWM": AssetConfig(
        ticker="IWM",
        name="iShares Russell 2000 ETF",
        asset_class=AssetClass.EQUITY,
        sector="US Small Cap"
    ),
    "VTI": AssetConfig(
        ticker="VTI",
        name="Vanguard Total Stock Market ETF",
        asset_class=AssetClass.EQUITY,
        sector="US Total Market"
    ),
    "EFA": AssetConfig(
        ticker="EFA",
        name="iShares MSCI EAFE ETF",
        asset_class=AssetClass.EQUITY,
        sector="International Developed"
    ),
    "EEM": AssetConfig(
        ticker="EEM",
        name="iShares MSCI Emerging Markets ETF",
        asset_class=AssetClass.EQUITY,
        sector="Emerging Markets"
    ),
    "VNQ": AssetConfig(
        ticker="VNQ",
        name="Vanguard Real Estate ETF",
        asset_class=AssetClass.EQUITY,
        sector="Real Estate"
    ),
    "XLF": AssetConfig(
        ticker="XLF",
        name="Financial Select Sector SPDR",
        asset_class=AssetClass.EQUITY,
        sector="Financials"
    ),
    "XLE": AssetConfig(
        ticker="XLE",
        name="Energy Select Sector SPDR",
        asset_class=AssetClass.EQUITY,
        sector="Energy"
    ),
    "XLK": AssetConfig(
        ticker="XLK",
        name="Technology Select Sector SPDR",
        asset_class=AssetClass.EQUITY,
        sector="Technology"
    ),
    
    # Fixed Income
    "TLT": AssetConfig(
        ticker="TLT",
        name="iShares 20+ Year Treasury Bond ETF",
        asset_class=AssetClass.FIXED_INCOME,
        sector="Long-Term Treasury"
    ),
    "IEF": AssetConfig(
        ticker="IEF",
        name="iShares 7-10 Year Treasury Bond ETF",
        asset_class=AssetClass.FIXED_INCOME,
        sector="Intermediate Treasury"
    ),
    "SHY": AssetConfig(
        ticker="SHY",
        name="iShares 1-3 Year Treasury Bond ETF",
        asset_class=AssetClass.FIXED_INCOME,
        sector="Short-Term Treasury"
    ),
    "LQD": AssetConfig(
        ticker="LQD",
        name="iShares iBoxx Investment Grade Corporate Bond ETF",
        asset_class=AssetClass.FIXED_INCOME,
        sector="Investment Grade Corporate"
    ),
    "HYG": AssetConfig(
        ticker="HYG",
        name="iShares iBoxx High Yield Corporate Bond ETF",
        asset_class=AssetClass.FIXED_INCOME,
        sector="High Yield Corporate"
    ),
    
    # Commodities
    "GLD": AssetConfig(
        ticker="GLD",
        name="SPDR Gold Shares",
        asset_class=AssetClass.COMMODITY,
        sector="Precious Metals"
    ),
    "SLV": AssetConfig(
        ticker="SLV",
        name="iShares Silver Trust",
        asset_class=AssetClass.COMMODITY,
        sector="Precious Metals"
    ),
    "DBC": AssetConfig(
        ticker="DBC",
        name="Invesco DB Commodity Index Tracking Fund",
        asset_class=AssetClass.COMMODITY,
        sector="Broad Commodities"
    ),
    "USO": AssetConfig(
        ticker="USO",
        name="United States Oil Fund",
        asset_class=AssetClass.COMMODITY,
        sector="Energy"
    ),
    "UNG": AssetConfig(
        ticker="UNG",
        name="United States Natural Gas Fund",
        asset_class=AssetClass.COMMODITY,
        sector="Energy"
    ),
    
    # Alternatives
    "BTC-USD": AssetConfig(
        ticker="BTC-USD",
        name="Bitcoin",
        asset_class=AssetClass.ALTERNATIVE,
        sector="Cryptocurrency"
    ),
    "ETH-USD": AssetConfig(
        ticker="ETH-USD",
        name="Ethereum",
        asset_class=AssetClass.ALTERNATIVE,
        sector="Cryptocurrency"
    ),
    "UUP": AssetConfig(
        ticker="UUP",
        name="Invesco DB US Dollar Index Bullish Fund",
        asset_class=AssetClass.ALTERNATIVE,
        sector="Currency"
    ),
    "VXX": AssetConfig(
        ticker="VXX",
        name="iPath Series B S&P 500 VIX Short-Term Futures ETN",
        asset_class=AssetClass.ALTERNATIVE,
        sector="Volatility"
    ),
    "TIP": AssetConfig(
        ticker="TIP",
        name="iShares TIPS Bond ETF",
        asset_class=AssetClass.ALTERNATIVE,
        sector="Inflation Protected"
    ),
}


def get_tickers_by_class(asset_class: AssetClass) -> List[str]:
    """
    Get list of tickers for a specific asset class.
    
    Parameters
    ----------
    asset_class : AssetClass
        Asset class to filter by
    
    Returns
    -------
    List[str]
        List of ticker symbols
    """
    return [
        ticker for ticker, config in ASSET_UNIVERSE.items()
        if config.asset_class == asset_class
    ]


def get_all_tickers() -> List[str]:
    """
    Get complete list of all ticker symbols.
    
    Returns
    -------
    List[str]
        All ticker symbols in asset universe
    """
    return list(ASSET_UNIVERSE.keys())


# Asset class groupings for easy access
EQUITY_TICKERS = get_tickers_by_class(AssetClass.EQUITY)
FIXED_INCOME_TICKERS = get_tickers_by_class(AssetClass.FIXED_INCOME)
COMMODITY_TICKERS = get_tickers_by_class(AssetClass.COMMODITY)
ALTERNATIVE_TICKERS = get_tickers_by_class(AssetClass.ALTERNATIVE)