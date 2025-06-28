"""
Configuration file for Portfolio Optimization Application
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AppConfig:
    """Application configuration settings"""
    
    # Data fetching settings
    DEFAULT_PERIOD: str = "1y"  # Historical data period
    MAX_TICKERS: int = 50  # Maximum number of tickers to process
    REQUEST_TIMEOUT: int = 30  # Timeout for API requests in seconds
    
    # Optimization settings
    N_PORTFOLIOS: int = 100  # Number of portfolios in efficient frontier
    RISK_FREE_RATE: float = 0.02  # Risk-free rate for Sharpe ratio calculation
    MIN_WEIGHT: float = 0.0  # Minimum weight for any asset
    MAX_WEIGHT: float = 1.0  # Maximum weight for any asset
    
    # UI settings
    PAGE_TITLE: str = "Portfolio Optimizer"
    PAGE_ICON: str = "📈"
    LAYOUT: str = "wide"
    
    # Default filter values
    DEFAULT_MIN_MCAP: float = 1.0  # Billion $
    DEFAULT_MAX_MCAP: float = 1000.0  # Billion $
    DEFAULT_MIN_VOLUME: float = 1.0  # Million shares
    DEFAULT_MAX_VOLUME: float = 1000.0  # Million shares
    
    # Chart settings
    CHART_WIDTH: int = 800
    CHART_HEIGHT: int = 600
    PIE_CHART_SIZE: int = 400
    
    # File settings
    MAX_FILE_SIZE: int = 1024 * 1024  # 1MB
    ALLOWED_FILE_TYPES: list = None
    
    def __post_init__(self):
        if self.ALLOWED_FILE_TYPES is None:
            self.ALLOWED_FILE_TYPES = ['txt']

@dataclass
class MarketConfig:
    """Market-specific configuration"""
    
    # Trading days
    TRADING_DAYS_PER_YEAR: int = 252
    
    # Market cap categories (in billions)
    LARGE_CAP_MIN: float = 10.0
    MID_CAP_MIN: float = 2.0
    SMALL_CAP_MIN: float = 0.3
    
    # Volume categories (in millions)
    HIGH_VOLUME_MIN: float = 10.0
    MEDIUM_VOLUME_MIN: float = 1.0
    LOW_VOLUME_MIN: float = 0.1
    
    # Sector classifications
    SECTORS: list = None
    
    def __post_init__(self):
        if self.SECTORS is None:
            self.SECTORS = [
                "Technology",
                "Healthcare",
                "Financial Services",
                "Consumer Cyclical",
                "Communication Services",
                "Industrials",
                "Consumer Defensive",
                "Energy",
                "Utilities",
                "Real Estate",
                "Basic Materials"
            ]

class ColorScheme:
    """Color scheme for visualizations"""
    
    # Main colors
    PRIMARY = "#1f77b4"
    SECONDARY = "#ff7f0e"
    SUCCESS = "#2ca02c"
    DANGER = "#d62728"
    WARNING = "#ff9900"
    
    # Portfolio optimization colors
    EFFICIENT_FRONTIER = "#1f77b4"
    MAX_SHARPE = "#d62728"
    MIN_VOLATILITY = "#2ca02c"
    INDIVIDUAL_STOCKS = "#87ceeb"
    
    # Chart background
    BACKGROUND = "#ffffff"
    GRID = "#e6e6e6"
    
    @classmethod
    def get_portfolio_colors(cls) -> list:
        """Get color palette for portfolio composition charts"""
        return [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
        ]

class ErrorMessages:
    """Standard error messages"""
    
    NO_TICKERS = "Please provide at least one ticker symbol."
    INVALID_TICKERS = "No valid tickers found. Please check your input."
    NO_FILTERED_TICKERS = "No tickers meet the specified criteria. Please adjust your filters."
    DATA_FETCH_ERROR = "Error fetching data for ticker {}: {}"
    OPTIMIZATION_ERROR = "Could not optimize portfolio. Please try with different parameters."
    FILE_TOO_LARGE = "File size exceeds maximum limit of {}MB."
    INVALID_FILE_TYPE = "Invalid file type. Please upload a .txt file."
    API_RATE_LIMIT = "API rate limit reached. Please try again later."
    INSUFFICIENT_DATA = "Insufficient historical data for analysis."

class SuccessMessages:
    """Standard success messages"""
    
    DATA_LOADED = "Successfully loaded data for {} stocks."
    ANALYSIS_COMPLETE = "Portfolio analysis completed successfully!"
    REPORT_GENERATED = "Report generated successfully."
    EXPORT_COMPLETE = "Data exported successfully."

class InfoMessages:
    """Informational messages"""
    
    FETCHING_DATA = "Fetching stock data..."
    FILTERING_STOCKS = "Applying filter criteria..."
    OPTIMIZING_PORTFOLIO = "Optimizing portfolio..."
    GENERATING_CHARTS = "Generating visualizations..."
    CREATING_REPORT = "Creating report..."

# Global configuration instances
app_config = AppConfig()
market_config = MarketConfig()
colors = ColorScheme()
errors = ErrorMessages()
success = SuccessMessages()
info = InfoMessages()

# Environment-specific settings
def get_environment_config() -> Dict[str, Any]:
    """Get environment-specific configuration"""
    return {
        "DEBUG": os.getenv("DEBUG", "False").lower() == "true",
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "CACHE_TTL": int(os.getenv("CACHE_TTL", "3600")),  # Cache time-to-live in seconds
        "API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY"),  # Optional API key
        "MAX_WORKERS": int(os.getenv("MAX_WORKERS", "4")),  # For concurrent processing
    }

# Validation functions
def validate_ticker(ticker: str) -> bool:
    """Validate ticker symbol format"""
    if not ticker or len(ticker) > 10:
        return False
    return ticker.replace(".", "").replace("-", "").isalnum()

def validate_market_cap(mcap: float) -> bool:
    """Validate market cap value"""
    return 0 <= mcap <= 10000  # 0 to 10 trillion

def validate_volume(volume: float) -> bool:
    """Validate volume value"""
    return 0 <= volume <= 100000  # 0 to 100 billion shares

def validate_file_size(file_size: int) -> bool:
    """Validate uploaded file size"""
    return file_size <= app_config.MAX_FILE_SIZE

# Helper functions
def format_currency(value: float, suffix: str = "") -> str:
    """Format currency values with appropriate suffixes"""
    if value >= 1e9:
        return f"${value/1e9:.1f}B{suffix}"
    elif value >= 1e6:
        return f"${value/1e6:.1f}M{suffix}"
    elif value >= 1e3:
        return f"${value/1e3:.1f}K{suffix}"
    else:
        return f"${value:.2f}{suffix}"

def format_percentage(value: float) -> str:
    """Format percentage values"""
    return f"{value:.2%}"

def format_number(value: float, decimals: int = 2) -> str:
    """Format numbers with appropriate decimal places"""
    if value >= 1e6:
        return f"{value/1e6:.1f}M"
    elif value >= 1e3:
        return f"{value/1e3:.1f}K"
    else:
        return f"{value:.{decimals}f}"

# Cache configuration
CACHE_CONFIG = {
    "stock_data_ttl": 3600,  # 1 hour
    "company_info_ttl": 86400,  # 24 hours
    "optimization_ttl": 1800,  # 30 minutes
}