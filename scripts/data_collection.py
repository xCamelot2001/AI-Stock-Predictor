"""
Stock Data Collection Pipeline
=============================

Collects historical stock data for multiple stocks and merges into single dataset.
Uses Alpha Vantage API with proper error handling and data validation.

Usage: python data_collection.py
"""

import os
import time
import pandas as pd
import logging
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataCollector:
    """
    Collects and merges stock data for multiple tickers
    """
    
    def __init__(self, api_key=None):
        """Initialize with Alpha Vantage API key"""
        load_dotenv()
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found. Set in .env file or pass as parameter.")
        
        self.ts = TimeSeries(key=self.api_key, output_format="pandas")
        
        # Create directories
        os.makedirs("data/raw", exist_ok=True)
        
        logger.info("Stock data collector initialized")
    
    def fetch_single_stock(self, symbol, outputsize="full", start_date="2014-01-01"):
        """
        Fetch data for a single stock with error handling
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            outputsize: 'full' or 'compact'
            start_date: Filter data from this date
        
        Returns:
            pandas.DataFrame: Stock data with OHLCV columns
        """
        logger.info(f"Fetching {symbol} data...")
        
        try:
            # Fetch from Alpha Vantage
            data, metadata = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
            
            # Rename columns to standard format
            data = data.rename(columns={
                "1. open": "Open",
                "2. high": "High", 
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume"
            })
            
            # Select only OHLCV columns
            data = data[["Open", "High", "Low", "Close", "Volume"]]
            data.index.name = "Date"
            
            # Sort by date (ascending)
            data = data.sort_index()
            
            # Filter by start date
            data = data[data.index >= start_date]
            
            # Data validation
            if len(data) == 0:
                raise ValueError(f"No data found for {symbol} after {start_date}")
            
            if data.isnull().any().any():
                logger.warning(f"Found missing values in {symbol} data")
            
            logger.info(f"{symbol}: {len(data)} rows fetched ({data.index.min().date()} to {data.index.max().date()})")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {str(e)}")
            raise
    
    def save_individual_stock(self, symbol, data):
        """Save individual stock data to CSV"""
        filepath = f"data/raw/{symbol}.csv"
        data.to_csv(filepath)
        logger.info(f"Saved {symbol} to {filepath}")
        return filepath
    
    def load_stock_from_file(self, symbol):
        """Load stock data from existing CSV file"""
        filepath = f"data/raw/{symbol}.csv"
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No data file found for {symbol}")
        
        try:
            data = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
            data["Ticker"] = symbol
            logger.info(f"Loaded {symbol}: {len(data)} rows")
            return data
        except Exception as e:
            logger.error(f"Error loading {symbol}: {str(e)}")
            raise
    
    def collect_all_stocks(self, tickers, start_date="2014-01-01", save_individual=True):
        """
        Collect data for all stocks with API rate limiting
        
        Args:
            tickers: List of stock symbols
            start_date: Start date for data collection
            save_individual: Save individual CSV files
        
        Returns:
            dict: {symbol: dataframe} for each stock
        """
        stock_data = {}
        
        logger.info(f"Starting collection for {len(tickers)} stocks")
        
        for i, ticker in enumerate(tickers):
            try:
                # Fetch data
                data = self.fetch_single_stock(ticker, start_date=start_date)
                stock_data[ticker] = data
                
                # Save individual file
                if save_individual:
                    self.save_individual_stock(ticker, data)
                
                # Rate limiting (Alpha Vantage: 5 calls/minute for free tier)
                if i < len(tickers) - 1:  # Don't sleep after last ticker
                    logger.info("Waiting 15 seconds (API rate limit)...")
                    time.sleep(15)
                    
            except Exception as e:
                logger.error(f"Failed to collect {ticker}: {str(e)}")
                # Continue with other stocks
                continue
        
        logger.info(f"Collection complete: {len(stock_data)}/{len(tickers)} stocks successful")
        return stock_data
    
    def merge_stocks(self, tickers, use_existing_files=True):
        """
        Merge multiple stocks into single DataFrame
        
        Args:
            tickers: List of stock symbols
            use_existing_files: Load from existing CSV files vs collect new data
        
        Returns:
            pandas.DataFrame: Merged stock data with Ticker column
        """
        logger.info(f"Merging {len(tickers)} stocks...")
        
        stock_dfs = []
        
        for ticker in tickers:
            try:
                if use_existing_files:
                    df = self.load_stock_from_file(ticker)
                else:
                    # Would need to collect first
                    raise NotImplementedError("Set use_existing_files=True or run collect_all_stocks first")
                
                stock_dfs.append(df)
                
            except Exception as e:
                logger.error(f"Failed to load {ticker}: {str(e)}")
                continue
        
        if not stock_dfs:
            raise ValueError("No stock data loaded successfully")
        
        # Merge all dataframes
        merged_df = pd.concat(stock_dfs, ignore_index=False)
        merged_df = merged_df.sort_index()  # Sort by date
        
        logger.info(f"Initial merged shape: {merged_df.shape}")
        
        # Find common trading dates across all stocks
        date_sets = [set(df.index.date) for df in stock_dfs]
        common_dates = sorted(set.intersection(*date_sets))
        
        if len(common_dates) == 0:
            logger.warning("No common trading dates found across all stocks")
            return merged_df
        
        # Filter to common dates only
        merged_df = merged_df.loc[merged_df.index.isin(pd.to_datetime(common_dates))]
        merged_df = merged_df.sort_index()
        
        logger.info(f"Final merged shape: {merged_df.shape} (common dates: {len(common_dates)})")
        logger.info(f"Date range: {merged_df.index.min().date()} to {merged_df.index.max().date()}")
        
        return merged_df
    
    def save_merged_data(self, merged_df, filename="multi_stock_merged.csv"):
        """Save merged dataset"""
        filepath = f"data/raw/{filename}"
        
        # Reset index to make Date a column for easier loading later
        save_df = merged_df.reset_index()
        save_df.to_csv(filepath, index=False)
        
        logger.info(f"Saved merged data: {filepath}")
        logger.info(f"Final dataset: {len(save_df)} rows × {len(save_df.columns)} columns")
        
        return filepath
    
    def run_full_pipeline(self, tickers, start_date="2014-01-01", collect_new=False):
        """
        Run complete data collection and merging pipeline
        
        Args:
            tickers: List of stock symbols
            start_date: Start date for data
            collect_new: If True, fetch new data; if False, use existing files
        
        Returns:
            str: Path to merged CSV file
        """
        logger.info("Starting full data collection pipeline")
        
        # Step 1: Collect individual stock data (if needed)
        if collect_new:
            self.collect_all_stocks(tickers, start_date=start_date)
        
        # Step 2: Merge stocks
        merged_df = self.merge_stocks(tickers, use_existing_files=True)
        
        # Step 3: Save merged dataset
        filepath = self.save_merged_data(merged_df)
        
        logger.info("Pipeline complete!")
        return filepath


def main():
    """Run the data collection pipeline"""
    
    # Configuration
    TICKERS = [
        "AAPL",   # Apple Inc.
        "MSFT",   # Microsoft Corporation  
        "NVDA",   # NVIDIA Corporation
        "AMZN",   # Amazon.com Inc.
        "GOOGL",  # Alphabet Inc. (Google)
    ]
    
    START_DATE = "2014-01-01"
    
    try:
        # Initialize collector
        collector = StockDataCollector()
        
        # Run full pipeline
        output_file = collector.run_full_pipeline(
            tickers=TICKERS,
            start_date=START_DATE,
            collect_new=False  # Set to True to fetch new data
        )
        # Output result
        print(f"\n✅ Success! Merged data saved to: {output_file}")
        print(f"📊 Ready for feature engineering and model training")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()