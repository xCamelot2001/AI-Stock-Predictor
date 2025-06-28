"""
Simple Alpha Vantage Data Collector
==================================

Purpose: Collect stock data using ONLY Alpha Vantage API
- Clean, focused approach with one data source
- Collect what we actually need for LSTM training
- Save data in organized structure

Requirements:
- Alpha Vantage API key (free tier: 5 calls/minute, 500 calls/day)
- Get yours at: https://www.alphavantage.co/support/#api-key
"""

import requests
import pandas as pd
import json
import time
import os
from datetime import datetime
import logging
from dotenv import load_dotenv
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class AlphaVantageCollector:
    """
    Simple Alpha Vantage data collector focused on what we need for LSTM
    
    Why this approach:
    - Single data source (no merging complexity)
    - Rate limiting built-in (respects free tier limits)
    - Gets exactly what we need for our model
    """
    
    def __init__(self, api_key):
        """
        Initialize collector with your API key
        
        Args:
            api_key: Your Alpha Vantage API key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
        # Create data directory
        os.makedirs('data/raw', exist_ok=True)
        
        logger.info("✅ Alpha Vantage collector initialized")
    
    def get_daily_prices(self, symbol, outputsize='full'):
        """
        Get daily adjusted stock prices
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            outputsize: 'compact' (100 days) or 'full' (20+ years)
            
        Why daily adjusted:
        - Automatically handles stock splits and dividends
        - Most research uses daily data for LSTM
        - Good balance between data volume and patterns
        """
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        logger.info(f"📊 Fetching daily prices for {symbol}...")
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()

            print("API Response keys:", list(data.keys()))
            print("API Response:", data)

            # Check for errors
            if 'Error Message' in data:
                logger.error(f"❌ Error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                logger.warning(f"⚠️  API Limit: {data['Note']}")
                return None
            
            # Extract time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                logger.error(f"❌ No time series data found for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()  # Sort by date ascending
            
            # Rename columns to standard format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            logger.info(f"✅ Got {len(df)} days of price data for {symbol}")
            logger.info(f"📅 Date range: {df.index.min().date()} to {df.index.max().date()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_technical_indicator(self, symbol, indicator, **kwargs):
        """
        Get a specific technical indicator from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            indicator: Indicator name (e.g., 'RSI', 'MACD', 'BBANDS')
            **kwargs: Additional parameters for the indicator
            
        Why use Alpha Vantage indicators:
        - Pre-calculated (saves computation time)
        - Professional implementation
        - Consistent with their price data
        """
        # Map common indicators to Alpha Vantage function names
        indicator_map = {
            'RSI': 'RSI',
            'MACD': 'MACD', 
            'BBANDS': 'BBANDS',
            'SMA': 'SMA',
            'EMA': 'EMA',
            'STOCH': 'STOCH',
            'ADX': 'ADX',
            'CCI': 'CCI',
            'AROON': 'AROON',
            'MFI': 'MFI'
        }
        
        if indicator not in indicator_map:
            logger.error(f"❌ Indicator {indicator} not supported")
            return None
        
        params = {
            'function': indicator_map[indicator],
            'symbol': symbol,
            'interval': 'daily',
            'apikey': self.api_key
        }
        
        # Add specific parameters
        params.update(kwargs)
        
        logger.info(f"📈 Fetching {indicator} for {symbol}...")
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            print("API Response keys:", list(data.keys()))

            # Check for errors
            if 'Error Message' in data:
                logger.error(f"❌ Error: {data['Error Message']}")
                return None
            
            # Find the technical analysis key (varies by indicator)
            tech_key = None
            for key in data.keys():
                if 'Technical Analysis' in key:
                    tech_key = key
                    break
            
            if not tech_key:
                logger.error(f"❌ No technical analysis data found for {indicator}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[tech_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            logger.info(f"✅ Got {indicator} data: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching {indicator} for {symbol}: {str(e)}")
            return None
    
    def collect_complete_dataset(self, symbol, save=True):
        """
        Collect complete dataset for one stock
        
        What we collect:
        1. Daily price data (OHLCV)
        2. Key technical indicators (RSI, MACD, Bollinger Bands)
        3. Save everything for processing later
        
        Args:
            symbol: Stock symbol
            save: Whether to save data to files
        """
        logger.info(f"\n🚀 Collecting complete dataset for {symbol}")
        
        # 1. Get price data
        price_data = self.get_daily_prices(symbol)
        if price_data is None:
            return None
        
        # Respect rate limits (free tier: 5 calls/minute)
        time.sleep(12)  # Wait 12 seconds between calls
        
        # 2. Get key technical indicators
        indicators = {}
        
        # RSI (Relative Strength Index)
        rsi_data = self.get_technical_indicator(symbol, 'RSI', time_period=14)
        if rsi_data is not None:
            indicators['RSI'] = rsi_data
        time.sleep(12)
        
        # MACD (Moving Average Convergence Divergence)
        macd_data = self.get_technical_indicator(symbol, 'MACD', 
                                               fastperiod=12, slowperiod=26, signalperiod=9)
        if macd_data is not None:
            indicators['MACD'] = macd_data
        time.sleep(12)
        
        # Bollinger Bands
        bb_data = self.get_technical_indicator(symbol, 'BBANDS', time_period=20)
        if bb_data is not None:
            indicators['BBANDS'] = bb_data
        time.sleep(12)
        
        # Simple Moving Average
        sma_data = self.get_technical_indicator(symbol, 'SMA', time_period=20)
        if sma_data is not None:
            indicators['SMA'] = sma_data
        time.sleep(12)
        
        logger.info(f"✅ Collected {len(indicators)} technical indicators")
        
        # 3. Package everything together
        dataset = {
            'symbol': symbol,
            'price_data': price_data,
            'indicators': indicators,
            'collection_date': datetime.now().isoformat(),
            'data_source': 'alpha_vantage'
        }
        
        # 4. Save if requested
        if save:
            self.save_dataset(dataset)
        
        return dataset
    
    def save_dataset(self, dataset):
        """Save dataset to organized files"""
        symbol = dataset['symbol']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save price data
        dataset['price_data'].index.name = 'Date'
        price_file = f"data/raw/{symbol}_prices_{timestamp}.csv"
        dataset['price_data'].to_csv(price_file)
        logger.info(f"💾 Saved price data: {price_file}")
        
        # Save indicators
        for indicator_name, indicator_data in dataset['indicators'].items():
            indicator_file = f"data/raw/{symbol}_{indicator_name}_{timestamp}.csv"
            indicator_data.to_csv(indicator_file)
            logger.info(f"💾 Saved {indicator_name}: {indicator_file}")
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'collection_date': dataset['collection_date'],
            'data_source': dataset['data_source'],
            'price_data_shape': list(dataset['price_data'].shape),
            'indicators': list(dataset['indicators'].keys()),
            'date_range': {
                'start': str(dataset['price_data'].index.min().date()),
                'end': str(dataset['price_data'].index.max().date())
            }
        }
        
        metadata_file = f"data/raw/{symbol}_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved metadata: {metadata_file}")
    
    def collect_multiple_stocks(self, symbols):
        """
        Collect data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            
        Note: This will take time due to rate limiting!
        For 5 stocks: ~5 minutes (5 API calls per stock × 12 seconds each)
        """
        logger.info(f"🚀 Collecting data for {len(symbols)} stocks: {symbols}")
        logger.info("⏳ This will take several minutes due to API rate limits...")
        
        results = {}
        
        for i, symbol in enumerate(symbols):
            logger.info(f"\n📊 Processing {symbol} ({i+1}/{len(symbols)})")
            
            try:
                dataset = self.collect_complete_dataset(symbol)
                if dataset:
                    results[symbol] = {'status': 'success', 'data': dataset}
                    logger.info(f"✅ {symbol} completed successfully")
                else:
                    results[symbol] = {'status': 'failed'}
                    logger.error(f"❌ {symbol} failed")
                    
            except Exception as e:
                logger.error(f"❌ Error with {symbol}: {str(e)}")
                results[symbol] = {'status': 'error', 'message': str(e)}
        
        # Summary
        successful = len([r for r in results.values() if r['status'] == 'success'])
        logger.info(f"\n🎉 Collection complete!")
        logger.info(f"✅ Successful: {successful}/{len(symbols)}")
        logger.info(f"📁 Data saved in: data/raw/")
        
        return results


# Example usage and testing
def test_collector():
    """
    Test the collector with a small example
    
    IMPORTANT: You need to set your API key!
    """
    load_dotenv()  # This loads the .env file

    # Then get your API key:
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    # Test with one stock first
    collector = AlphaVantageCollector(api_key)
    
    # Collect data for Apple
    dataset = collector.collect_complete_dataset('AAPL')
    
    if dataset:
        print(f"\n📊 Sample of collected data:")
        print(f"Price data shape: {dataset['price_data'].shape}")
        print(f"Price data columns: {list(dataset['price_data'].columns)}")
        print(f"Indicators collected: {list(dataset['indicators'].keys())}")
        
        # Show sample price data
        print(f"\nSample price data (last 5 days):")
        print(dataset['price_data'][['Open', 'High', 'Low', 'Close', 'Volume']].tail())


if __name__ == "__main__":
    test_collector()