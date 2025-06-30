"""
Enhanced Alpha Vantage Data Collector - Get Substantial AAPL Data
================================================================

Purpose: Collect MONTHS of AAPL data for proper LSTM training
- Get full historical data (20+ years if available)
- Focus on getting enough data for the model to learn patterns
- Collect key technical indicators that actually help prediction

Key changes from original:
- Use 'full' outputsize to get maximum historical data
- Add more relevant technical indicators
- Better error handling and data validation
"""

import requests
import pandas as pd
import json
import time
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDataCollector:
    """Enhanced collector focused on getting substantial training data"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
        # Create data directories
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        logger.info("✅ Enhanced collector initialized")
    
    def get_full_daily_data(self, symbol='AAPL'):
        params = {
            'function': 'TIME_SERIES_DAILY',  # Use basic daily (not adjusted)
            'symbol': symbol,
            'outputsize': 'full',
            'apikey': self.api_key
        }
        
        logger.info(f"📊 Fetching FULL historical data for {symbol}...")
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()

            if 'Error Message' in data:
                logger.error(f"❌ Error: {data['Error Message']}")
                return None
            
            if 'Note' in data:
                logger.warning(f"⚠️  API Limit: {data['Note']}")
                return None
            
            # Extract time series data
            time_series_key = 'Time Series (Daily)'
            if time_series_key not in data:
                logger.error(f"❌ No time series data found. Available keys: {list(data.keys())}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Fix column names (Alpha Vantage uses numbered columns)
            df.columns = [col.split('. ')[1].title() for col in df.columns]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            
            logger.info(f"✅ Got {len(df)} days of price data for {symbol}")
            logger.info(f"📅 Date range: {df.index.min().date()} to {df.index.max().date()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_essential_indicators(self, symbol='AAPL'):
        """
        Get only the most proven technical indicators for stock prediction
        Based on your research references
        """
        indicators = {}
        
        essential_indicators = [
            ('RSI', {'time_period': 14}),
            ('MACD', {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}),
            ('BBANDS', {'time_period': 20, 'nbdevup': 2, 'nbdevdn': 2}),
            ('SMA', {'time_period': 20}),
            ('SMA', {'time_period': 50}),  # Second SMA call
            ('EMA', {'time_period': 12})
        ]
        
        for i, (indicator, params) in enumerate(essential_indicators):
            logger.info(f"📈 Fetching {indicator} ({i+1}/{len(essential_indicators)})...")
            
            # Special handling for multiple SMA calls
            if indicator == 'SMA' and params['time_period'] == 50:
                function_name = 'SMA'
                result_key = 'SMA_50'
            else:
                function_name = indicator
                result_key = indicator
            
            api_params = {
                'function': function_name,
                'symbol': symbol,
                'interval': 'daily',
                'apikey': self.api_key
            }
            api_params.update(params)
            
            try:
                response = requests.get(self.base_url, params=api_params)
                data = response.json()
                
                # Check for errors
                if 'Error Message' in data or 'Note' in data:
                    logger.warning(f"⚠️  Skipping {indicator}: {data.get('Error Message', data.get('Note', 'Unknown error'))}")
                    time.sleep(12)
                    continue
                
                # Find the technical analysis key
                tech_key = None
                for key in data.keys():
                    if 'Technical Analysis' in key:
                        tech_key = key
                        break
                
                if tech_key:
                    df = pd.DataFrame.from_dict(data[tech_key], orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    
                    # Convert to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col])
                    
                    indicators[result_key] = df
                    logger.info(f"✅ {indicator} collected: {len(df)} rows")
                else:
                    logger.warning(f"⚠️  No technical data found for {indicator}")
                
            except Exception as e:
                logger.error(f"❌ Error with {indicator}: {str(e)}")
            
            # Rate limiting
            time.sleep(12)
        
        logger.info(f"✅ Collected {len(indicators)} technical indicators")
        return indicators
    
    def collect_aapl_dataset(self):
        """
        Collect comprehensive AAPL dataset for LSTM training
        
        Focus: Get enough data for the model to actually learn patterns
        """
        logger.info("\n🚀 Collecting comprehensive AAPL dataset...")
        
        # 1. Get full price history
        price_data = self.get_full_daily_data('AAPL')
        if price_data is None:
            logger.error("❌ Failed to get price data")
            return None
        
        # Check if we have enough data
        if len(price_data) < 500:
            logger.warning(f"⚠️  Only {len(price_data)} days of data. LSTM needs more for good training!")
        
        # 2. Get essential technical indicators
        indicators = self.get_essential_indicators('AAPL')
        
        # 3. Save everything
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save price data
        price_file = f"data/raw/AAPL_prices_full_{timestamp}.csv"
        price_data.to_csv(price_file)
        logger.info(f"💾 Saved price data: {price_file}")
        
        # Save indicators
        for name, data in indicators.items():
            indicator_file = f"data/raw/AAPL_{name}_{timestamp}.csv"
            data.to_csv(indicator_file)
            logger.info(f"💾 Saved {name}: {indicator_file}")
        
        # Create metadata
        metadata = {
            'symbol': 'AAPL',
            'collection_date': datetime.now().isoformat(),
            'data_source': 'alpha_vantage_full',
            'price_data_shape': list(price_data.shape),
            'price_file': price_file,
            'indicators': {name: f"data/raw/AAPL_{name}_{timestamp}.csv" for name in indicators.keys()},
            'date_range': {
                'start': str(price_data.index.min().date()),
                'end': str(price_data.index.max().date()),
                'total_days': len(price_data),
                'years_of_data': len(price_data) / 252
            }
        }
        
        metadata_file = f"data/raw/AAPL_metadata_full_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        # Summary
        logger.info(f"\n🎉 Data collection complete!")
        logger.info(f"📊 Total price records: {len(price_data)}")
        logger.info(f"📅 Date range: {price_data.index.min().date()} to {price_data.index.max().date()}")
        logger.info(f"⏱️  Trading years: {len(price_data)/252:.1f}")
        logger.info(f"📈 Technical indicators: {len(indicators)}")
        logger.info(f"📁 All files saved to: data/raw/")
        
        return {
            'price_data': price_data,
            'indicators': indicators,
            'metadata': metadata
        }


def main():
    """Run the enhanced data collection"""
    load_dotenv()
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if not api_key:
        print("❌ Please set ALPHA_VANTAGE_API_KEY in your .env file")
        print("Get free API key at: https://www.alphavantage.co/support/#api-key")
        return
    
    # Collect AAPL data
    collector = EnhancedDataCollector(api_key)
    result = collector.collect_aapl_dataset()
    
    if result:
        price_data = result['price_data']
        print(f"\n📊 Data Collection Summary:")
        print(f"{'='*50}")
        print(f"Symbol: AAPL")
        print(f"Records: {len(price_data):,}")
        print(f"Date Range: {price_data.index.min().date()} to {price_data.index.max().date()}")
        print(f"Years of Data: {len(price_data)/252:.1f}")
        print(f"Technical Indicators: {len(result['indicators'])}")
        print(f"\n🚀 Ready for processing and LSTM training!")
        print(f"💡 With {len(price_data)} days of data, your LSTM should have enough to learn patterns!")
    else:
        print("❌ Data collection failed")


if __name__ == "__main__":
    main()