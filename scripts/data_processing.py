"""
Data Processing: Convert Raw Prices to LSTM-Ready Features
==========================================================

Purpose: Load AAPL price data and create features for LSTM training
- Technical indicators using TA-Lib
- Feature engineering (returns, ratios)
- Target variable (binary: up/down tomorrow)
"""

import pandas as pd
import numpy as np
import talib
import os
import glob
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataProcessor:
    """
    Simple processor to convert price data into LSTM features
    
    What we create:
    1. Technical indicators (RSI, MACD, etc.)
    2. Price features (returns, ratios)  
    3. Target variable (1=up tomorrow, 0=down)
    """
    
    def __init__(self):
        os.makedirs('data/processed', exist_ok=True)
        logger.info("✅ Data processor initialized")
    
    def load_latest_price_data(self, symbol='AAPL'):
        """Load the most recent price data file"""
        
        # Find the latest price file
        pattern = f"data/raw/{symbol}_prices_*.csv"
        files = glob.glob(pattern)
        
        if not files:
            logger.error(f"❌ No price data found for {symbol}")
            return None
        
        latest_file = max(files, key=os.path.getctime)
        logger.info(f"📊 Loading: {latest_file}")
        
        # Load data
        df = pd.read_csv(latest_file, index_col='Date', parse_dates=True)
        
        # Convert to numeric (just in case)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        logger.info(f"✅ Loaded {len(df)} days of {symbol} data")
        return df
    
    def create_technical_indicators(self, df):
        """
        Create technical indicators using TA-Lib
        
        Why these indicators:
        - RSI: Shows overbought/oversold conditions
        - MACD: Shows momentum changes
        - Bollinger Bands: Shows volatility and price channels
        - Moving averages: Show trends
        """
        logger.info("🔧 Creating technical indicators...")
        
        # Extract price arrays for TA-Lib
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        close = df['Close'].values.astype(np.float64)
        volume = df['Volume'].values.astype(np.float64)
        
        indicators = {}
        
        # 1. RSI (Relative Strength Index)
        indicators['RSI_14'] = talib.RSI(close, timeperiod=14)
        
        # 2. MACD (Moving Average Convergence Divergence)
        macd, macd_signal, macd_hist = talib.MACD(close)
        indicators['MACD'] = macd
        indicators['MACD_Signal'] = macd_signal
        indicators['MACD_Hist'] = macd_hist
        
        # 3. Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
        indicators['BB_Upper'] = bb_upper
        indicators['BB_Middle'] = bb_middle  
        indicators['BB_Lower'] = bb_lower
        indicators['BB_Width'] = (bb_upper - bb_lower) / bb_middle  # Normalized width
        indicators['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)  # Position within bands
        
        # 4. Moving Averages
        indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
        indicators['SMA_50'] = talib.SMA(close, timeperiod=50)
        indicators['EMA_12'] = talib.EMA(close, timeperiod=12)
        indicators['EMA_26'] = talib.EMA(close, timeperiod=26)
        
        # 5. Volume indicators
        indicators['OBV'] = talib.OBV(close, volume)
        
        # 6. Volatility
        indicators['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        
        # 7. Momentum
        indicators['ROC'] = talib.ROC(close, timeperiod=10)
        indicators['Stoch_K'], indicators['Stoch_D'] = talib.STOCH(high, low, close)
        
        logger.info(f"✅ Created {len(indicators)} technical indicators")
        return pd.DataFrame(indicators, index=df.index)
    
    def create_price_features(self, df):
        """
        Create price-based features
        
        Why these features:
        - Returns capture price movements better than absolute prices
        - Ratios normalize across different price levels
        - Log returns have better statistical properties
        """
        logger.info("🔧 Creating price features...")
        
        features = {}
        
        # 1. Returns (most important features)
        features['Close_Return'] = df['Close'].pct_change()  # Daily return
        features['Close_Return_Log'] = np.log(df['Close'] / df['Close'].shift(1))  # Log return
        
        # 2. Price ratios
        features['High_Low_Ratio'] = df['High'] / df['Low']
        features['Close_Open_Ratio'] = df['Close'] / df['Open']
        features['High_Close_Ratio'] = df['High'] / df['Close']
        features['Low_Close_Ratio'] = df['Low'] / df['Close']
        
        # 3. Volume features
        features['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        features['Volume_Ratio'] = df['Volume'] / features['Volume_MA']
        
        # 4. Volatility features
        features['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        features['Price_Range'] = df['High'] - df['Low']
        
        # 5. Price position features
        features['Close_vs_SMA20'] = df['Close'] / df['Close'].rolling(window=20).mean()
        features['Close_vs_SMA50'] = df['Close'] / df['Close'].rolling(window=50).mean()
        
        logger.info(f"✅ Created {len(features)} price features")
        return pd.DataFrame(features, index=df.index)
    
    def create_target_variable(self, df):
        """
        Create binary target variable: 1 = price goes up tomorrow, 0 = down
        
        Why binary classification:
        - More practical for trading decisions (buy/sell/hold)
        - Better performance than regression in most research
        - Easier to evaluate and interpret
        """
        logger.info("🎯 Creating target variable...")
        
        # Binary: 1 if tomorrow's close > today's close, 0 otherwise
        target = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Remove the last row (no future price to predict)
        target = target[:-1]
        
        # Count distribution
        up_days = target.sum()
        down_days = len(target) - up_days
        
        logger.info(f"🎯 Target distribution:")
        logger.info(f"   📈 Up days: {up_days} ({up_days/len(target)*100:.1f}%)")
        logger.info(f"   📉 Down days: {down_days} ({down_days/len(target)*100:.1f}%)")
        
        return target
    
    def process_stock_data(self, symbol='AAPL'):
        """
        Complete processing pipeline
        
        Steps:
        1. Load raw price data
        2. Create technical indicators
        3. Create price features
        4. Create target variable
        5. Combine everything
        6. Clean and save
        """
        logger.info(f"\n🚀 Processing {symbol} data...")
        
        # 1. Load data
        df = self.load_latest_price_data(symbol)
        if df is None:
            return None
        
        # 2. Create technical indicators
        tech_indicators = self.create_technical_indicators(df)
        
        # 3. Create price features
        price_features = self.create_price_features(df)
        
        # 4. Combine price data with features
        combined_df = pd.concat([df, tech_indicators, price_features], axis=1)
        
        # 5. Create target variable
        target = self.create_target_variable(combined_df)
        
        # 6. Align data with target (remove last row)
        final_df = combined_df.iloc[:-1].copy()
        final_df['Target'] = target
        
        # 7. Clean data (remove NaN values)
        initial_rows = len(final_df)
        final_df = final_df.dropna()
        dropped_rows = initial_rows - len(final_df)
        
        logger.info(f"🧹 Dropped {dropped_rows} rows with missing values")
        logger.info(f"📊 Final dataset: {len(final_df)} rows × {len(final_df.columns)} columns")
        
        # 8. Save processed data
        self.save_processed_data(final_df, symbol)
        
        return final_df
    
    def save_processed_data(self, df, symbol):
        """Save processed data with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main data
        filename = f"data/processed/{symbol}_processed_{timestamp}.csv"
        df.to_csv(filename)
        logger.info(f"💾 Saved processed data: {filename}")
        
        # Create and save metadata
        feature_cols = [col for col in df.columns if col != 'Target']
        
        metadata = {
            'symbol': symbol,
            'processing_date': datetime.now().isoformat(),
            'total_rows': len(df),
            'total_features': len(feature_cols),
            'feature_columns': feature_cols,
            'target_distribution': df['Target'].value_counts().to_dict(),
            'date_range': {
                'start': str(df.index.min().date()),
                'end': str(df.index.max().date())
            }
        }
        
        import json
        metadata_file = f"data/processed/{symbol}_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        return filename, metadata_file


def main():
    """Process AAPL data"""
    processor = StockDataProcessor()
    
    # Process AAPL
    processed_df = processor.process_stock_data('AAPL')
    
    if processed_df is not None:
        print(f"\n✅ Processing complete!")
        print(f"📊 Final dataset shape: {processed_df.shape}")
        print(f"📁 Files saved in: data/processed/")
        
        # Show sample of processed data
        print(f"\n📋 Sample features (last 5 rows):")
        sample_cols = ['Close', 'RSI_14', 'MACD', 'BB_Position', 'Close_Return', 'Target']
        available_cols = [col for col in sample_cols if col in processed_df.columns]
        print(processed_df[available_cols].tail())
        
        print(f"\n🚀 Ready for LSTM training!")
    else:
        print("❌ Processing failed")


if __name__ == "__main__":
    main()