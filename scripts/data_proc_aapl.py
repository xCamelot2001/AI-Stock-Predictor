"""
Enhanced Data Processor - Process Full AAPL Dataset for LSTM
============================================================

Purpose: Process the substantial AAPL dataset into LSTM-ready features
- Load full historical price data
- Create comprehensive technical indicators using TA-Lib
- Engineer effective features based on research best practices
- Create balanced dataset for training

Key improvements:
- Handles much larger datasets efficiently
- Creates features proven to work in your references
- Better feature engineering and selection
"""

import pandas as pd
import numpy as np
import talib
import os
import glob
import json
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedStockProcessor:
    """
    Enhanced processor for creating LSTM-ready features from comprehensive stock data
    """
    
    def __init__(self):
        os.makedirs('data/processed', exist_ok=True)
        logger.info("✅ Enhanced data processor initialized")
    
    def load_full_price_data(self, symbol='AAPL'):
        """Load the most recent full price dataset"""
        
        # Find the latest full price file
        pattern = f"data/raw/{symbol}_prices_full_*.csv"
        files = glob.glob(pattern)
        
        if not files:
            logger.error(f"❌ No full price data found for {symbol}")
            logger.info("💡 Run the enhanced data collector first!")
            return None
        
        latest_file = max(files, key=os.path.getctime)
        logger.info(f"📊 Loading: {latest_file}")
        
        # Load data
        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        
        # Convert to numeric and handle any data issues
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with missing price data
        df = df.dropna()
        
        logger.info(f"✅ Loaded {len(df)} days of {symbol} data")
        logger.info(f"📅 Date range: {df.index.min().date()} to {df.index.max().date()}")
        logger.info(f"⏱️  Years of data: {len(df)/252:.1f}")
        
        return df
    
    def create_comprehensive_technical_indicators(self, df):
        """
        Create comprehensive technical indicators based on research best practices
        
        Focus on indicators proven effective in your references:
        - RSI, MACD, Bollinger Bands (core momentum/volatility)
        - Moving averages (trend)
        - Volume indicators (market interest)
        """
        logger.info("🔧 Creating comprehensive technical indicators...")
        
        # Convert to numpy arrays for TA-Lib (required format)
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        close = df['Close'].values.astype(np.float64)
        volume = df['Volume'].values.astype(np.float64)
        open_price = df['Open'].values.astype(np.float64)
        
        indicators = {}
        
        # === MOMENTUM INDICATORS ===
        logger.info("📈 Creating momentum indicators...")
        
        # RSI - Multiple periods for different timeframes
        indicators['RSI_14'] = talib.RSI(close, timeperiod=14)
        indicators['RSI_21'] = talib.RSI(close, timeperiod=21)
        
        # MACD - Full suite
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        indicators['MACD'] = macd
        indicators['MACD_Signal'] = macd_signal
        indicators['MACD_Histogram'] = macd_hist
        
        # Stochastic Oscillator
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        indicators['Stoch_K'] = slowk
        indicators['Stoch_D'] = slowd
        
        # Rate of Change
        indicators['ROC_10'] = talib.ROC(close, timeperiod=10)
        indicators['ROC_20'] = talib.ROC(close, timeperiod=20)
        
        # Commodity Channel Index
        indicators['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        
        # === TREND INDICATORS ===
        logger.info("📊 Creating trend indicators...")
        
        # Moving Averages
        indicators['SMA_10'] = talib.SMA(close, timeperiod=10)
        indicators['SMA_20'] = talib.SMA(close, timeperiod=20)
        indicators['SMA_50'] = talib.SMA(close, timeperiod=50)
        indicators['SMA_200'] = talib.SMA(close, timeperiod=200)
        
        # Exponential Moving Averages
        indicators['EMA_12'] = talib.EMA(close, timeperiod=12)
        indicators['EMA_26'] = talib.EMA(close, timeperiod=26)
        indicators['EMA_50'] = talib.EMA(close, timeperiod=50)
        
        # === VOLATILITY INDICATORS ===
        logger.info("📉 Creating volatility indicators...")
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        indicators['BB_Upper'] = bb_upper
        indicators['BB_Middle'] = bb_middle
        indicators['BB_Lower'] = bb_lower
        
        # Average True Range
        indicators['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        
        # === VOLUME INDICATORS ===
        logger.info("📊 Creating volume indicators...")
        
        # On Balance Volume
        indicators['OBV'] = talib.OBV(close, volume)
        
        # Volume SMA
        indicators['Volume_SMA'] = talib.SMA(volume, timeperiod=20)
        
        # Money Flow Index
        indicators['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        logger.info(f"✅ Created {len(indicators)} technical indicators")
        return pd.DataFrame(indicators, index=df.index)
    
    def create_advanced_features(self, df, tech_indicators):
        """
        Create advanced engineered features proven effective for LSTM
        
        Based on your research references, these features help LSTM learn better:
        - Price ratios and normalized features
        - Trend strength indicators
        - Volatility measures
        - Relative position indicators
        """
        logger.info("🔧 Creating advanced engineered features...")
        
        features = {}
        
        # === PRICE-BASED FEATURES ===
        logger.info("💰 Creating price features...")
        
        # Returns (most important for LSTM)
        features['Close_Return'] = df['Close'].pct_change()
        features['High_Return'] = df['High'].pct_change()
        features['Low_Return'] = df['Low'].pct_change()
        features['Open_Return'] = df['Open'].pct_change()
        
        # Log returns (better statistical properties)
        features['Close_LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price ratios
        features['High_Low_Ratio'] = df['High'] / df['Low']
        features['Close_Open_Ratio'] = df['Close'] / df['Open']
        features['High_Close_Ratio'] = df['High'] / df['Close']
        features['Low_Close_Ratio'] = df['Low'] / df['Close']
        
        # === TECHNICAL INDICATOR RATIOS ===
        logger.info("📈 Creating indicator ratios...")
        
        # Price vs Moving Averages (trend strength)
        features['Price_vs_SMA20'] = df['Close'] / tech_indicators['SMA_20']
        features['Price_vs_SMA50'] = df['Close'] / tech_indicators['SMA_50']
        features['Price_vs_EMA12'] = df['Close'] / tech_indicators['EMA_12']
        
        # Bollinger Band position (where price sits within bands)
        bb_width = tech_indicators['BB_Upper'] - tech_indicators['BB_Lower']
        features['BB_Position'] = (df['Close'] - tech_indicators['BB_Lower']) / bb_width
        features['BB_Width_Norm'] = bb_width / df['Close']  # Normalized width
        
        # RSI momentum
        features['RSI_14_Norm'] = (tech_indicators['RSI_14'] - 50) / 50  # Normalized around 50
        
        # === VOLUME FEATURES ===
        logger.info("📊 Creating volume features...")
        
        # Volume ratios
        features['Volume_Ratio'] = df['Volume'] / tech_indicators['Volume_SMA']
        features['Volume_Return'] = df['Volume'].pct_change()
        
        # Price-Volume relationship
        features['Price_Volume_Trend'] = features['Close_Return'] * features['Volume_Ratio']
        
        # === VOLATILITY FEATURES ===
        logger.info("📉 Creating volatility features...")
        
        # Price range features
        features['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        features['Open_Close_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Rolling volatility
        features['Return_Volatility_5'] = features['Close_Return'].rolling(window=5).std()
        features['Return_Volatility_10'] = features['Close_Return'].rolling(window=10).std()
        
        # === MOMENTUM FEATURES ===
        logger.info("🚀 Creating momentum features...")
        
        # Multi-period returns
        features['Return_2d'] = df['Close'].pct_change(2)
        features['Return_5d'] = df['Close'].pct_change(5)
        features['Return_10d'] = df['Close'].pct_change(10)
        
        # Trend consistency
        features['SMA_Trend'] = (tech_indicators['SMA_20'] > tech_indicators['SMA_50']).astype(int)
        features['EMA_Trend'] = (tech_indicators['EMA_12'] > tech_indicators['EMA_26']).astype(int)
        
        logger.info(f"✅ Created {len(features)} advanced features")
        return pd.DataFrame(features, index=df.index)
    
    def create_target_variable(self, df, prediction_horizon=1):
        """
        Create target variable for LSTM prediction
        
        Args:
            df: Price dataframe
            prediction_horizon: Days ahead to predict (default=1 for next day)
        """
        logger.info(f"🎯 Creating target variable (predict {prediction_horizon} day(s) ahead)...")
        
        # Binary classification: 1 if price goes up in prediction_horizon days, 0 otherwise
        future_close = df['Close'].shift(-prediction_horizon)
        target = (future_close > df['Close']).astype(int)
        
        # Remove rows where we can't predict (at the end)
        target = target[:-prediction_horizon]
        
        # Calculate distribution
        up_days = target.sum()
        total_days = len(target)
        down_days = total_days - up_days
        
        logger.info(f"🎯 Target distribution:")
        logger.info(f"   📈 Up days: {up_days:,} ({up_days/total_days*100:.1f}%)")
        logger.info(f"   📉 Down days: {down_days:,} ({down_days/total_days*100:.1f}%)")
        
        return target
    
    def combine_and_clean_features(self, price_df, tech_indicators, advanced_features, target):
        """
        Combine all features and clean the dataset
        """
        logger.info("🧹 Combining and cleaning all features...")
        
        # Combine all features
        combined_df = pd.concat([
            price_df,
            tech_indicators, 
            advanced_features
        ], axis=1)
        
        # Align with target (remove last row(s))
        combined_df = combined_df.iloc[:len(target)]
        combined_df['Target'] = target
        
        # Remove rows with any missing values
        initial_rows = len(combined_df)
        combined_df = combined_df.dropna()
        final_rows = len(combined_df)
        dropped_rows = initial_rows - final_rows
        
        logger.info(f"🧹 Data cleaning results:")
        logger.info(f"   Initial rows: {initial_rows:,}")
        logger.info(f"   Final rows: {final_rows:,}")
        logger.info(f"   Dropped rows: {dropped_rows:,}")
        logger.info(f"   Features: {len(combined_df.columns)-1}")  # -1 for target
        
        # Check for sufficient data
        if final_rows < 1000:
            logger.warning(f"⚠️  Only {final_rows} samples. LSTM typically needs >1000 for good performance")
        else:
            logger.info(f"✅ {final_rows:,} samples - excellent for LSTM training!")
        
        return combined_df
    
    def process_full_dataset(self, symbol='AAPL'):
        """
        Complete processing pipeline for full dataset
        """
        logger.info(f"\n🚀 Processing full {symbol} dataset...")
        
        # 1. Load price data
        df = self.load_full_price_data(symbol)
        if df is None:
            return None
        
        # 2. Create technical indicators
        tech_indicators = self.create_comprehensive_technical_indicators(df)
        
        # 3. Create advanced features
        advanced_features = self.create_advanced_features(df, tech_indicators)
        
        # 4. Create target variable
        target = self.create_target_variable(df)
        
        # 5. Combine and clean
        final_df = self.combine_and_clean_features(df, tech_indicators, advanced_features, target)
        
        # 6. Save processed data
        self.save_processed_data(final_df, symbol)
        
        return final_df
    
    def save_processed_data(self, df, symbol):
        """Save processed data with comprehensive metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main dataset
        filename = f"data/processed/{symbol}_processed_full_{timestamp}.csv"
        df.to_csv(filename)
        logger.info(f"💾 Saved processed data: {filename}")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != 'Target']
        
        # Create comprehensive metadata
        metadata = {
            'symbol': symbol,
            'processing_date': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(df),
                'total_features': len(feature_cols),
                'date_range': {
                    'start': str(df.index.min().date()),
                    'end': str(df.index.max().date()),
                    'total_days': len(df),
                    'years_of_data': len(df) / 252
                }
            },
            'target_distribution': {
                'up_days': int(df['Target'].sum()),
                'down_days': int(len(df) - df['Target'].sum()),
                'up_percentage': float(df['Target'].mean() * 100)
            },
            'feature_categories': {
                'price_features': [col for col in feature_cols if any(x in col for x in ['Open', 'High', 'Low', 'Close', 'Volume'])],
                'technical_indicators': [col for col in feature_cols if any(x in col for x in ['RSI', 'MACD', 'SMA', 'EMA', 'BB_', 'ATR', 'OBV'])],
                'engineered_features': [col for col in feature_cols if any(x in col for x in ['Return', 'Ratio', 'Pct', 'Trend', 'Position'])]
            },
            'lstm_readiness': {
                'samples_available': len(df),
                'recommended_lookback': min(60, len(df) // 20),  # 60 days or 5% of data
                'training_samples_estimate': int(len(df) * 0.7),
                'validation_samples_estimate': int(len(df) * 0.15),
                'test_samples_estimate': int(len(df) * 0.15)
            }
        }
        
        # Save metadata
        metadata_file = f"data/processed/{symbol}_metadata_full_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        # Summary
        logger.info(f"\n🎉 Processing complete!")
        logger.info(f"📊 Final dataset: {len(df):,} samples × {len(feature_cols)} features")
        logger.info(f"📈 Target distribution: {df['Target'].mean()*100:.1f}% up days")
        logger.info(f"🚀 Ready for LSTM training with proven architecture!")
        
        return filename, metadata_file


def main():
    """Process the full AAPL dataset"""
    processor = EnhancedStockProcessor()
    
    # Process AAPL
    processed_df = processor.process_full_dataset('AAPL')
    
    if processed_df is not None:
        print(f"\n✅ Processing complete!")
        print(f"📊 Dataset shape: {processed_df.shape}")
        print(f"📁 Files saved in: data/processed/")
        
        # Show feature summary
        feature_cols = [col for col in processed_df.columns if col != 'Target']
        print(f"\n📋 Feature Summary:")
        print(f"   Total features: {len(feature_cols)}")
        print(f"   Price features: {len([col for col in feature_cols if 'Close' in col or 'Open' in col])}")
        print(f"   Technical indicators: {len([col for col in feature_cols if any(x in col for x in ['RSI', 'MACD', 'SMA'])])}")
        print(f"   Engineered features: {len([col for col in feature_cols if 'Return' in col or 'Ratio' in col])}")
        
        # Show sample data
        print(f"\n📊 Sample data (last 3 days):")
        sample_cols = ['Close', 'RSI_14', 'MACD', 'BB_Position', 'Close_Return', 'Target']
        available_cols = [col for col in sample_cols if col in processed_df.columns]
        print(processed_df[available_cols].tail(3))
        
        print(f"\n🚀 Ready for LSTM training with {len(processed_df):,} samples!")
    else:
        print("❌ Processing failed - make sure you've run the data collector first")


if __name__ == "__main__":
    main()