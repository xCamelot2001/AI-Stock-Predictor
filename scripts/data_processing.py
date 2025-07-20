"""
Optimized Stock Data Processing Pipeline
=======================================

Processes raw stock data with 15 optimal features for LSTM training.
Eliminates redundancy while maintaining signal diversity.
"""

import os
import pandas as pd
import numpy as np
import logging
import ta
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataProcessor:
    """Processes stock data with 15 optimal features for LSTM training"""
    
    def __init__(self):
        self.feature_columns = []
        os.makedirs("data/processed", exist_ok=True)
        logger.info("Optimized data processor initialized")
    
    def load_data(self, filepath="data/raw/multi_stock_merged.csv"):
        """Load raw stock data"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_values(['Date', 'Ticker'])
        logger.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def add_optimal_features(self, df):
        """Add only the 15 most effective features for LSTM"""
        logger.info("Adding 15 optimal features...")
        
        processed_dfs = []
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            
            # === CORE PRICE ACTION FEATURES (4) ===
            # 1. Log Return - Primary momentum signal
            ticker_df['Log_Return'] = np.log(ticker_df['Close'] / ticker_df['Close'].shift(1))
            
            # 2. High_Low_Ratio - Intraday volatility
            ticker_df['High_Low_Ratio'] = ticker_df['High'] / ticker_df['Low']
            
            # 3. Close_Open_Ratio - Daily sentiment
            ticker_df['Close_Open_Ratio'] = ticker_df['Close'] / ticker_df['Open']
            
            # 4. Gap - Overnight gap signal
            ticker_df['Gap'] = (ticker_df['Open'] - ticker_df['Close'].shift(1)) / ticker_df['Close'].shift(1)
            
            # === TECHNICAL INDICATORS (5) ===
            # 5. RSI_14 - Momentum oscillator
            ticker_df['RSI_14'] = ta.momentum.rsi(ticker_df['Close'], window=14)
            
            # 6. MACD - Trend following
            ticker_df['MACD'] = ta.trend.macd_diff(ticker_df['Close'])
            
            # 7. BB_Position - Bollinger Bands position
            bb_high = ta.volatility.bollinger_hband(ticker_df['Close'])
            bb_low = ta.volatility.bollinger_lband(ticker_df['Close'])
            ticker_df['BB_Position'] = (ticker_df['Close'] - bb_low) / (bb_high - bb_low)
            
            # 8. ATR_14 - True volatility measure
            ticker_df['ATR_14'] = ta.volatility.average_true_range(
                ticker_df['High'], ticker_df['Low'], ticker_df['Close'], window=14
            )
            
            # 9. ADX - Trend strength
            ticker_df['ADX'] = ta.trend.adx(ticker_df['High'], ticker_df['Low'], ticker_df['Close'])
            
            # === VOLUME FEATURES (2) ===
            # 10. Volume_Ratio_20 - Relative volume activity
            volume_ma_20 = ticker_df['Volume'].rolling(20).mean()
            ticker_df['Volume_Ratio_20'] = ticker_df['Volume'] / volume_ma_20
            
            # 11. Volume_Price_Correlation - Price-volume relationship
            ticker_df['Volume_Price_Correlation'] = ticker_df['Close'].rolling(20).corr(ticker_df['Volume'])
            
            # === POSITION/TREND FEATURES (3) ===
            # 12. Close_vs_SMA20 - Short-term trend position
            sma_20 = ta.trend.sma_indicator(ticker_df['Close'], window=20)
            ticker_df['Close_vs_SMA20'] = ticker_df['Close'] / sma_20
            
            # 13. Close_vs_SMA50 - Medium-term trend position
            sma_50 = ta.trend.sma_indicator(ticker_df['Close'], window=50)
            ticker_df['Close_vs_SMA50'] = ticker_df['Close'] / sma_50
            
            # 14. Close_vs_High20d - Relative strength vs recent highs
            high_20d = ticker_df['High'].rolling(20).max()
            ticker_df['Close_vs_High20d'] = ticker_df['Close'] / high_20d
            
            # === VOLATILITY FEATURE (1) ===
            # 15. Volatility_20d - Realized volatility
            ticker_df['Volatility_20d'] = ticker_df['Log_Return'].rolling(20).std()
            
            processed_dfs.append(ticker_df)
        
        df = pd.concat(processed_dfs, ignore_index=False)
        df = df.sort_values(['Date', 'Ticker'])
        logger.info(f"15 optimal features added. New shape: {df.shape}")
        return df
    
    def add_targets(self, df, horizon=1):
        """Add binary classification target"""
        logger.info("Adding target variables...")
        
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            ticker_data = df.loc[ticker_mask]
            
            # Binary classification: will price go up?
            future_close = ticker_data['Close'].shift(-horizon)
            df.loc[ticker_mask, 'Target'] = (future_close > ticker_data['Close']).astype(int)
        
        return df
    
    def _validate_stock_data(self, df):
        """Validate stock data quality"""
        issues = []
        
        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if (df[col] <= 0).any():
                issues.append(f"Found non-positive values in {col}")
        
        # Check OHLC logic
        if ((df['High'] < df['Low']) | (df['High'] < df['Open']) | 
            (df['High'] < df['Close']) | (df['Low'] > df['Open']) | 
            (df['Low'] > df['Close'])).any():
            issues.append("OHLC logic violations found")
        
        # Check for negative volume
        if (df['Volume'] < 0).any():
            issues.append("Found negative volume values")
        
        if issues:
            logger.warning("⚠️ Data quality issues found:")
            for issue in issues:
                logger.warning(f"   - {issue}")
        else:
            logger.info("✅ Data quality validation passed")
    
    def _check_data_sufficiency(self, df):
        """Check if we have sufficient data per stock"""
        logger.info("📊 Data sufficiency check:")
        min_samples = 1000
        
        for ticker in df['Ticker'].unique():
            count = len(df[df['Ticker'] == ticker])
            if count < min_samples:
                logger.warning(f"⚠️ {ticker}: Only {count:,} samples (< {min_samples:,})")
            else:
                logger.info(f"✅ {ticker}: {count:,} samples")

    def clean_and_scale(self, df):
        """Clean data and prepare features with validation"""
        logger.info("Cleaning and scaling data...")
        
        # Validate data quality first
        self._validate_stock_data(df)
        
        # Define the 15 optimal feature columns
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Log_Return', 'High_Low_Ratio', 'Close_Open_Ratio', 'Gap',
            'RSI_14', 'MACD', 'BB_Position', 'ATR_14', 'ADX',
            'Volume_Ratio_20', 'Volume_Price_Correlation',
            'Close_vs_SMA20', 'Close_vs_SMA50', 'Close_vs_High20d',
            'Volatility_20d'
        ]
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove columns with too many missing values (>50%)
        missing_pct = df.isnull().sum() / len(df)
        high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
        if high_missing_cols:
            logger.warning(f"⚠️ Removing columns with >50% missing: {high_missing_cols}")
            df = df.drop(columns=high_missing_cols)
        
        # Forward fill by ticker, then drop remaining NaNs
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            df.loc[ticker_mask] = df.loc[ticker_mask].fillna(method='ffill')
        
        df = df.dropna()
        
        # Check data sufficiency
        self._check_data_sufficiency(df)
        
        # Scale features by ticker
        scaler = StandardScaler()
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            ticker_data = df.loc[ticker_mask, self.feature_columns]
            df.loc[ticker_mask, self.feature_columns] = scaler.fit_transform(ticker_data)
        
        logger.info(f"Final dataset: {df.shape[0]} rows, {len(self.feature_columns)} features")
        logger.info(f"Class balance: {df['Target'].value_counts().to_dict()}")
        
        return df
    
    def save_data(self, df):
        """Save processed dataset"""
        output_path = "data/processed/stock_data_optimized.csv"
        df.to_csv(output_path)  # Keep index=True to save Date
        
        # Save metadata
        import json
        metadata = {
            'feature_columns': self.feature_columns,
            'num_features': len(self.feature_columns),
            'dataset_shape': df.shape,
            'class_distribution': df['Target'].value_counts().to_dict(),
            'stocks': sorted(df['Ticker'].unique().tolist()),
            'date_range': {
                'start': str(df.index.min().date()),  # Use df.index instead of df['Date']
                'end': str(df.index.max().date())    # Use df.index instead of df['Date']
            }
        }
        with open("data/processed/metadata_optimized.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Data saved: {output_path}")
        return output_path
    
    def run_pipeline(self, input_file="data/raw/multi_stock_merged.csv"):
        """Run complete data processing pipeline"""
        logger.info("Starting data processing pipeline")
        
        # Load and process
        df = self.load_data(input_file)
        df = self.add_optimal_features(df)
        df = self.add_targets(df)
        df = self.clean_and_scale(df)
        
        # Save processed data
        output_path = self.save_data(df)
        
        logger.info("Optimized processing complete!")
        return output_path


def main():
    """Run the optimized data processing pipeline"""
    try:
        processor = StockDataProcessor()
        output_path = processor.run_pipeline()
        
        print(f"\n✅ Data processing complete!")
        print(f"📁 Processed data: {output_path}")
        print(f"🎯 Features: {len(processor.feature_columns)} (reduced from 98+)")
        print(f"🚀 Ready for LSTM training!")
        
        # Print feature list
        print(f"\n📊 Selected features:")
        for i, feature in enumerate(processor.feature_columns, 1):
            print(f"  {i:2d}. {feature}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()