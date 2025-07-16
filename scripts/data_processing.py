"""
Stock Data Processing Pipeline
=============================

Processes raw stock data with essential technical analysis features for LSTM training.
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
    """Processes stock data with technical analysis features for LSTM training"""
    
    def __init__(self):
        self.feature_columns = []
        os.makedirs("data/processed", exist_ok=True)
        logger.info("Data processor initialized")
    
    def load_data(self, filepath="data/raw/multi_stock_merged.csv"):
        """Load raw stock data"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_values(['Date', 'Ticker'])
        logger.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def add_features(self, df):
        """Add essential technical analysis features"""
        logger.info("Adding features...")
        
        processed_dfs = []
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            
            # Price features
            ticker_df['Returns'] = ticker_df['Close'].pct_change()
            ticker_df['Log_Returns'] = np.log(ticker_df['Close'] / ticker_df['Close'].shift(1))
            ticker_df['High_Low_Ratio'] = ticker_df['High'] / ticker_df['Low']
            ticker_df['Close_Open_Ratio'] = ticker_df['Close'] / ticker_df['Open']
            ticker_df['Daily_Range'] = (ticker_df['High'] - ticker_df['Low']) / ticker_df['Close']
            
            # Volume features
            ticker_df['Volume_MA'] = ticker_df['Volume'].rolling(20).mean()
            ticker_df['Volume_Ratio'] = ticker_df['Volume'] / ticker_df['Volume_MA']
            
            # Technical indicators
            ticker_df['RSI'] = ta.momentum.rsi(ticker_df['Close'], window=14)
            ticker_df['MACD'] = ta.trend.macd_diff(ticker_df['Close'])
            ticker_df['MACD_Signal'] = ta.trend.macd_signal(ticker_df['Close'])
            
            # Moving averages
            ticker_df['SMA_20'] = ta.trend.sma_indicator(ticker_df['Close'], window=20)
            ticker_df['SMA_50'] = ta.trend.sma_indicator(ticker_df['Close'], window=50)
            ticker_df['Close_SMA20_Ratio'] = ticker_df['Close'] / ticker_df['SMA_20']
            
            # Bollinger Bands
            ticker_df['BB_High'] = ta.volatility.bollinger_hband(ticker_df['Close'])
            ticker_df['BB_Low'] = ta.volatility.bollinger_lband(ticker_df['Close'])
            ticker_df['BB_Position'] = (ticker_df['Close'] - ticker_df['BB_Low']) / (ticker_df['BB_High'] - ticker_df['BB_Low'])
            
            # Volatility
            ticker_df['ATR'] = ta.volatility.average_true_range(ticker_df['High'], ticker_df['Low'], ticker_df['Close'])
            
            # Lag features (essential for time series)
            for col in ['Close', 'Returns', 'RSI']:
                for lag in [1, 2, 3, 5]:
                    ticker_df[f'{col}_lag_{lag}'] = ticker_df[col].shift(lag)
            
            processed_dfs.append(ticker_df)
        
        df = pd.concat(processed_dfs, ignore_index=False)
        df = df.sort_values(['Date', 'Ticker'])
        logger.info(f"Features added. New shape: {df.shape}")
        return df
    
    def add_targets(self, df, horizons=[1, 5]):
        """Add target variables for prediction"""
        logger.info("Adding target variables...")
        
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            ticker_data = df.loc[ticker_mask]
            
            for horizon in horizons:
                # Future price
                df.loc[ticker_mask, f'Close_future_{horizon}'] = ticker_data['Close'].shift(-horizon)
                # Future return
                df.loc[ticker_mask, f'Return_future_{horizon}'] = (ticker_data['Close'].shift(-horizon) / ticker_data['Close']) - 1
                # Direction (up/down)
                df.loc[ticker_mask, f'Direction_future_{horizon}'] = (ticker_data['Close'].shift(-horizon) > ticker_data['Close']).astype(int)
        
        return df
    
    def clean_and_scale(self, df):
        """Clean data and prepare features"""
        logger.info("Cleaning and scaling data...")
        
        # Define feature columns (exclude metadata and targets)
        exclude_cols = ['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume'] + \
                      [col for col in df.columns if 'future' in col.lower()]
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with too many missing values
        df = df.dropna(thresh=int(0.7 * len(df.columns)))
        
        # Forward fill by ticker, then drop remaining NaNs
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            df.loc[ticker_mask] = df.loc[ticker_mask].fillna(method='ffill')
        
        df = df.dropna()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Scale features by ticker
        scaler = StandardScaler()
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            ticker_data = df.loc[ticker_mask, self.feature_columns]
            df.loc[ticker_mask, self.feature_columns] = scaler.fit_transform(ticker_data)
        
        logger.info(f"Final dataset: {df.shape[0]} rows, {len(self.feature_columns)} features")
        return df
    
    def train_test_split(self, df, train_ratio=0.8):
        """Split data chronologically"""
        unique_dates = sorted(df.index.unique())
        split_date = unique_dates[int(len(unique_dates) * train_ratio)]
        
        train_df = df[df.index <= split_date]
        test_df = df[df.index > split_date]
        
        logger.info(f"Train: {train_df.shape[0]} rows, Test: {test_df.shape[0]} rows")
        return train_df, test_df
    
    def save_data(self, train_df, test_df):
        """Save processed datasets"""
        train_path = "data/processed/train_data.csv"
        test_path = "data/processed/test_data.csv"
        
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        
        # Save metadata
        import json
        metadata = {
            'feature_columns': self.feature_columns,
            'train_shape': train_df.shape,
            'test_shape': test_df.shape
        }
        with open("data/processed/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Data saved: {train_path}, {test_path}")
        return train_path, test_path
    
    def run_pipeline(self, input_file="data/raw/multi_stock_merged.csv"):
        """Run complete processing pipeline"""
        logger.info("Starting data processing pipeline")
        
        # Load and process
        df = self.load_data(input_file)
        df = self.add_features(df)
        df = self.add_targets(df)
        df = self.clean_and_scale(df)
        
        # Split and save
        train_df, test_df = self.train_test_split(df)
        train_path, test_path = self.save_data(train_df, test_df)
        
        logger.info("Processing complete!")
        return train_path, test_path


def main():
    """Run the data processing pipeline"""
    try:
        processor = StockDataProcessor()
        train_path, test_path = processor.run_pipeline()
        
        print(f"\nData processing complete!")
        print(f"Train data: {train_path}")
        print(f"Test data: {test_path}")
        print(f"Features: {len(processor.feature_columns)}")
        print(f"Ready for LSTM training!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()