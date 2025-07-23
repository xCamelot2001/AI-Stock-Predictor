"""
Optimized Stock Data Processing Pipeline
=======================================

Processes raw stock data with optimal features for LSTM training.
This version cleans the data but DOES NOT scale it.
"""

import os
import pandas as pd
import numpy as np
import logging
import ta
# StandardScaler is no longer needed in this script
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataProcessor:
    """Processes stock data with optimal features for LSTM training"""
    
    def __init__(self):
        self.feature_columns = []
        os.makedirs("data/processed", exist_ok=True)
        logger.info("Optimized data processor initialized")
    
    def load_data(self, filepath="data/raw/multi_stock_merged.csv"):
        # This method remains unchanged
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_values(['Date', 'Ticker'])
        logger.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def add_optimal_features(self, df):
        """Add optimal features for stock prediction"""
        logger.info("Adding optimal features...")
        
        processed_dfs = []
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            
            # --- All feature creation logic remains the same ---
            # === CORE PRICE ACTION FEATURES (4) ===
            ticker_df['Log_Return'] = np.log(ticker_df['Close'] / ticker_df['Close'].shift(1))
            ticker_df['High_Low_Ratio'] = ticker_df['High'] / ticker_df['Low']
            ticker_df['Close_Open_Ratio'] = ticker_df['Close'] / ticker_df['Open']
            ticker_df['Gap'] = (ticker_df['Open'] - ticker_df['Close'].shift(1)) / ticker_df['Close'].shift(1)
            # ... (rest of your feature engineering)...
            ticker_df['RSI_14'] = ta.momentum.rsi(ticker_df['Close'], window=14)
            ticker_df['MACD'] = ta.trend.macd_diff(ticker_df['Close'])
            bb_high = ta.volatility.bollinger_hband(ticker_df['Close'])
            bb_low = ta.volatility.bollinger_lband(ticker_df['Close'])
            ticker_df['BB_Position'] = (ticker_df['Close'] - bb_low) / (bb_high - bb_low)
            ticker_df['ATR_14'] = ta.volatility.average_true_range(ticker_df['High'], ticker_df['Low'], ticker_df['Close'], window=14)
            ticker_df['ADX'] = ta.trend.adx(ticker_df['High'], ticker_df['Low'], ticker_df['Close'])
            volume_ma_20 = ticker_df['Volume'].rolling(20).mean()
            ticker_df['Volume_Ratio_20'] = ticker_df['Volume'] / volume_ma_20
            ticker_df['Volume_Price_Correlation'] = ticker_df['Close'].rolling(20).corr(ticker_df['Volume'])
            sma_20 = ta.trend.sma_indicator(ticker_df['Close'], window=20)
            ticker_df['Close_vs_SMA20'] = ticker_df['Close'] / sma_20
            sma_50 = ta.trend.sma_indicator(ticker_df['Close'], window=50)
            ticker_df['Close_vs_SMA50'] = ticker_df['Close'] / sma_50
            high_20d = ticker_df['High'].rolling(20).max()
            ticker_df['Close_vs_High20d'] = ticker_df['Close'] / high_20d
            ticker_df['Volatility_20d'] = ticker_df['Log_Return'].rolling(20).std()
            ticker_df['Return_3d'] = ticker_df['Close'].pct_change(3)
            ticker_df['Return_5d'] = ticker_df['Close'].pct_change(5)
            ticker_df['ROC_10'] = ta.momentum.roc(ticker_df['Close'], window=10)
            ticker_df['Momentum_10d'] = ticker_df['Close'] / ticker_df['Close'].shift(10)
            ticker_df['Volatility_5d'] = ticker_df['Log_Return'].rolling(5).std()
            vol20 = ticker_df['Log_Return'].rolling(20).std()
            ticker_df['Volatility_Z20'] = (vol20 - vol20.rolling(20).mean()) / vol20.rolling(20).std()
            ticker_df['High_PrevClose'] = ticker_df['High'] - ticker_df['Close'].shift(1)
            ticker_df['Up_5d'] = (ticker_df['Close'] > ticker_df['Close'].shift(1)).rolling(5).sum()
            ticker_df['OBV'] = ta.volume.on_balance_volume(ticker_df['Close'], ticker_df['Volume'])
            ticker_df['Volume_Change_1d'] = ticker_df['Volume'].pct_change(1)
            ticker_df['Close_Z20'] = (ticker_df['Close'] - ticker_df['Close'].rolling(20).mean()) / ticker_df['Close'].rolling(20).std()
            rolling_max_10 = ticker_df['Close'].rolling(10, min_periods=1).max()
            ticker_df['Drawdown_10d'] = (ticker_df['Close'] - rolling_max_10) / rolling_max_10
            ticker_df['RSI_14_Lag1'] = ticker_df['RSI_14'].shift(1)
            ticker_df['MACD_Lag1'] = ticker_df['MACD'].shift(1)
            
            processed_dfs.append(ticker_df)
        
        df = pd.concat(processed_dfs, ignore_index=False)
        df = df.sort_values(['Date', 'Ticker'])
        logger.info(f"Optimal features added. New shape: {df.shape}")
        return df
    
    def add_targets(self, df, horizon=5): # tweak horizon as needed
        # """Add target variable: Will price be higher in N days?"""
        logger.info(f"Adding target variable: Will price be higher in {horizon} days?")
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            ticker_data = df.loc[ticker_mask]
            future_close = ticker_data['Close'].shift(-horizon)
            df.loc[ticker_mask, 'Target'] = (future_close > ticker_data['Close']).astype(int)
        df = df.dropna(subset=['Target'])
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
            logger.warning("Data quality issues found:")
            for issue in issues:
                logger.warning(f"   - {issue}")
        else:
            logger.info("Data quality validation passed")
    
    def _check_data_sufficiency(self, df):
        """Check if we have sufficient data per stock"""
        logger.info("Data sufficiency check:")
        min_samples = 1000
        
        for ticker in df['Ticker'].unique():
            count = len(df[df['Ticker'] == ticker])
            if count < min_samples:
                logger.warning(f"{ticker}: Only {count:,} samples (< {min_samples:,})")
            else:
                logger.info(f"{ticker}: {count:,} samples")

    def clean_data(self, df):
        """Cleans data and prepares features, but does NOT scale them."""
        # --- UPDATED LOG MESSAGE ---
        logger.info("Cleaning data...")
        
        self._validate_stock_data(df)
        
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Log_Return', 'High_Low_Ratio', 'Close_Open_Ratio', 'Gap', 'RSI_14', 'MACD',
            'BB_Position', 'ATR_14', 'ADX', 'Volume_Ratio_20', 'Volume_Price_Correlation',
            'Close_vs_SMA20', 'Close_vs_SMA50', 'Close_vs_High20d', 'Volatility_20d',
            'Return_3d', 'Return_5d', 'ROC_10', 'Momentum_10d', 'Volatility_5d',
            'Volatility_Z20', 'High_PrevClose', 'Up_5d', 'OBV', 'Volume_Change_1d',
            'Close_Z20', 'Drawdown_10d', 'RSI_14_Lag1', 'MACD_Lag1'
        ]
                
        df = df.replace([np.inf, -np.inf], np.nan)
        
        missing_pct = df.isnull().sum() / len(df)
        high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
        if high_missing_cols:
            logger.warning(f"Removing columns with >50% missing: {high_missing_cols}")
            df = df.drop(columns=high_missing_cols)
        
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            df.loc[ticker_mask] = df.loc[ticker_mask].fillna(method='ffill')
        
        df = df.dropna()
        
        self._check_data_sufficiency(df)
        
        logger.info(f"Final cleaned (unscaled) dataset: {df.shape[0]} rows, {len(self.feature_columns)} features")
        logger.info(f"Class balance: {df['Target'].value_counts().to_dict()}")
        
        return df
    
    def save_data(self, df):
        """Save processed dataset"""
        output_path = "data/processed/stock_data_optimized.csv"
        df.to_csv(output_path)
        
        # Save metadata
        import json
        metadata = {
            'feature_columns': self.feature_columns,
            'num_features': len(self.feature_columns),
            'dataset_shape': df.shape,
            'class_distribution': df['Target'].value_counts().to_dict(),
            'stocks': sorted(df['Ticker'].unique().tolist()),
            'date_range': {
                'start': str(df.index.min().date()),
                'end': str(df.index.max().date())
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
        df = self.clean_data(df)

        # Save processed data
        output_path = self.save_data(df)
        
        logger.info("Optimized processing complete!")
        return output_path


def main():
    """Run the optimized data processing pipeline"""
    try:
        processor = StockDataProcessor()
        output_path = processor.run_pipeline()
        
        print(f"\nData processing complete!")
        print(f"Processed data: {output_path}")
        print(f"Features: {len(processor.feature_columns)} (reduced from 98+)")
        print(f"Ready for LSTM training!")
        
        # Print feature list
        print(f"\nSelected features:")
        for i, feature in enumerate(processor.feature_columns, 1):
            print(f"  {i:2d}. {feature}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()