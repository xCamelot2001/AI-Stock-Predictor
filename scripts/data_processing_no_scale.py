"""
Optimized Stock Data Processing Pipeline
=======================================

Processes raw stock data with research-validated features for LSTM training.
"""

import os
import pandas as pd
import numpy as np
import logging
import ta
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
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.sort_values(['Date', 'Ticker'])
        logger.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def add_optimal_features(self, df):
        """Add research-validated features for stock prediction"""
        logger.info("Adding optimal features...")
        
        processed_dfs = []
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            
            # 1. CORE RETURN FEATURES
            ticker_df['Log_Return'] = np.log(ticker_df['Close'] / ticker_df['Close'].shift(1))
            ticker_df['Intraday_Range'] = (ticker_df['High'] - ticker_df['Low']) / ticker_df['Close']
            ticker_df['Overnight_Return'] = (ticker_df['Open'] - ticker_df['Close'].shift(1)) / ticker_df['Close'].shift(1)
            ticker_df['High_Low_Ratio'] = ticker_df['High'] / ticker_df['Low']
            ticker_df['Close_Open_Ratio'] = ticker_df['Close'] / ticker_df['Open']
            
            # 2. MICROSTRUCTURE FEATURES
            ticker_df['Bid_Ask_Proxy'] = 2 * np.sqrt(np.abs(ticker_df['Log_Return']))
            ticker_df['Amihud_Illiquidity'] = np.abs(ticker_df['Log_Return']) / ticker_df['Volume']
            ticker_df['Kyle_Lambda'] = ticker_df['Log_Return'].abs() / np.sqrt(ticker_df['Volume'])
            
            # 3. VOLUME FEATURES
            ticker_df['VWAP'] = (ticker_df['Close'] * ticker_df['Volume']).rolling(20).sum() / ticker_df['Volume'].rolling(20).sum()
            ticker_df['Price_to_VWAP'] = ticker_df['Close'] / ticker_df['VWAP']
            ticker_df['Volume_Imbalance'] = ticker_df['Volume'].diff() / ticker_df['Volume'].rolling(20).mean()
            ticker_df['Volume_Ratio_20'] = ticker_df['Volume'] / ticker_df['Volume'].rolling(20).mean()
            
            # 4. MOMENTUM FEATURES
            for period in [5, 10, 20]:
                ticker_df[f'Return_{period}d'] = ticker_df['Close'].pct_change(period)
                ret_mean = ticker_df[f'Return_{period}d'].rolling(period).mean()
                ret_std = ticker_df[f'Return_{period}d'].rolling(period).std()
                ticker_df[f'Sharpe_{period}d'] = ret_mean / (ret_std + 1e-6)
            
            # 5. VOLATILITY FEATURES
            ticker_df['Realized_Vol_5d'] = ticker_df['Log_Return'].rolling(5).std() * np.sqrt(252)
            ticker_df['Realized_Vol_20d'] = ticker_df['Log_Return'].rolling(20).std() * np.sqrt(252)
            ticker_df['Vol_of_Vol'] = ticker_df['Realized_Vol_5d'].rolling(20).std()
            ticker_df['Skewness_20d'] = ticker_df['Log_Return'].rolling(20).skew()
            
            # 6. TECHNICAL INDICATORS
            rsi = ta.momentum.rsi(ticker_df['Close'], window=14)
            ticker_df['RSI_14'] = rsi / 100  # Normalize to 0-1
            ticker_df['RSI_Oversold'] = (rsi < 30).astype(int)
            ticker_df['RSI_Overbought'] = (rsi > 70).astype(int)
            
            # Moving average features
            ma20 = ticker_df['Close'].rolling(20).mean()
            ma50 = ticker_df['Close'].rolling(50).mean()
            ticker_df['Price_to_MA20'] = ticker_df['Close'] / ma20
            ticker_df['Price_to_MA50'] = ticker_df['Close'] / ma50
            ticker_df['MA20_Slope'] = ma20.pct_change(5)
            
            # 7. TIME FEATURES (for market anomalies)
            ticker_df['Day_of_Week'] = ticker_df.index.dayofweek
            ticker_df['Is_Month_End'] = (ticker_df.index.day > 25).astype(int)
            
            processed_dfs.append(ticker_df)
        
        df = pd.concat(processed_dfs, ignore_index=False)
        df = df.sort_values(['Date', 'Ticker'])
        logger.info(f"Features added. New shape: {df.shape}")
        return df
    
    def add_targets(self, df, horizon=20, threshold=0.02):
        """Add target: significant price move in N days"""
        logger.info(f"Adding target: >{threshold*100}% move in {horizon} days")
        
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            ticker_data = df.loc[ticker_mask]
            
            future_return = (ticker_data['Close'].shift(-horizon) - ticker_data['Close']) / ticker_data['Close']
            df.loc[ticker_mask, 'Target'] = (future_return > threshold).astype(int)
            
        df = df.dropna(subset=['Target'])
        return df
    
    def _validate_stock_data(self, df):
        """Validate stock data quality"""
        issues = []
        
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if (df[col] <= 0).any():
                issues.append(f"Found non-positive values in {col}")
        
        if (df['Volume'] < 0).any():
            issues.append("Found negative volume values")
        
        if issues:
            logger.warning("Data quality issues found:")
            for issue in issues:
                logger.warning(f"   - {issue}")
        else:
            logger.info("Data quality validation passed")
    
    def clean_data(self, df):
        """Clean data and select final features"""
        logger.info("Cleaning data...")
        
        self._validate_stock_data(df)
        
        # Define features to use (NO raw OHLCV)
        self.feature_columns = [
            # Core returns
            'Log_Return', 'Intraday_Range', 'Overnight_Return',
            'High_Low_Ratio', 'Close_Open_Ratio',
            
            # Microstructure
            'Bid_Ask_Proxy', 'Amihud_Illiquidity', 'Kyle_Lambda',
            
            # Volume
            'Price_to_VWAP', 'Volume_Imbalance', 'Volume_Ratio_20',
            
            # Momentum
            'Return_5d', 'Return_10d', 'Return_20d',
            'Sharpe_5d', 'Sharpe_10d', 'Sharpe_20d',
            
            # Volatility
            'Realized_Vol_5d', 'Realized_Vol_20d', 'Vol_of_Vol', 'Skewness_20d',
            
            # Technical
            'RSI_14', 'RSI_Oversold', 'RSI_Overbought',
            'Price_to_MA20', 'Price_to_MA50', 'MA20_Slope',
            
            # Time
            'Day_of_Week', 'Is_Month_End'
        ]
        
        # Remove stocks with insufficient data
        min_price = 5.0
        min_volume = 100000
        
        valid_stocks = []
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker]
            avg_price = ticker_data['Close'].mean()
            avg_volume = ticker_data['Volume'].mean()
            
            if avg_price >= min_price and avg_volume >= min_volume and len(ticker_data) >= 1000:
                valid_stocks.append(ticker)
            else:
                logger.info(f"Removing {ticker}: price=${avg_price:.2f}, volume={avg_volume:,.0f}")
        
        df = df[df['Ticker'].isin(valid_stocks)]
        
        # Handle infinities and missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill by ticker
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            df.loc[ticker_mask] = df.loc[ticker_mask].fillna(method='ffill')
        
        # Drop remaining NaNs
        df = df.dropna()
        
        # Keep only feature columns + metadata
        keep_cols = self.feature_columns + ['Ticker', 'Target']
        df = df[keep_cols]
        
        logger.info(f"Final dataset: {df.shape[0]} rows, {len(self.feature_columns)} features")
        logger.info(f"Class balance: {df['Target'].value_counts(normalize=True).to_dict()}")
        
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
            'dataset_shape': list(df.shape),
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
        df = self.add_targets(df, horizon=5, threshold=0.02)  # 20-day, 2% moves
        df = self.clean_data(df)
        
        # Save processed data
        output_path = self.save_data(df)
        
        logger.info("Processing complete!")
        return output_path


def main():
    """Run the optimized data processing pipeline"""
    try:
        processor = StockDataProcessor()
        output_path = processor.run_pipeline()
        
        print(f"\nData processing complete!")
        print(f"Output: {output_path}")
        print(f"Features: {len(processor.feature_columns)}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()