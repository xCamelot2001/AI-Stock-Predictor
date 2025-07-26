import os
import pandas as pd
import numpy as np
import logging
import ta
import warnings
# MODIFICATION: Corrected the import location for MACD
from ta.momentum import PercentagePriceOscillator
from ta.trend    import MACD
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataProcessor:
    """Processes stock data with lagged features for prediction."""
    
    def __init__(self):
        self.feature_columns = []
        os.makedirs("data/processed", exist_ok=True)
        logger.info("Optimized data processor initialized")
    
    def load_data(self, filepath="data/raw/multi_stock_merged.csv"):
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Date', 'Ticker'])
        return df

    def add_features_and_target(self, df):
        """Adds all technical features and the binary target."""
        logger.info("Adding features and target...")
        
        processed_dfs = []
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy().set_index('Date')

            # --- Create Features ---
            ticker_df['Log_Return'] = np.log(ticker_df['Close'] / ticker_df['Close'].shift(1))
            ticker_df['Intraday_Range'] = (ticker_df['High'] - ticker_df['Low']) / ticker_df['Close']
            ticker_df['Close_Open_Ratio'] = ticker_df['Close'] / ticker_df['Open']
            ticker_df['Volume_Ratio_20'] = ticker_df['Volume'] / ticker_df['Volume'].rolling(20).mean()
            ticker_df['Price_to_MA20'] = ticker_df['Close'] / ticker_df['Close'].rolling(20).mean()
            
            ticker_df['Return_5d'] = ticker_df['Close'].pct_change(5)
            ticker_df['Realized_Vol_5d'] = ticker_df['Log_Return'].rolling(5).std() * np.sqrt(252)
            
            ticker_df['RSI_14'] = ta.momentum.rsi(ticker_df['Close'], window=14)
            macd = MACD(ticker_df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            ticker_df['MACD_Diff'] = macd.macd_diff()
            
            # --- Create Target ---
            ticker_df['Target'] = (ticker_df['Close'].shift(-1) > ticker_df['Close']).astype(int)

            processed_dfs.append(ticker_df.reset_index())
        
        df_featured = pd.concat(processed_dfs, ignore_index=True)
        df_featured = df_featured.sort_values(['Date', 'Ticker'])
        return df_featured

    def create_lagged_features(self, df, lag_periods=3):
        """Creates lagged versions of features to prevent data leakage."""
        logger.info(f"Creating lagged features for {lag_periods} periods...")
        
        base_features = [col for col in df.columns if col not in ['Date', 'Ticker', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        for feature in base_features:
            for lag in range(1, lag_periods + 1):
                lag_col_name = f'{feature}_lag_{lag}'
                df[lag_col_name] = df.groupby('Ticker')[feature].shift(lag)
        
        self.feature_columns = [col for col in df.columns if 'lag' in col]
        
        df = df.dropna(subset=self.feature_columns + ['Target'])
        return df

    def clean_and_save(self, df):
        """Cleans and saves the final dataset."""
        keep_cols = ['Date', 'Ticker', 'Target'] + self.feature_columns
        df_final = df[keep_cols].set_index('Date')

        output_path = "data/processed/stock_data_lagged.csv"
        df_final.to_csv(output_path)
        
        logger.info(f"Final dataset: {df_final.shape[0]} rows, {len(self.feature_columns)} features")
        logger.info(f"Class balance: {df_final['Target'].value_counts(normalize=True).to_dict()}")
        
        return output_path

    def run_pipeline(self, input_file="data/raw/multi_stock_merged.csv"):
        """Run complete data processing pipeline"""
        logger.info("Starting data processing pipeline")
        df = self.load_data(input_file)
        df = self.add_features_and_target(df)
        df = self.create_lagged_features(df)
        output_path = self.clean_and_save(df)
        logger.info("Processing complete!")
        return output_path

def main():
    processor = StockDataProcessor()
    output_path = processor.run_pipeline()
    print(f"\nData processing complete!\nOutput: {output_path}\nFeatures: {len(processor.feature_columns)}")

if __name__ == "__main__":
    main()