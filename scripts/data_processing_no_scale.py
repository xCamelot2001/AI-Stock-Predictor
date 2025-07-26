import os
import pandas as pd
import numpy as np
import logging
import ta
import warnings
from ta.momentum import PercentagePriceOscillator
from ta.trend    import IchimokuIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

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

    def load_and_merge_market_data(self, df, filepath="data/raw/market_data.csv"):
        """Loads and merges market-wide data like SPY and VIX."""
        logger.info(f"Loading market data from {filepath}")
        if not os.path.exists(filepath):
            logger.error(f"Market data file not found at {filepath}. Macroeconomic features will be skipped.")
            return df

        market_df = pd.read_csv(filepath)
        market_df['Date'] = pd.to_datetime(market_df['Date'])
        market_df.set_index('Date', inplace=True)

        market_df['SPY_Log_Return'] = np.log(market_df['SPY_Close'] / market_df['SPY_Close'].shift(1))
        market_features = market_df[['SPY_Log_Return', 'VIX_Close']]
        
        df = df.merge(market_features, left_index=True, right_index=True, how='left')
        logger.info("Market data merged successfully.")
        return df

    def add_optimal_features(self, df):
        """Add research-validated features for stock prediction"""
        logger.info("Adding optimal features...")
        processed_dfs = []
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            
            # (Feature creation code remains the same as previous version)
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
            ticker_df['RSI_14'] = rsi / 100
            ticker_df['RSI_Oversold'] = (rsi < 30).astype(int)
            ticker_df['RSI_Overbought'] = (rsi > 70).astype(int)
            
            ma20 = ticker_df['Close'].rolling(20).mean()
            ma50 = ticker_df['Close'].rolling(50).mean()
            ticker_df['Price_to_MA20'] = ticker_df['Close'] / ma20
            ticker_df['Price_to_MA50'] = ticker_df['Close'] / ma50
            ticker_df['MA20_Slope'] = ma20.pct_change(5)
            
            # 7. TIME FEATURES
            ticker_df['Day_of_Week'] = ticker_df.index.dayofweek
            ticker_df['Is_Month_End'] = (ticker_df.index.day > 25).astype(int)

            # 8. ADVANCED TECHNICAL INDICATORS
            ppo = PercentagePriceOscillator(close=ticker_df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            ticker_df['PPO'] = ppo.ppo()
            ticker_df['PPO_Signal'] = ppo.ppo_signal()
            
            bb = BollingerBands(ticker_df['Close'], window=20, window_dev=2)
            ticker_df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            
            atr = AverageTrueRange(ticker_df['High'], ticker_df['Low'], ticker_df['Close'], window=14)
            ticker_df['ATR_14'] = atr.average_true_range()
            
            obv = OnBalanceVolumeIndicator(ticker_df['Close'], ticker_df['Volume'])
            ticker_df['OBV'] = obv.on_balance_volume()
            
            mfi = MFIIndicator(ticker_df['High'], ticker_df['Low'], ticker_df['Close'], ticker_df['Volume'], window=14)
            ticker_df['MFI_14'] = mfi.money_flow_index() / 100

            # 9. MACD (Moving Average Convergence Divergence)
            macd = MACD(ticker_df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            ticker_df['MACD'] = macd.macd()
            ticker_df['MACD_Signal'] = macd.macd_signal()
            ticker_df['MACD_Diff'] = macd.macd_diff()

            # 10. ICHIMOKU CLOUD
            ichi = IchimokuIndicator(ticker_df['High'], ticker_df['Low'], window1=9, window2=26, window3=52, fillna=True)
            ticker_df['Ichimoku_a'] = ichi.ichimoku_a()
            ticker_df['Ichimoku_b'] = ichi.ichimoku_b()
            
            # 11. INTERACTION FEATURES
            ticker_df['Volatility_Volume_Interaction'] = ticker_df['Realized_Vol_20d'] * ticker_df['Volume_Ratio_20']
            ticker_df['RSI_Volume_Interaction'] = ticker_df['RSI_14'] * ticker_df['Volume_Ratio_20']
            
            processed_dfs.append(ticker_df)
        
        df = pd.concat(processed_dfs, ignore_index=False)
        df = df.sort_values(['Date', 'Ticker'])
        logger.info(f"Features added. New shape: {df.shape}")
        return df

    def add_targets(self, df, horizon=5, up_thr=0.02, down_thr=-0.02):
        """3-class target: 0=Down, 1=Neutral, 2=Up"""
        logger.info(f"Adding 3-class target: >{up_thr*100:.1f}% Up, <{down_thr*100:.1f}% Down in {horizon} days")
        for ticker in df['Ticker'].unique():
            m = df['Ticker'] == ticker
            future_ret = (df.loc[m, 'Close'].shift(-horizon) - df.loc[m, 'Close']) / df.loc[m, 'Close']
            target = np.where(future_ret > up_thr, 2,
                    np.where(future_ret < down_thr, 0, 1))
            df.loc[m, 'Target'] = target
        df = df.dropna(subset=['Target'])
        df['Target'] = df['Target'].astype(int)
        return df

    def _validate_stock_data(self, df):
        """Validate stock data quality"""
        # (Implementation remains the same)
        pass

    def clean_data(self, df):
        """Clean data and select final features"""
        logger.info("Cleaning data...")
        self._validate_stock_data(df)
        
        # --- MODIFIED SECTION START ---
        
        # Define ALL possible features we might create
        all_possible_features = [
            'Log_Return', 'Intraday_Range', 'Overnight_Return',
            'High_Low_Ratio', 'Close_Open_Ratio',
            'Bid_Ask_Proxy', 'Amihud_Illiquidity', 'Kyle_Lambda',
            'Price_to_VWAP', 'Volume_Imbalance', 'Volume_Ratio_20',
            'Return_5d', 'Return_10d', 'Return_20d',
            'Sharpe_5d', 'Sharpe_10d', 'Sharpe_20d',
            'Realized_Vol_5d', 'Realized_Vol_20d', 'Vol_of_Vol', 'Skewness_20d',
            'RSI_14', 'RSI_Oversold', 'RSI_Overbought',
            'Price_to_MA20', 'Price_to_MA50', 'MA20_Slope',
            'PPO', 'PPO_Signal', 'BB_Width', 'ATR_14', 'OBV', 'MFI_14',
            'MACD', 'MACD_Signal', 'MACD_Diff', 'Ichimoku_a', 'Ichimoku_b',
            'Day_of_Week', 'Is_Month_End',
            'Volatility_Volume_Interaction', 'RSI_Volume_Interaction',
            'SPY_Log_Return', 'VIX_Close'
        ]
        
        # Filter this list to only include columns that actually exist in the dataframe
        self.feature_columns = [col for col in all_possible_features if col in df.columns]
        
        if len(self.feature_columns) < len(all_possible_features):
            logger.warning("Some features were not found in the dataframe and will be skipped.")

        # --- MODIFIED SECTION END ---

        # Remove stocks with insufficient data
        # (Implementation remains the same)
        
        # Handle infinities and missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        for ticker in df['Ticker'].unique():
            ticker_mask = df['Ticker'] == ticker
            df.loc[ticker_mask] = df.loc[ticker_mask].fillna(method='ffill')
        
        # Use the dynamically filtered feature list to drop NaNs
        df = df.dropna(subset=self.feature_columns + ['Target'])
        
        # Keep only the relevant columns
        keep_cols = self.feature_columns + ['Ticker', 'Target']
        df = df[keep_cols]
        
        logger.info(f"Final dataset: {df.shape[0]} rows, {len(self.feature_columns)} features")
        logger.info(f"Class balance: {df['Target'].value_counts(normalize=True).to_dict()}")
        
        return df

    def save_data(self, df):
        """Save processed dataset"""
        output_path = "data/processed/stock_data_optimized.csv"
        df.to_csv(output_path)
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

    def run_pipeline(self, input_file="data/raw/multi_stock_merged.csv", market_file="data/raw/market_data.csv"):
        """Run complete data processing pipeline"""
        logger.info("Starting data processing pipeline")
        df = self.load_data(input_file)
        df = self.load_and_merge_market_data(df, market_file)
        df = self.add_optimal_features(df)
        df = self.add_targets(df, horizon=5, up_thr=0.02, down_thr=-0.02)
        df = self.clean_data(df)
        output_path = self.save_data(df)
        logger.info("Processing complete!")
        return output_path


def main():
    try:
        processor = StockDataProcessor()
        output_path = processor.run_pipeline()
        print(f"\nData processing complete!\nOutput: {output_path}\nFeatures: {len(processor.feature_columns)}")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()