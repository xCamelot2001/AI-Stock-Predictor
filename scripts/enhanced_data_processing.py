"""
Enhanced Stock Data Processing Pipeline
======================================

Professional data processing pipeline using best practices:
- Uses 'ta' library for reliable technical indicators
- Comprehensive feature engineering
- Robust data validation and cleaning
- Scalable for multiple stocks
- Binary classification targets for LSTM

Dependencies: pip install ta

Usage: python scripts/enhanced_data_processing.py
"""

import pandas as pd
import numpy as np
import logging
import os
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json
import ta  # Technical Analysis library

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedStockDataProcessor:
    """
    Professional stock data processor with comprehensive feature engineering
    
    Features:
    - 60+ technical indicators using 'ta' library
    - Price-based features (returns, ratios, volatility)
    - Volume analysis features
    - Market structure features
    - Binary classification targets
    - Multi-timeframe analysis
    - Robust data validation
    """
    
    def __init__(self, target_prediction_days: int = 1):
        """
        Initialize processor
        
        Args:
            target_prediction_days: Days ahead to predict (1 = next day)
        """
        self.target_prediction_days = target_prediction_days
        
        # Create output directory
        os.makedirs("data/processed", exist_ok=True)
        
        logger.info("🚀 Enhanced Stock Data Processor initialized")
        logger.info(f"📊 Target prediction horizon: {target_prediction_days} day(s)")
    
    def load_merged_data(self, filepath: str = "data/raw/multi_stock_merged.csv") -> pd.DataFrame:
        """Load merged stock data with validation"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"📂 Loading data from: {filepath}")
        
        # Load data
        df = pd.read_csv(filepath, parse_dates=["Date"])
        
        # Validate required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by ticker and date
        df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        
        # Basic validation
        self._validate_stock_data(df)
        
        logger.info(f"✅ Loaded data: {df.shape} ({df['Ticker'].nunique()} stocks)")
        logger.info(f"📅 Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        return df
    
    def _validate_stock_data(self, df: pd.DataFrame) -> None:
        """Validate stock data quality"""
        
        issues = []
        
        # Check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if (df[col] <= 0).any():
                issues.append(f"Found non-positive values in {col}")
        
        # Check OHLC logic
        if ((df['High'] < df['Low']) | 
            (df['High'] < df['Open']) | 
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) | 
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
    
    def create_comprehensive_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical indicators using 'ta' library
        
        Categories:
        1. Trend indicators (SMA, EMA, MACD, ADX, etc.)
        2. Momentum indicators (RSI, Stochastic, Williams %R, etc.)
        3. Volatility indicators (Bollinger Bands, ATR, etc.)
        4. Volume indicators (OBV, CMF, etc.)
        """
        logger.info("🔧 Creating comprehensive technical indicators...")
        
        processed_stocks = []
        
        for ticker in df['Ticker'].unique():
            logger.info(f"   📊 Processing {ticker}...")
            
            stock_df = df[df['Ticker'] == ticker].copy().sort_values('Date')
            
            # Basic OHLCV
            high = stock_df['High']
            low = stock_df['Low']
            close = stock_df['Close']
            open_price = stock_df['Open']
            volume = stock_df['Volume']
            
            # =====================================
            # 1. TREND INDICATORS
            # =====================================
            
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                stock_df[f'SMA_{period}'] = ta.trend.sma_indicator(close, window=period)
            
            # Exponential Moving Averages
            for period in [12, 26, 50]:
                stock_df[f'EMA_{period}'] = ta.trend.ema_indicator(close, window=period)
            
            # MACD
            macd = ta.trend.MACD(close)
            stock_df['MACD'] = macd.macd()
            stock_df['MACD_Signal'] = macd.macd_signal()
            stock_df['MACD_Histogram'] = macd.macd_diff()
            
            # Average Directional Index (ADX)
            stock_df['ADX'] = ta.trend.adx(high, low, close, window=14)
            stock_df['DI_Plus'] = ta.trend.adx_pos(high, low, close, window=14)
            stock_df['DI_Minus'] = ta.trend.adx_neg(high, low, close, window=14)
            
            # Parabolic SAR
            stock_df['PSAR'] = ta.trend.psar_down(high, low, close)
            
            # Ichimoku
            ichimoku = ta.trend.IchimokuIndicator(high, low)
            stock_df['Ichimoku_Conv'] = ichimoku.ichimoku_conversion_line()
            stock_df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
            stock_df['Ichimoku_A'] = ichimoku.ichimoku_a()
            stock_df['Ichimoku_B'] = ichimoku.ichimoku_b()
            
            # =====================================
            # 2. MOMENTUM INDICATORS
            # =====================================
            
            # RSI
            stock_df['RSI_14'] = ta.momentum.rsi(close, window=14)
            stock_df['RSI_21'] = ta.momentum.rsi(close, window=21)
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(high, low, close)
            stock_df['Stoch_K'] = stoch.stoch()
            stock_df['Stoch_D'] = stoch.stoch_signal()
            
            # Williams %R
            stock_df['Williams_R'] = ta.momentum.williams_r(high, low, close, lbp=14)
            
            # Rate of Change
            for period in [10, 20]:
                stock_df[f'ROC_{period}'] = ta.momentum.roc(close, window=period)
            
            # Commodity Channel Index
            stock_df['CCI'] = ta.trend.cci(high, low, close, window=20)
            
            # Money Flow Index
            stock_df['MFI'] = ta.volume.money_flow_index(high, low, close, volume, window=14)
            
            # =====================================
            # 3. VOLATILITY INDICATORS
            # =====================================
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close)
            stock_df['BB_Upper'] = bb.bollinger_hband()
            stock_df['BB_Middle'] = bb.bollinger_mavg()
            stock_df['BB_Lower'] = bb.bollinger_lband()
            stock_df['BB_Width'] = bb.bollinger_wband()
            stock_df['BB_Position'] = bb.bollinger_pband()
            
            # Average True Range
            stock_df['ATR_14'] = ta.volatility.average_true_range(high, low, close, window=14)
            
            # Donchian Channels
            donchian = ta.volatility.DonchianChannel(high, low, close)
            stock_df['DC_Upper'] = donchian.donchian_channel_hband()
            stock_df['DC_Lower'] = donchian.donchian_channel_lband()
            
            # Keltner Channels
            keltner = ta.volatility.KeltnerChannel(high, low, close)
            stock_df['KC_Upper'] = keltner.keltner_channel_hband()
            stock_df['KC_Lower'] = keltner.keltner_channel_lband()
            
            # =====================================
            # 4. VOLUME INDICATORS
            # =====================================
            
            # On-Balance Volume
            stock_df['OBV'] = ta.volume.on_balance_volume(close, volume)
            
            # Chaikin Money Flow
            stock_df['CMF'] = ta.volume.chaikin_money_flow(high, low, close, volume, window=20)
            
            # Accumulation/Distribution Line
            stock_df['ADL'] = ta.volume.acc_dist_index(high, low, close, volume)
            
            # Volume Price Trend
            stock_df['VPT'] = ta.volume.volume_price_trend(close, volume)
            
            # Negative Volume Index
            stock_df['NVI'] = ta.volume.negative_volume_index(close, volume)
            
            # Volume Weighted Average Price (approximation)
            stock_df['VWAP'] = ta.volume.volume_weighted_average_price(high, low, close, volume)
            
            processed_stocks.append(stock_df)
        
        # Combine all stocks
        result_df = pd.concat(processed_stocks, ignore_index=True)
        result_df = result_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        
        logger.info("✅ Technical indicators created")
        return result_df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced price-based features"""
        logger.info("🔧 Creating price features...")
        
        processed_stocks = []
        
        for ticker in df['Ticker'].unique():
            stock_df = df[df['Ticker'] == ticker].copy().sort_values('Date')
            
            # =====================================
            # RETURNS AND PRICE CHANGES
            # =====================================
            
            # Basic returns
            stock_df['Close_Return'] = stock_df['Close'].pct_change()
            stock_df['Open_Return'] = stock_df['Open'].pct_change()
            stock_df['High_Return'] = stock_df['High'].pct_change()
            stock_df['Low_Return'] = stock_df['Low'].pct_change()
            
            # Log returns
            stock_df['Log_Return'] = np.log(stock_df['Close'] / stock_df['Close'].shift(1))
            
            # Multi-period returns
            for period in [2, 3, 5, 10, 20]:
                stock_df[f'Return_{period}d'] = stock_df['Close'].pct_change(period)
            
            # =====================================
            # PRICE RATIOS AND RELATIONSHIPS
            # =====================================
            
            # Intraday ratios
            stock_df['High_Low_Ratio'] = stock_df['High'] / stock_df['Low']
            stock_df['Close_Open_Ratio'] = stock_df['Close'] / stock_df['Open']
            stock_df['High_Close_Ratio'] = stock_df['High'] / stock_df['Close']
            stock_df['Low_Close_Ratio'] = stock_df['Low'] / stock_df['Close']
            
            # Gap analysis
            stock_df['Gap'] = (stock_df['Open'] - stock_df['Close'].shift(1)) / stock_df['Close'].shift(1)
            
            # True Range components
            stock_df['True_Range'] = np.maximum(
                stock_df['High'] - stock_df['Low'],
                np.maximum(
                    abs(stock_df['High'] - stock_df['Close'].shift(1)),
                    abs(stock_df['Low'] - stock_df['Close'].shift(1))
                )
            )
            
            # =====================================
            # VOLATILITY FEATURES
            # =====================================
            
            # Rolling volatility (different windows)
            for window in [5, 10, 20, 30]:
                stock_df[f'Volatility_{window}d'] = stock_df['Close_Return'].rolling(window=window).std()
            
            # Realized volatility (annualized)
            stock_df['Realized_Vol_20d'] = stock_df['Close_Return'].rolling(window=20).std() * np.sqrt(252)
            
            # Volatility of volatility
            stock_df['Vol_of_Vol'] = stock_df['Volatility_20d'].rolling(window=10).std()
            
            # Price range features
            stock_df['Price_Range'] = stock_df['High'] - stock_df['Low']
            stock_df['Price_Range_Pct'] = stock_df['Price_Range'] / stock_df['Close']
            
            # =====================================
            # VOLUME FEATURES
            # =====================================
            
            # Volume changes
            stock_df['Volume_Return'] = stock_df['Volume'].pct_change()
            
            # Volume moving averages
            for window in [5, 20, 50]:
                stock_df[f'Volume_MA_{window}'] = stock_df['Volume'].rolling(window=window).mean()
                stock_df[f'Volume_Ratio_{window}'] = stock_df['Volume'] / stock_df[f'Volume_MA_{window}']
            
            # Price-Volume relationships
            stock_df['Price_Volume_Trend'] = stock_df['Close_Return'] * stock_df['Volume_Ratio_20']
            stock_df['Volume_Price_Correlation'] = stock_df['Close_Return'].rolling(window=20).corr(
                stock_df['Volume_Return']
            )
            
            # =====================================
            # POSITION FEATURES (vs moving averages)
            # =====================================
            
            # Position relative to moving averages
            for ma_period in [20, 50, 200]:
                if f'SMA_{ma_period}' in stock_df.columns:
                    stock_df[f'Close_vs_SMA{ma_period}'] = stock_df['Close'] / stock_df[f'SMA_{ma_period}']
            
            # Distance from highs/lows
            for window in [20, 50, 252]:
                stock_df[f'High_{window}d'] = stock_df['High'].rolling(window=window).max()
                stock_df[f'Low_{window}d'] = stock_df['Low'].rolling(window=window).min()
                stock_df[f'Close_vs_High{window}d'] = stock_df['Close'] / stock_df[f'High_{window}d']
                stock_df[f'Close_vs_Low{window}d'] = stock_df['Close'] / stock_df[f'Low_{window}d']
            
            # =====================================
            # TREND FEATURES
            # =====================================
            
            # Moving average trends
            if 'SMA_20' in stock_df.columns and 'SMA_50' in stock_df.columns:
                stock_df['MA_Trend_20_50'] = (stock_df['SMA_20'] > stock_df['SMA_50']).astype(int)
            
            if 'EMA_12' in stock_df.columns and 'EMA_26' in stock_df.columns:
                stock_df['EMA_Trend_12_26'] = (stock_df['EMA_12'] > stock_df['EMA_26']).astype(int)
            
            # Price momentum
            for window in [5, 10, 20]:
                stock_df[f'Momentum_{window}d'] = stock_df['Close'] / stock_df['Close'].shift(window) - 1
            
            processed_stocks.append(stock_df)
        
        # Combine all stocks
        result_df = pd.concat(processed_stocks, ignore_index=True)
        result_df = result_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        
        logger.info("✅ Price features created")
        return result_df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary classification target variable"""
        logger.info(f"🎯 Creating target variable ({self.target_prediction_days} day ahead)...")
        
        processed_stocks = []
        
        for ticker in df['Ticker'].unique():
            stock_df = df[df['Ticker'] == ticker].copy().sort_values('Date')
            
            # Future close price
            future_close = stock_df['Close'].shift(-self.target_prediction_days)
            
            # Binary target: 1 if price goes up, 0 if down
            stock_df['Target'] = (future_close > stock_df['Close']).astype(int)
            
            # Remove rows where we can't predict (at the end)
            stock_df = stock_df.iloc[:-self.target_prediction_days]
            
            processed_stocks.append(stock_df)
        
        # Combine all stocks
        result_df = pd.concat(processed_stocks, ignore_index=True)
        result_df = result_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        
        # Calculate target distribution
        self._analyze_target_distribution(result_df)
        
        return result_df
    
    def _analyze_target_distribution(self, df: pd.DataFrame) -> None:
        """Analyze target variable distribution"""
        
        total_samples = len(df)
        up_samples = df['Target'].sum()
        down_samples = total_samples - up_samples
        
        logger.info("📊 Target Variable Analysis:")
        logger.info(f"   📈 Up days: {up_samples:,} ({up_samples/total_samples*100:.1f}%)")
        logger.info(f"   📉 Down days: {down_samples:,} ({down_samples/total_samples*100:.1f}%)")
        
        # Per-stock analysis
        logger.info("\n📊 Per-stock target distribution:")
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker]
            ticker_up = ticker_df['Target'].sum()
            ticker_total = len(ticker_df)
            logger.info(f"   {ticker}: {ticker_up}/{ticker_total} ({ticker_up/ticker_total*100:.1f}% up)")
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and handle missing values with validation"""
        logger.info("🧹 Cleaning and validating data...")
        
        initial_rows = len(df)
        initial_features = len(df.columns)
        
        # Check for infinite values
        inf_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if np.isinf(df[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            logger.warning(f"⚠️ Found infinite values in columns: {inf_cols}")
            df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remove columns with too many missing values (>50%)
        missing_pct = df.isnull().sum() / len(df)
        high_missing_cols = missing_pct[missing_pct > 0.5].index.tolist()
        
        if high_missing_cols:
            logger.warning(f"⚠️ Removing columns with >50% missing values: {high_missing_cols}")
            df = df.drop(columns=high_missing_cols)
        
        # Remove rows with any missing values
        df_clean = df.dropna()
        
        final_rows = len(df_clean)
        final_features = len(df_clean.columns)
        dropped_rows = initial_rows - final_rows
        dropped_features = initial_features - final_features
        
        logger.info(f"📊 Data cleaning summary:")
        logger.info(f"   Rows: {initial_rows:,} → {final_rows:,} (dropped: {dropped_rows:,})")
        logger.info(f"   Features: {initial_features} → {final_features} (dropped: {dropped_features})")
        
        # Check data sufficiency per stock
        self._check_data_sufficiency(df_clean)
        
        return df_clean
    
    def _check_data_sufficiency(self, df: pd.DataFrame) -> None:
        """Check if we have sufficient data per stock"""
        logger.info("\n📊 Data sufficiency check:")
        
        min_samples_recommended = 1000  # Minimum for reliable LSTM training
        
        for ticker in df['Ticker'].unique():
            ticker_count = len(df[df['Ticker'] == ticker])
            
            if ticker_count < min_samples_recommended:
                logger.warning(f"⚠️ {ticker}: Only {ticker_count:,} samples (< {min_samples_recommended:,})")
            else:
                logger.info(f"✅ {ticker}: {ticker_count:,} samples")
    
    def save_processed_data(self, df: pd.DataFrame) -> Tuple[str, str]:
        """Save processed data with comprehensive metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main dataset
        filename = f"data/processed/enhanced_multi_stock_{timestamp}.csv"
        df.to_csv(filename, index=False)
        
        # Create comprehensive metadata
        feature_cols = [col for col in df.columns if col not in ['Date', 'Ticker', 'Target']]
        
        # Categorize features
        feature_categories = self._categorize_features(feature_cols)
        
        metadata = {
            'processing_info': {
                'processing_date': datetime.now().isoformat(),
                'processor_version': 'EnhancedStockDataProcessor v1.0',
                'target_prediction_days': self.target_prediction_days
            },
            'dataset_summary': {
                'total_rows': len(df),
                'total_features': len(feature_cols),
                'stocks': sorted(df['Ticker'].unique().tolist()),
                'date_range': {
                    'start': str(df['Date'].min().date()),
                    'end': str(df['Date'].max().date()),
                    'total_days': (df['Date'].max() - df['Date'].min()).days
                }
            },
            'target_analysis': {
                'target_distribution': df['Target'].value_counts().to_dict(),
                'class_balance': {
                    'up_percentage': float(df['Target'].mean() * 100),
                    'down_percentage': float((1 - df['Target'].mean()) * 100)
                },
                'per_stock_distribution': df.groupby('Ticker')['Target'].agg(['count', 'mean']).to_dict()
            },
            'feature_analysis': {
                'total_features': len(feature_cols),
                'feature_categories': feature_categories,
                'features_by_category_count': {k: len(v) for k, v in feature_categories.items()}
            },
            'samples_per_stock': df['Ticker'].value_counts().to_dict(),
            'data_quality': {
                'missing_values': df.isnull().sum().sum(),
                'infinite_values': 0,  # Should be 0 after cleaning
                'duplicate_rows': df.duplicated().sum()
            }
        }
        
        # Save metadata
        metadata_file = f"data/processed/enhanced_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved processed data: {filename}")
        logger.info(f"💾 Saved metadata: {metadata_file}")
        logger.info(f"📊 Final dataset: {len(df):,} rows × {len(df.columns)} columns")
        
        return filename, metadata_file
    
    def _categorize_features(self, feature_cols: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type"""
        
        categories = {
            'price_basic': [],
            'technical_trend': [],
            'technical_momentum': [],
            'technical_volatility': [],
            'technical_volume': [],
            'price_returns': [],
            'price_ratios': [],
            'volatility_measures': [],
            'volume_features': [],
            'position_features': [],
            'trend_features': [],
            'other': []
        }
        
        for col in feature_cols:
            col_lower = col.lower()
            
            # Basic OHLCV
            if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                categories['price_basic'].append(col)
            
            # Technical indicators - Trend
            elif any(x in col for x in ['SMA', 'EMA', 'MACD', 'ADX', 'PSAR', 'Ichimoku', 'CCI']):
                categories['technical_trend'].append(col)
            
            # Technical indicators - Momentum
            elif any(x in col for x in ['RSI', 'Stoch', 'Williams', 'ROC', 'MFI']):
                categories['technical_momentum'].append(col)
            
            # Technical indicators - Volatility
            elif any(x in col for x in ['BB_', 'ATR', 'DC_', 'KC_']):
                categories['technical_volatility'].append(col)
            
            # Technical indicators - Volume
            elif any(x in col for x in ['OBV', 'CMF', 'ADL', 'VPT', 'NVI', 'VWAP']):
                categories['technical_volume'].append(col)
            
            # Returns
            elif 'return' in col_lower and col != 'Volume_Return':
                categories['price_returns'].append(col)
            
            # Ratios
            elif 'ratio' in col_lower:
                categories['price_ratios'].append(col)
            
            # Volatility measures
            elif any(x in col_lower for x in ['volatility', 'vol_', 'true_range']):
                categories['volatility_measures'].append(col)
            
            # Volume features
            elif 'volume' in col_lower:
                categories['volume_features'].append(col)
            
            # Position features
            elif any(x in col for x in ['_vs_', 'Position']):
                categories['position_features'].append(col)
            
            # Trend features
            elif any(x in col for x in ['Trend', 'Momentum']):
                categories['trend_features'].append(col)
            
            else:
                categories['other'].append(col)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def run_complete_pipeline(self, input_file: str = "data/raw/multi_stock_merged.csv") -> Tuple[pd.DataFrame, str, str]:
        """Run the complete enhanced processing pipeline"""
        
        logger.info("🚀 Starting Enhanced Data Processing Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load data
        df = self.load_merged_data(input_file)
        
        # Step 2: Create technical indicators
        df = self.create_comprehensive_technical_indicators(df)
        
        # Step 3: Create price features
        df = self.create_price_features(df)
        
        # Step 4: Create target variable
        df = self.create_target_variable(df)
        
        # Step 5: Clean and validate data
        df = self.clean_and_validate_data(df)
        
        # Step 6: Save processed data
        output_file, metadata_file = self.save_processed_data(df)
        
        logger.info("=" * 60)
        logger.info("🎉 Enhanced Processing Pipeline Complete!")
        
        return df, output_file, metadata_file


def main():
    """Run the enhanced data processing pipeline"""
    
    try:
        # Initialize processor
        processor = EnhancedStockDataProcessor(target_prediction_days=1)
        
        # Run processing pipeline
        processed_df, output_file, metadata_file = processor.run_complete_pipeline()
        
        print(f"\n✅ Enhanced Processing Complete!")
        print(f"📊 Processed data: {output_file}")
        print(f"📄 Metadata: {metadata_file}")
        print(f"🚀 Ready for advanced LSTM training!")
        
        # Display summary
        print(f"\n📈 Enhanced Dataset Summary:")
        print(f"Stocks: {processed_df['Ticker'].nunique()}")
        print(f"Total samples: {len(processed_df):,}")
        print(f"Total features: {len([col for col in processed_df.columns if col not in ['Date', 'Ticker', 'Target']])}")
        print(f"Date range: {processed_df['Date'].min().date()} to {processed_df['Date'].max().date()}")
        
        # Feature breakdown
        print(f"\n🔧 Feature Categories:")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for category, count in metadata['feature_analysis']['features_by_category_count'].items():
            print(f"   {category.replace('_', ' ').title()}: {count} features")
        
    except Exception as e:
        logger.error(f"❌ Enhanced processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
