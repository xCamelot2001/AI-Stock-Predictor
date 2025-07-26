"""
Data Integration Script - Combine Stock Data with Sentiment
==========================================================

Purpose: Integrate processed stock data with sentiment analysis results
- Load processed stock data (35 technical features)
- Load daily sentiment data (6 sentiment features)
- Merge by date with proper alignment
- Create enhanced dataset for LSTM training

Research Goal: Test if sentiment features improve stock prediction accuracy
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIntegrator:
    """
    Integrate stock price data with sentiment analysis results
    
    Creates enhanced dataset combining:
    - Technical indicators (35 features)
    - Sentiment features (6 features)
    - Target variable (up/down tomorrow)
    """
    
    def __init__(self):
        """Initialize data integrator"""
        os.makedirs('data/integrated', exist_ok=True)
        logger.info("🔗 Data integrator initialized")
    
    def load_latest_file(self, pattern, data_type):
        """Load the most recent file matching pattern"""
        files = glob.glob(pattern)
        
        if not files:
            logger.error(f"❌ No {data_type} files found: {pattern}")
            return None
        
        latest_file = max(files, key=os.path.getctime)
        logger.info(f"📊 Loading {data_type}: {latest_file}")
        
        return latest_file
    
    def load_stock_data(self):
        """Load processed stock data with technical indicators"""
        
        stock_file = "data/processed/stock_data_research_optimized.csv"
        
        if not os.path.exists(stock_file):
            logger.error(f"❌ Stock data file not found: {stock_file}")
            return None
        
        logger.info(f"📊 Loading stock data: {stock_file}")
        
        # Load stock data
        df_stock = pd.read_csv(stock_file)
        df_stock['Date'] = pd.to_datetime(df_stock['Date'])
        df_stock.set_index('Date', inplace=True)
        
        logger.info(f"✅ Loaded stock data: {len(df_stock)} rows, {len(df_stock.columns)} columns")
        logger.info(f"📅 Stock date range: {df_stock.index.min().date()} to {df_stock.index.max().date()}")
        logger.info(f"📊 Tickers: {sorted(df_stock['Ticker'].unique())}")
        
        return df_stock
    
    def load_sentiment_data(self):
        """Load daily sentiment data for all stocks"""
        
        pattern = "data/sentiment/ALL_STOCKS_daily_sentiment_*.csv"
        latest_file = self.load_latest_file(pattern, "sentiment data")
        
        if not latest_file:
            return None
        
        # Load sentiment data
        df_sentiment = pd.read_csv(latest_file)
        
        # Convert date column to datetime index and rename symbol to Ticker
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])
        df_sentiment = df_sentiment.rename(columns={'symbol': 'Ticker'})
        df_sentiment.set_index('date', inplace=True)
        
        logger.info(f"✅ Loaded sentiment data: {len(df_sentiment)} rows, {len(df_sentiment.columns)} columns")
        logger.info(f"📅 Sentiment date range: {df_sentiment.index.min().date()} to {df_sentiment.index.max().date()}")
        logger.info(f"📊 Sentiment tickers: {sorted(df_sentiment['Ticker'].unique())}")
        
        return df_sentiment
    
    def align_data_by_date(self, df_stock, df_sentiment):
        """
        Align stock and sentiment data by date and ticker
        
        Strategy:
        - Merge on both date and ticker for proper alignment
        - Only keep data where we have both stock AND sentiment data
        """
        logger.info("🔗 Aligning stock and sentiment data by date and ticker...")
        
        # Reset index to have Date as column for merging
        df_stock_reset = df_stock.reset_index()
        df_sentiment_reset = df_sentiment.reset_index()
        df_sentiment_reset = df_sentiment_reset.rename(columns={'date': 'Date'})
        
        # Get date ranges
        stock_start, stock_end = df_stock_reset['Date'].min(), df_stock_reset['Date'].max()
        sentiment_start, sentiment_end = df_sentiment_reset['Date'].min(), df_sentiment_reset['Date'].max()
        
        logger.info(f"📊 Date range comparison:")
        logger.info(f"   Stock: {stock_start.date()} to {stock_end.date()}")
        logger.info(f"   Sentiment: {sentiment_start.date()} to {sentiment_end.date()}")
        
        # INNER JOIN - merge on both Date and Ticker
        df_merged = pd.merge(df_stock_reset, df_sentiment_reset, on=['Date', 'Ticker'], how='inner')
        
        # Set Date back as index
        df_merged.set_index('Date', inplace=True)
        
        sentiment_cols = [col for col in df_sentiment.columns if col != 'Ticker']
        
        logger.info(f"📊 Merge results (inner join on Date + Ticker):")
        logger.info(f"   Original stock data rows: {len(df_stock)}")
        logger.info(f"   Sentiment data rows: {len(df_sentiment)}")
        logger.info(f"   Final merged rows: {len(df_merged)}")
        logger.info(f"   ✅ All rows have real sentiment data!")
        
        return df_merged, sentiment_cols
    
    def handle_missing_sentiment(self, df_merged, sentiment_cols, method='none'):
        """
        Handle missing sentiment data
        
        Since we're using inner join, there should be no missing sentiment data
        """
        logger.info("🔧 Checking for missing sentiment data...")
        
        missing_count = df_merged[sentiment_cols].isnull().sum().sum()
        
        if missing_count == 0:
            logger.info("✅ No missing sentiment data (inner join successful)")
        else:
            logger.warning(f"⚠️  Found {missing_count} missing sentiment values")
            # If any missing, drop those rows
            df_merged = df_merged.dropna(subset=sentiment_cols)
            logger.info(f"📊 Dropped rows with missing sentiment: {len(df_merged)} rows remaining")
        
        return df_merged
    
    def create_enhanced_features(self, df_merged):
        """
        Create additional features from combined data
        
        Ideas:
        - Sentiment momentum (change in sentiment)
        - Sentiment-price correlation features
        - Sentiment volatility
        """
        logger.info("🚀 Creating enhanced features...")
        
        # 1. Sentiment momentum (rate of change)
        if 'compound_score_mean' in df_merged.columns:
            df_merged['sentiment_momentum'] = df_merged['compound_score_mean'].diff()
            df_merged['sentiment_volatility'] = df_merged['compound_score_mean'].rolling(window=5).std()
        
        # 2. News volume momentum
        if 'article_count' in df_merged.columns:
            df_merged['news_volume_ma'] = df_merged['article_count'].rolling(window=3).mean()
            df_merged['news_volume_momentum'] = df_merged['article_count'].diff()
        
        # 3. Sentiment strength (absolute compound score)
        if 'compound_score_mean' in df_merged.columns:
            df_merged['sentiment_strength'] = abs(df_merged['compound_score_mean'])
        
        # 4. Sentiment divergence (positive vs negative ratio difference)
        if 'positive_ratio' in df_merged.columns and 'negative_ratio' in df_merged.columns:
            df_merged['sentiment_divergence'] = df_merged['positive_ratio'] - df_merged['negative_ratio']
        
        logger.info("✅ Enhanced features created")
        return df_merged
    
    def analyze_integration(self, df_merged, sentiment_cols):
        """Analyze the integrated dataset"""
        logger.info("📊 Analyzing integrated dataset...")
        
        # Basic statistics
        total_features = len(df_merged.columns) - 1  # Exclude target
        sentiment_features = len(sentiment_cols)
        price_features = total_features - sentiment_features
        
        logger.info(f"📈 Dataset composition:")
        logger.info(f"   Total rows: {len(df_merged)}")
        logger.info(f"   Total features: {total_features}")
        logger.info(f"   Price/technical features: {price_features}")
        logger.info(f"   Sentiment features: {sentiment_features}")
        
        # Correlation analysis
        if 'Target' in df_merged.columns and 'compound_score_mean' in df_merged.columns:
            target_corr = df_merged['Target'].corr(df_merged['compound_score_mean'])
            logger.info(f"🎯 Sentiment-Target correlation: {target_corr:.4f}")
        
        # Sentiment statistics
        if 'compound_score_mean' in df_merged.columns:
            sentiment_stats = df_merged['compound_score_mean'].describe()
            logger.info(f"📊 Sentiment statistics:")
            logger.info(f"   Mean: {sentiment_stats['mean']:.4f}")
            logger.info(f"   Std: {sentiment_stats['std']:.4f}")
            logger.info(f"   Range: {sentiment_stats['min']:.4f} to {sentiment_stats['max']:.4f}")
        
        return {
            'total_rows': len(df_merged),
            'total_features': total_features,
            'sentiment_features': sentiment_features,
            'price_features': price_features
        }
    
    def plot_integration_analysis(self, df_merged):
        """Create visualizations of the integrated data"""
        logger.info("📊 Creating integration analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Sentiment over time
        if 'compound_score_mean' in df_merged.columns:
            axes[0, 0].plot(df_merged.index, df_merged['compound_score_mean'])
            axes[0, 0].set_title('Sentiment Over Time')
            axes[0, 0].set_ylabel('Compound Sentiment Score')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Price vs Sentiment
        if 'Close' in df_merged.columns and 'compound_score_mean' in df_merged.columns:
            ax2 = axes[0, 1]
            ax2_twin = ax2.twinx()
            
            ax2.plot(df_merged.index, df_merged['Close'], color='blue', label='Price')
            ax2_twin.plot(df_merged.index, df_merged['compound_score_mean'], color='red', label='Sentiment')
            
            ax2.set_ylabel('Stock Price', color='blue')
            ax2_twin.set_ylabel('Sentiment Score', color='red')
            ax2.set_title('Price vs Sentiment')
            ax2.grid(True, alpha=0.3)
        
        # 3. Sentiment distribution
        if 'compound_score_mean' in df_merged.columns:
            axes[1, 0].hist(df_merged['compound_score_mean'].dropna(), bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Sentiment Distribution')
            axes[1, 0].set_xlabel('Compound Sentiment Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Target vs Sentiment
        if 'Target' in df_merged.columns and 'compound_score_mean' in df_merged.columns:
            sentiment_by_target = df_merged.groupby('Target')['compound_score_mean'].mean()
            axes[1, 1].bar(['Down (0)', 'Up (1)'], sentiment_by_target.values)
            axes[1, 1].set_title('Average Sentiment by Target')
            axes[1, 1].set_ylabel('Average Sentiment Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/integrated/integration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("📊 Integration analysis plot saved: data/integrated/integration_analysis.png")
    
    def save_integrated_data(self, df_merged, symbol, analysis_stats):
        """Save the integrated dataset"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save integrated data
        filename = f"data/integrated/{symbol}_integrated_{timestamp}.csv"
        df_merged.to_csv(filename)
        logger.info(f"💾 Saved integrated data: {filename}")
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'integration_date': datetime.now().isoformat(),
            'dataset_stats': analysis_stats,
            'feature_categories': {
                'price_technical': [col for col in df_merged.columns if col not in ['Target'] and 'sentiment' not in col.lower() and 'compound' not in col.lower() and 'positive' not in col.lower() and 'negative' not in col.lower() and 'neutral' not in col.lower() and 'article' not in col.lower() and 'news' not in col.lower()],
                'sentiment': [col for col in df_merged.columns if any(word in col.lower() for word in ['sentiment', 'compound', 'positive', 'negative', 'neutral', 'article', 'news'])],
                'target': ['Target']
            },
            'date_range': {
                'start': str(df_merged.index.min().date()),
                'end': str(df_merged.index.max().date())
            }
        }
        
        metadata_file = f"data/integrated/{symbol}_integrated_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        return filename, metadata_file
    
    def integrate_data(self, save=True):
        """
        Complete data integration pipeline - REAL SENTIMENT DATA ONLY
        
        Args:
            save: Whether to save results
            
        Strategy: Only use stock data where we have real sentiment data
        """
        logger.info(f"\n🚀 Starting data integration for all stocks")
        logger.info("📊 Strategy: Using ONLY dates with real sentiment data")
        
        # 1. Load stock data
        df_stock = self.load_stock_data()
        if df_stock is None:
            logger.error("❌ Failed to load stock data")
            return None
        
        # 2. Load sentiment data
        df_sentiment = self.load_sentiment_data()
        if df_sentiment is None:
            logger.error("❌ Failed to load sentiment data")
            return None
        
        # 3. Align data by date and ticker (INNER JOIN - only real sentiment dates)
        df_merged, sentiment_cols = self.align_data_by_date(df_stock, df_sentiment)
        
        # 4. Check for any missing sentiment data
        df_merged = self.handle_missing_sentiment(df_merged, sentiment_cols)
        
        # 5. Create enhanced features
        df_merged = self.create_enhanced_features(df_merged)
        
        # 6. Analyze integration
        analysis_stats = self.analyze_integration(df_merged, sentiment_cols)
        
        # 7. Create visualizations
        self.plot_integration_analysis(df_merged)
        
        # 8. Save results
        if save:
            filename, metadata_file = self.save_integrated_data(df_merged, "ALL_STOCKS", analysis_stats)
        
        logger.info(f"\n✅ Data integration completed!")
        logger.info(f"📊 Scientific validity: All sentiment data is REAL (no forward-filling)")
        return df_merged


def main():
    """
    Main execution function
    """
    
    # Create integrator
    integrator = DataIntegrator()
    
    try:
        df_integrated = integrator.integrate_data()
        
        if df_integrated is not None:
            print(f"\n✅ Integration successful!")
            print(f"📊 Final dataset: {len(df_integrated)} rows × {len(df_integrated.columns)} columns")
            print(f"📅 Date range: {df_integrated.index.min().date()} to {df_integrated.index.max().date()}")
            print(f"📁 Data saved in: data/integrated/")
            print(f"📊 Tickers: {', '.join(sorted(df_integrated['Ticker'].unique()))}")
            
            # Show feature breakdown
            price_features = [col for col in df_integrated.columns if col not in ['Target', 'Ticker'] and 'sentiment' not in col.lower() and 'compound' not in col.lower() and 'positive' not in col.lower() and 'negative' not in col.lower() and 'neutral' not in col.lower() and 'article' not in col.lower() and 'news' not in col.lower()]
            sentiment_features = [col for col in df_integrated.columns if any(word in col.lower() for word in ['sentiment', 'compound', 'positive', 'negative', 'neutral', 'article', 'news'])]
            
            print(f"\n📊 Feature breakdown:")
            print(f"   Price/Technical features: {len(price_features)}")
            print(f"   Sentiment features: {len(sentiment_features)}")
            print(f"   Target: 1")
            print(f"   Ticker: 1")
            print(f"   Total: {len(df_integrated.columns)}")
            
            # Show sample data
            print(f"\n📋 Sample of integrated dataset:")
            key_cols = ['Ticker', 'compound_score_mean', 'positive_ratio', 'negative_ratio', 'article_count', 'Target']
            available_cols = [col for col in key_cols if col in df_integrated.columns]
            print(df_integrated[available_cols].head(10))
            
            print(f"\n🚀 Ready for LSTM+FinBERT training!")
            print(f"📊 Dataset combines technical indicators with sentiment analysis")
        else:
            print("❌ Integration failed")
            
    except Exception as e:
        print(f"❌ Error during integration: {str(e)}")


if __name__ == "__main__":
    main()