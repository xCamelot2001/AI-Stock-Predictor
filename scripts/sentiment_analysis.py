"""
FinBERT Sentiment Analysis Script
=================================

Purpose: Analyze sentiment of collected news articles using FinBERT
- Load news data from news collection
- Use pre-trained FinBERT model for financial sentiment analysis
- Extract sentiment scores (positive/negative/neutral)
- Aggregate sentiment by date for LSTM integration

Usage: python scripts/sentiment_analysis.py
"""

import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# HuggingFace transformers for FinBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class FinBERTSentimentAnalyzer:
    """
    FinBERT-based sentiment analysis for financial news
    
    Features:
    - Pre-trained FinBERT model (finance-specific)
    - Batch processing for efficiency
    - Daily sentiment aggregation
    - Multiple sentiment metrics
    """
    
    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initialize FinBERT sentiment analyzer
        
        Args:
            model_name: HuggingFace model identifier for FinBERT
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs('data/sentiment', exist_ok=True)
        
        logger.info(f"🤖 Initializing FinBERT sentiment analyzer")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"📦 Model: {model_name}")
        
        # Load FinBERT model and tokenizer
        self.load_finbert_model()
    
    def load_finbert_model(self):
        """Load pre-trained FinBERT model"""
        try:
            logger.info("📥 Loading FinBERT model...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create sentiment analysis pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == 'cuda' else -1,
                return_all_scores=True
            )
            
            logger.info("✅ FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading FinBERT model: {str(e)}")
            raise
    
    def load_news_data(self, symbol='AAPL'):
        """Load the latest news data"""
        
        # Find latest news file
        pattern = f"data/news/{symbol}_news_*.csv"
        files = glob.glob(pattern)
        
        if not files:
            logger.error(f"❌ No news data found for {symbol}")
            return None
        
        latest_file = max(files, key=os.path.getctime)
        logger.info(f"📊 Loading news data: {latest_file}")
        
        # Load data
        df = pd.read_csv(latest_file)
        
        logger.info(f"✅ Loaded {len(df)} news articles")
        logger.info(f"📊 Sources: {df['source'].value_counts().to_dict()}")
        
        return df
    
    def prepare_text_for_sentiment(self, df):
        """
        Prepare text for sentiment analysis
        
        Combines title and description/summary for better context
        """
        logger.info("🔧 Preparing text for sentiment analysis...")
        
        texts = []
        
        for _, row in df.iterrows():
            # Combine title with description/summary for more context
            title = str(row.get('title', ''))
            
            # Get description/summary based on source
            if 'description' in row and pd.notna(row['description']):
                content = str(row['description'])
            elif 'summary' in row and pd.notna(row['summary']):
                content = str(row['summary'])
            else:
                content = ''
            
            # Combine title and content
            full_text = f"{title}. {content}".strip()
            
            # Truncate to avoid model limits (FinBERT max ~512 tokens)
            if len(full_text) > 1000:  # Conservative limit
                full_text = full_text[:1000] + "..."
            
            texts.append(full_text)
        
        logger.info(f"✅ Prepared {len(texts)} texts for analysis")
        return texts
    
    def analyze_sentiment_batch(self, texts, batch_size=16):
        """
        Analyze sentiment in batches for efficiency
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
        """
        logger.info(f"🔍 Analyzing sentiment for {len(texts)} texts...")
        
        all_results = []
        
        # Process in batches with progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch = texts[i:i + batch_size]
            
            try:
                # Get sentiment predictions
                batch_results = self.sentiment_pipeline(batch)
                all_results.extend(batch_results)
                
            except Exception as e:
                logger.warning(f"⚠️  Error processing batch {i//batch_size + 1}: {str(e)}")
                # Add dummy results for failed batch
                dummy_results = [[{'label': 'neutral', 'score': 0.33} for _ in range(3)] for _ in batch]
                all_results.extend(dummy_results)
        
        logger.info("✅ Sentiment analysis completed")
        return all_results
    
    def process_sentiment_results(self, sentiment_results):
        """
        Process raw sentiment results into structured format
        
        FinBERT returns: positive, negative, neutral with confidence scores
        """
        logger.info("🔧 Processing sentiment results...")
        
        processed_results = []
        
        for result in sentiment_results:
            # FinBERT returns all scores
            sentiment_dict = {item['label']: item['score'] for item in result}
            
            # Get the dominant sentiment
            dominant_label = max(sentiment_dict, key=sentiment_dict.get)
            dominant_score = sentiment_dict[dominant_label]
            
            # Extract individual scores
            positive_score = sentiment_dict.get('positive', 0)
            negative_score = sentiment_dict.get('negative', 0)
            neutral_score = sentiment_dict.get('neutral', 0)
            
            # Calculate compound sentiment score (-1 to +1)
            compound_score = positive_score - negative_score
            
            processed_result = {
                'sentiment_label': dominant_label,
                'sentiment_confidence': dominant_score,
                'positive_score': positive_score,
                'negative_score': negative_score,
                'neutral_score': neutral_score,
                'compound_score': compound_score
            }
            
            processed_results.append(processed_result)
        
        logger.info("✅ Sentiment results processed")
        return processed_results
    
    def extract_article_dates(self, df):
        """Extract and standardize article dates"""
        logger.info("📅 Extracting article dates...")
        
        dates = []
        
        for _, row in df.iterrows():
            # Try different date columns based on source
            date_str = None
            
            if 'publishedAt' in row and pd.notna(row['publishedAt']):
                date_str = row['publishedAt']
            elif 'time_published' in row and pd.notna(row['time_published']):
                date_str = row['time_published']
            
            if date_str:
                try:
                    # Parse different date formats
                    if 'T' in str(date_str):  # ISO format
                        date = pd.to_datetime(date_str).date()
                    else:  # Alpha Vantage format (YYYYMMDDTHHMMSS)
                        date = pd.to_datetime(str(date_str)[:8], format='%Y%m%d').date()
                    
                    dates.append(date)
                except:
                    dates.append(datetime.now().date())  # Fallback to today
            else:
                dates.append(datetime.now().date())  # Fallback to today
        
        logger.info(f"✅ Extracted dates for {len(dates)} articles")
        return dates
    
    def aggregate_daily_sentiment(self, df_with_sentiment):
        """
        Aggregate sentiment scores by date
        
        Creates daily sentiment metrics for LSTM training
        """
        logger.info("📊 Aggregating sentiment by date...")
        
        # Group by date and calculate aggregations
        daily_sentiment = df_with_sentiment.groupby('date').agg({
            'positive_score': ['mean', 'std', 'count'],
            'negative_score': ['mean', 'std'],
            'neutral_score': ['mean', 'std'],
            'compound_score': ['mean', 'std', 'min', 'max'],
            'sentiment_confidence': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        daily_sentiment.columns = [f"{col[0]}_{col[1]}" for col in daily_sentiment.columns]
        
        # Calculate additional metrics
        daily_sentiment['article_count'] = df_with_sentiment.groupby('date').size()
        daily_sentiment['positive_ratio'] = df_with_sentiment.groupby('date')['sentiment_label'].apply(
            lambda x: (x == 'positive').sum() / len(x)
        ).round(4)
        daily_sentiment['negative_ratio'] = df_with_sentiment.groupby('date')['sentiment_label'].apply(
            lambda x: (x == 'negative').sum() / len(x)
        ).round(4)
        
        # Reset index to make date a column
        daily_sentiment = daily_sentiment.reset_index()
        
        logger.info(f"✅ Created daily sentiment for {len(daily_sentiment)} days")
        
        return daily_sentiment
    
    def analyze_news_sentiment(self, symbol='AAPL', save=True):
        """
        Complete sentiment analysis pipeline
        
        Steps:
        1. Load news data
        2. Prepare texts
        3. Analyze sentiment with FinBERT
        4. Process results
        5. Aggregate by date
        6. Save results
        """
        logger.info(f"\n🚀 Starting sentiment analysis for {symbol}")
        
        # 1. Load news data
        df = self.load_news_data(symbol)
        if df is None:
            return None
        
        # 2. Prepare texts for analysis
        texts = self.prepare_text_for_sentiment(df)
        
        # 3. Analyze sentiment
        sentiment_results = self.analyze_sentiment_batch(texts)
        
        # 4. Process sentiment results
        processed_sentiments = self.process_sentiment_results(sentiment_results)
        
        # 5. Extract dates
        dates = self.extract_article_dates(df)
        
        # 6. Combine with original data
        df_with_sentiment = df.copy()
        df_with_sentiment['date'] = dates
        for key in processed_sentiments[0].keys():
            df_with_sentiment[key] = [result[key] for result in processed_sentiments]
        
        # 7. Aggregate by date
        daily_sentiment = self.aggregate_daily_sentiment(df_with_sentiment)
        
        # 8. Save results
        if save:
            self.save_sentiment_data(df_with_sentiment, daily_sentiment, symbol)
        
        return df_with_sentiment, daily_sentiment
    
    def save_sentiment_data(self, df_articles, df_daily, symbol):
        """Save sentiment analysis results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save article-level sentiment
        articles_file = f"data/sentiment/{symbol}_articles_sentiment_{timestamp}.csv"
        df_articles.to_csv(articles_file, index=False)
        logger.info(f"💾 Saved article sentiment: {articles_file}")
        
        # Save daily aggregated sentiment
        daily_file = f"data/sentiment/{symbol}_daily_sentiment_{timestamp}.csv"
        df_daily.to_csv(daily_file, index=False)
        logger.info(f"💾 Saved daily sentiment: {daily_file}")
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'analysis_date': datetime.now().isoformat(),
            'model_used': self.model_name,
            'total_articles': len(df_articles),
            'unique_dates': len(df_daily),
            'sentiment_distribution': df_articles['sentiment_label'].value_counts().to_dict(),
            'avg_compound_score': float(df_articles['compound_score'].mean()),
            'files': {
                'articles': articles_file,
                'daily': daily_file
            }
        }
        
        metadata_file = f"data/sentiment/{symbol}_sentiment_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        return articles_file, daily_file, metadata_file


def main():
    """
    Main execution function
    """
    
    # Initialize analyzer
    analyzer = FinBERTSentimentAnalyzer()
    
    # Analyze sentiment for AAPL
    symbol = 'AAPL'
    
    try:
        df_articles, df_daily = analyzer.analyze_news_sentiment(symbol)
        
        if df_articles is not None:
            print(f"\n✅ Sentiment analysis completed!")
            print(f"📊 Analyzed {len(df_articles)} articles")
            print(f"📅 Covering {len(df_daily)} unique dates")
            print(f"📁 Results saved in: data/sentiment/")
            
            # Show sentiment distribution
            print(f"\n📊 Sentiment Distribution:")
            sentiment_counts = df_articles['sentiment_label'].value_counts()
            for label, count in sentiment_counts.items():
                print(f"   {label.capitalize()}: {count} ({count/len(df_articles)*100:.1f}%)")
            
            # Show sample daily sentiment
            print(f"\n📋 Sample Daily Sentiment (last 5 days):")
            sample_cols = ['date', 'compound_score_mean', 'positive_ratio', 'negative_ratio', 'article_count']
            print(df_daily[sample_cols].tail())
            
            print(f"\n🚀 Next step: Integrate sentiment with stock price data!")
        else:
            print("❌ Sentiment analysis failed")
            
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {str(e)}")
        print("💡 Make sure you have transformers installed: pip install transformers torch")


if __name__ == "__main__":
    main()