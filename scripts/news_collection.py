"""
Financial News Data Collection Script
=====================================

Purpose: Collect financial news from multiple sources for sentiment analysis
- Alpha Vantage News (finance-focused)
- NewsAPI (broader coverage)
- Filter for specific stocks (AAPL)
- Align with stock price dates

Usage: python scripts/news_collection.py
"""

import requests
import pandas as pd
import json
import time
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsCollector:
    """
    Collect financial news from multiple sources
    
    Features:
    - Alpha Vantage news (finance-specific)
    - NewsAPI (general news with financial filtering)
    - Date-based filtering
    - Stock-specific news (AAPL focus)
    """
    
    def __init__(self):
        """Initialize with API keys from environment"""
        self.av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        
        if not self.av_key:
            logger.warning("⚠️  Alpha Vantage API key not found")
        if not self.newsapi_key:
            logger.warning("⚠️  NewsAPI key not found")
        
        # Create directories
        os.makedirs('data/news', exist_ok=True)
        
        logger.info("📰 News collector initialized")
    
    def get_alpha_vantage_news(self, symbol='AAPL', limit=50):
        """
        Get news from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            limit: Number of articles (max 1000)
        """
        if not self.av_key:
            logger.error("❌ Alpha Vantage API key required")
            return None
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'limit': limit,
            'apikey': self.av_key
        }
        
        logger.info(f"📊 Fetching Alpha Vantage news for {symbol}...")
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for errors
            if 'Error Message' in data:
                logger.error(f"❌ Alpha Vantage Error: {data['Error Message']}")
                return None
            
            if 'Information' in data:
                logger.warning(f"⚠️  Alpha Vantage: {data['Information']}")
                return None
            
            # Extract news feed
            if 'feed' not in data:
                logger.error("❌ No news feed found in Alpha Vantage response")
                return None
            
            articles = data['feed']
            logger.info(f"✅ Got {len(articles)} articles from Alpha Vantage")
            
            # Convert to DataFrame
            processed_articles = []
            for article in articles:
                processed_article = {
                    'source': 'alpha_vantage',
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'url': article.get('url', ''),
                    'time_published': article.get('time_published', ''),
                    'authors': ', '.join(article.get('authors', [])),
                    'topics': ', '.join([topic.get('topic', '') for topic in article.get('topics', [])]),
                    'overall_sentiment_score': article.get('overall_sentiment_score', 0),
                    'overall_sentiment_label': article.get('overall_sentiment_label', ''),
                    'ticker_sentiment': json.dumps(article.get('ticker_sentiment', []))
                }
                processed_articles.append(processed_article)
            
            return pd.DataFrame(processed_articles)
            
        except Exception as e:
            logger.error(f"❌ Error fetching Alpha Vantage news: {str(e)}")
            return None
    
    def get_newsapi_articles(self, query='Apple AND stock', days_back=30, language='en'):
        """
        Get news from NewsAPI
        
        Args:
            query: Search query (focus on financial terms)
            days_back: How many days back to search
            language: Article language
        """
        if not self.newsapi_key:
            logger.error("❌ NewsAPI key required")
            return None
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'language': language,
            'sortBy': 'publishedAt',
            'pageSize': 100,  # Max per request
            'apiKey': self.newsapi_key
        }
        
        logger.info(f"📊 Fetching NewsAPI articles: '{query}' ({days_back} days back)")
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] != 'ok':
                logger.error(f"❌ NewsAPI Error: {data.get('message', 'Unknown error')}")
                return None
            
            articles = data['articles']
            logger.info(f"✅ Got {len(articles)} articles from NewsAPI")
            
            # Convert to DataFrame
            processed_articles = []
            for article in articles:
                processed_article = {
                    'source': 'newsapi',
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'publishedAt': article.get('publishedAt', ''),
                    'source_name': article.get('source', {}).get('name', ''),
                    'author': article.get('author', '')
                }
                processed_articles.append(processed_article)
            
            return pd.DataFrame(processed_articles)
            
        except Exception as e:
            logger.error(f"❌ Error fetching NewsAPI articles: {str(e)}")
            return None
    
    def filter_financial_news(self, df, financial_keywords=None):
        """
        Filter articles for financial relevance
        
        Args:
            df: DataFrame of articles
            financial_keywords: List of keywords to look for
        """
        if financial_keywords is None:
            financial_keywords = [
                'stock', 'earnings', 'revenue', 'profit', 'shares', 'market',
                'investor', 'trading', 'financial', 'quarterly', 'analyst',
                'price', 'valuation', 'dividend', 'SEC', 'IPO', 'merger'
            ]
        
        logger.info("🔍 Filtering for financial relevance...")
        
        initial_count = len(df)
        
        # Create text to search (combine title and description/summary)
        if 'description' in df.columns:
            search_text = df['title'].fillna('') + ' ' + df['description'].fillna('')
        elif 'summary' in df.columns:
            search_text = df['title'].fillna('') + ' ' + df['summary'].fillna('')
        else:
            search_text = df['title'].fillna('')
        
        # Convert to lowercase for case-insensitive search
        search_text = search_text.str.lower()
        
        # Create filter for financial keywords
        financial_filter = search_text.str.contains('|'.join(financial_keywords), case=False, na=False)
        
        # Apply filter
        filtered_df = df[financial_filter].copy()
        
        logger.info(f"📊 Filtered: {len(filtered_df)}/{initial_count} articles are financial")
        
        return filtered_df
    
    def collect_all_news(self, symbol='AAPL', days_back=30, save=True):
        """
        Collect news from all sources for a specific stock
        
        Args:
            symbol: Stock symbol
            days_back: Days of history to collect
            save: Whether to save to files
        """
        logger.info(f"\n🚀 Collecting news for {symbol} ({days_back} days back)")
        
        all_news = []
        
        # 1. Alpha Vantage News
        if self.av_key:
            av_news = self.get_alpha_vantage_news(symbol)
            if av_news is not None and len(av_news) > 0:
                all_news.append(av_news)
                logger.info(f"📰 Alpha Vantage: {len(av_news)} articles")
            
            # Rate limiting
            time.sleep(12)  # Alpha Vantage free tier limit
        
        # 2. NewsAPI - Multiple queries for better coverage
        if self.newsapi_key:
            queries = [
                f'{symbol} stock',
                f'Apple AND (earnings OR revenue OR financial)',
                f'Apple AND (stock OR shares OR market)'
            ]
            
            for query in queries:
                newsapi_articles = self.get_newsapi_articles(query, days_back)
                if newsapi_articles is not None and len(newsapi_articles) > 0:
                    # Filter for financial relevance
                    filtered_articles = self.filter_financial_news(newsapi_articles)
                    if len(filtered_articles) > 0:
                        all_news.append(filtered_articles)
                        logger.info(f"📰 NewsAPI ('{query}'): {len(filtered_articles)} financial articles")
                
                # Rate limiting for NewsAPI
                time.sleep(1)
        
        if not all_news:
            logger.error("❌ No news collected from any source")
            return None
        
        # 3. Combine all news
        combined_news = pd.concat(all_news, ignore_index=True)
        
        # 4. Remove duplicates based on title
        initial_count = len(combined_news)
        combined_news = combined_news.drop_duplicates(subset=['title'], keep='first')
        duplicate_count = initial_count - len(combined_news)
        
        logger.info(f"🧹 Removed {duplicate_count} duplicates")
        logger.info(f"📊 Final collection: {len(combined_news)} unique articles")
        
        # 5. Save if requested
        if save:
            self.save_news_data(combined_news, symbol)
        
        return combined_news
    
    def save_news_data(self, df, symbol):
        """Save news data with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main data
        filename = f"data/news/{symbol}_news_{timestamp}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"💾 Saved news data: {filename}")
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'collection_date': datetime.now().isoformat(),
            'total_articles': len(df),
            'sources': df['source'].value_counts().to_dict() if 'source' in df.columns else {},
            'date_range': 'Mixed sources with different date formats'
        }
        
        metadata_file = f"data/news/{symbol}_news_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        return filename, metadata_file


def main():
    """
    Main execution function
    """
    
    # Create collector
    collector = NewsCollector()
    
    # Check API keys
    if not collector.av_key and not collector.newsapi_key:
        print("❌ No API keys found! Please set up your .env file:")
        print("ALPHA_VANTAGE_API_KEY=your_key")
        print("NEWSAPI_KEY=your_key")
        return
    
    # Collect news for AAPL
    symbol = 'AAPL'
    days_back = 30  # Last 30 days
    
    logger.info(f"🚀 Starting news collection for {symbol}")
    news_df = collector.collect_all_news(symbol, days_back)
    
    if news_df is not None:
        print(f"\n✅ News collection successful!")
        print(f"📊 Collected {len(news_df)} articles")
        print(f"📁 Data saved in: data/news/")
        
        # Show sample of collected news
        print(f"\n📋 Sample articles:")
        for i, row in news_df.head(3).iterrows():
            print(f"   {i+1}. {row['title'][:80]}...")
        
        print(f"\n🚀 Next step: Sentiment analysis with FinBERT!")
    else:
        print("❌ News collection failed")


if __name__ == "__main__":
    main()