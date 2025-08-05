"""
News Preprocessing Script for FinBERT Sentiment Analysis
======================================================

Purpose: Process and consolidate news articles from data/news and data/sentiment 
directories to prepare them for FinBERT sentiment analysis.

Features:
- Consolidate all news files from multiple timestamps
- Remove duplicates based on title and URL
- Clean and standardize text content
- Filter for financial relevance
- Prepare text in FinBERT-compatible format
- Export clean dataset for sentiment analysis

Usage: python scripts/prepare_news_for_finbert.py
"""

import pandas as pd
import json
import os
import re
import logging
from datetime import datetime
from pathlib import Path
import hashlib
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsPreprocessor:
    """
    Consolidate and preprocess news articles for FinBERT sentiment analysis
    """
    
    def __init__(self, news_dir: str = "data/news", sentiment_dir: str = "data/sentiment"):
        self.news_dir = Path(news_dir)
        self.sentiment_dir = Path(sentiment_dir)
        self.output_dir = Path("data/finbert")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 News directory: {self.news_dir}")
        logger.info(f"📁 Sentiment directory: {self.sentiment_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")
    
    def load_all_news_files(self) -> pd.DataFrame:
        """
        Load and consolidate all news CSV files from the news directory
        """
        logger.info("📚 Loading all news files...")
        
        all_dataframes = []
        news_files = list(self.news_dir.glob("*_news_*.csv"))
        
        if not news_files:
            logger.warning("⚠️  No news CSV files found!")
            return pd.DataFrame()
        
        logger.info(f"Found {len(news_files)} news files")
        
        for file_path in news_files:
            try:
                logger.info(f"📖 Loading {file_path.name}")
                df = pd.read_csv(file_path)
                
                # Extract stock symbol from filename
                symbol_match = re.search(r'([A-Z]+)_news_', file_path.name)
                if symbol_match:
                    df['stock_symbol'] = symbol_match.group(1)
                else:
                    df['stock_symbol'] = 'UNKNOWN'
                
                # Add file metadata
                df['source_file'] = file_path.name
                df['collection_timestamp'] = self._extract_timestamp_from_filename(file_path.name)
                
                all_dataframes.append(df)
                logger.info(f"✅ Loaded {len(df)} articles from {file_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Error loading {file_path}: {str(e)}")
                continue
        
        if not all_dataframes:
            logger.error("❌ No valid news files could be loaded!")
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        logger.info(f"📊 Combined total: {len(combined_df)} articles")
        
        return combined_df
    
    def _extract_timestamp_from_filename(self, filename: str) -> str:
        """Extract timestamp from filename pattern"""
        timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            try:
                dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                return dt.isoformat()
            except ValueError:
                pass
        return ""
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and extract the main text content
        """
        logger.info("🔧 Standardizing columns and extracting text...")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Initialize standardized columns
        df['title_clean'] = ''
        df['text_content'] = ''
        df['published_date'] = ''
        df['article_url'] = ''
        df['news_source'] = ''
        
        # Standardize title
        if 'title' in df.columns:
            df['title_clean'] = df['title'].fillna('')
        
        # Standardize URL
        if 'url' in df.columns:
            df['article_url'] = df['url'].fillna('')
        
        # Standardize source information
        if 'source' in df.columns:
            df['news_source'] = df['source'].fillna('')
        elif 'source_name' in df.columns:
            df['news_source'] = df['source_name'].fillna('')
        
        # Extract main text content (priority order)
        text_columns = ['summary', 'description', 'content', 'snippet']
        for col in text_columns:
            if col in df.columns:
                mask = (df['text_content'] == '') & (df[col].notna())
                df.loc[mask, 'text_content'] = df.loc[mask, col]
        
        # If no text content, use title
        mask = df['text_content'] == ''
        df.loc[mask, 'text_content'] = df.loc[mask, 'title_clean']
        
        # Standardize published date (try multiple formats)
        date_columns = ['time_published', 'publishedAt', 'published_at', 'datetime']
        for col in date_columns:
            if col in df.columns:
                mask = (df['published_date'] == '') & (df[col].notna())
                df.loc[mask, 'published_date'] = df.loc[mask, col].astype(str)
        
        logger.info(f"✅ Standardized {len(df)} articles")
        return df
    
    def clean_text_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize text content for FinBERT
        """
        logger.info("🧹 Cleaning text content...")
        
        df = df.copy()
        
        def clean_text(text: str) -> str:
            if pd.isna(text) or text == '':
                return ''
            
            # Convert to string and strip
            text = str(text).strip()
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove common artifacts
            text = re.sub(r'\[.*?\]', '', text)  # Remove [brackets]
            text = re.sub(r'\(.*?\)', ' ', text)  # Remove (parentheses) but keep space

            # Clean up special characters but keep sentence structure
            text = re.sub(r'[^\w\s\.\!\?\,\:\;\-\']', ' ', text)
            
            # Remove extra dots
            text = re.sub(r'\.{2,}', '.', text)
            
            # Clean up extra spaces again
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        # Clean title and text content
        df['title_clean'] = df['title_clean'].apply(clean_text)
        df['text_content'] = df['text_content'].apply(clean_text)
        
        # Create combined text for FinBERT (title + text)
        df['finbert_text'] = df.apply(
            lambda row: f"{row['title_clean']}. {row['text_content']}" 
            if row['text_content'] and row['text_content'] != row['title_clean'] 
            else row['title_clean'], 
            axis=1
        )
        
        # Remove rows with empty text
        initial_count = len(df)
        df = df[df['finbert_text'].str.strip() != ''].copy()
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.info(f"🗑️  Removed {removed_count} articles with empty text")
        
        logger.info(f"✅ Cleaned text for {len(df)} articles")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate articles based on title similarity and URL
        """
        logger.info("🔍 Removing duplicates...")
        
        initial_count = len(df)
        
        # Create hash for duplicate detection
        def create_content_hash(row):
            # Combine title and URL for uniqueness
            content = f"{row['title_clean']}{row['article_url']}"
            return hashlib.md5(content.encode()).hexdigest()
        
        df['content_hash'] = df.apply(create_content_hash, axis=1)
        
        # Remove exact duplicates
        df = df.drop_duplicates(subset=['content_hash'], keep='first')
        
        # Remove duplicates based on title similarity (after cleaning)
        df = df.drop_duplicates(subset=['title_clean'], keep='first')
        
        # Sort by published date (most recent first) and remove hash column
        df = df.sort_values('published_date', ascending=False).drop('content_hash', axis=1)
        
        duplicate_count = initial_count - len(df)
        logger.info(f"🧹 Removed {duplicate_count} duplicates ({len(df)} unique articles remaining)")
        
        return df
    
    def filter_financial_relevance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter articles for financial relevance
        """
        logger.info("💰 Filtering for financial relevance...")
        
        financial_keywords = [
            'stock', 'shares', 'earnings', 'revenue', 'profit', 'loss',
            'financial', 'market', 'trading', 'investor', 'investment',
            'quarterly', 'annual', 'analyst', 'price', 'valuation',
            'dividend', 'SEC', 'IPO', 'merger', 'acquisition', 'buyback',
            'guidance', 'outlook', 'forecast', 'beat', 'miss', 'eps',
            'growth', 'decline', 'rise', 'fall', 'bull', 'bear'
        ]
        
        # Create search text (title + content)
        search_text = (df['title_clean'] + ' ' + df['text_content']).str.lower()
        
        # Filter for financial relevance
        financial_filter = search_text.str.contains('|'.join(financial_keywords), case=False, na=False)
        
        initial_count = len(df)
        df_filtered = df[financial_filter].copy()
        filtered_count = initial_count - len(df_filtered)
        
        logger.info(f"📊 Filtered out {filtered_count} non-financial articles")
        logger.info(f"✅ {len(df_filtered)} financially relevant articles remaining")
        
        return df_filtered
    
    def prepare_finbert_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare final dataset for FinBERT sentiment analysis
        """
        logger.info("🎯 Preparing FinBERT dataset...")
        
        # Select and order columns for FinBERT
        finbert_columns = [
            'stock_symbol',
            'published_date', 
            'title_clean',
            'finbert_text',
            'article_url',
            'news_source',
            'source_file',
            'collection_timestamp'
        ]
        
        # Ensure all columns exist
        for col in finbert_columns:
            if col not in df.columns:
                df[col] = ''
        
        df_finbert = df[finbert_columns].copy()
        
        # Add text length for analysis
        df_finbert['text_length'] = df_finbert['finbert_text'].str.len()
        
        # Filter out articles that are too short (less than 20 characters)
        initial_count = len(df_finbert)
        df_finbert = df_finbert[df_finbert['text_length'] >= 20].copy()
        short_removed = initial_count - len(df_finbert)
        
        if short_removed > 0:
            logger.info(f"🗑️  Removed {short_removed} articles with text < 20 characters")
        
        # Sort by stock symbol and date
        df_finbert = df_finbert.sort_values(['stock_symbol', 'published_date'], ascending=[True, False])
        
        logger.info(f"✅ Prepared {len(df_finbert)} articles for FinBERT")
        
        return df_finbert
    
    def generate_summary_stats(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics about the processed dataset
        """
        stats = {
            'total_articles': len(df),
            'unique_stocks': df['stock_symbol'].nunique(),
            'stock_distribution': df['stock_symbol'].value_counts().to_dict(),
            'source_distribution': df['news_source'].value_counts().to_dict(),
            'date_range': {
                'earliest': df['published_date'].min() if len(df) > 0 else None,
                'latest': df['published_date'].max() if len(df) > 0 else None
            },
            'text_length_stats': {
                'mean': df['text_length'].mean() if len(df) > 0 else 0,
                'median': df['text_length'].median() if len(df) > 0 else 0,
                'min': df['text_length'].min() if len(df) > 0 else 0,
                'max': df['text_length'].max() if len(df) > 0 else 0
            },
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return stats
    
    def save_processed_data(self, df: pd.DataFrame, stats: Dict) -> tuple:
        """
        Save processed data and metadata
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save main dataset
        output_file = self.output_dir / f"news_finbert_ready_{timestamp}.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"💾 Saved processed dataset: {output_file}")
        
        # Save metadata
        metadata_file = self.output_dir / f"news_finbert_metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        logger.info(f"💾 Saved metadata: {metadata_file}")
        
        return str(output_file), str(metadata_file)
    
    def process_all_news(self) -> tuple:
        """
        Main processing pipeline
        """
        logger.info("🚀 Starting news preprocessing pipeline...")
        
        # Step 1: Load all news files
        df = self.load_all_news_files()
        if df.empty:
            logger.error("❌ No news data to process!")
            return None, None
        
        # Step 2: Standardize columns and extract text
        df = self.standardize_columns(df)
        
        # Step 3: Clean text content
        df = self.clean_text_content(df)
        
        # Step 4: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 5: Filter for financial relevance
        df = self.filter_financial_relevance(df)
        
        # Step 6: Prepare FinBERT dataset
        df_final = self.prepare_finbert_dataset(df)
        
        # Step 7: Generate statistics
        stats = self.generate_summary_stats(df_final)
        
        # Step 8: Save processed data
        output_file, metadata_file = self.save_processed_data(df_final, stats)
        
        logger.info("✅ News preprocessing completed successfully!")
        logger.info(f"📊 Final dataset: {stats['total_articles']} articles")
        logger.info(f"📈 Stock coverage: {stats['unique_stocks']} stocks")
        
        return output_file, metadata_file


def main():
    """
    Main execution function
    """
    logger.info("📰 News Preprocessing for FinBERT")
    logger.info("=" * 50)
    
    # Create preprocessor
    preprocessor = NewsPreprocessor()
    
    # Process all news
    output_file, metadata_file = preprocessor.process_all_news()
    
    if output_file:
        print(f"\n🎉 Processing Complete!")
        print(f"📁 Processed dataset: {output_file}")
        print(f"📊 Metadata: {metadata_file}")
        print(f"\n🚀 Next step: Run FinBERT sentiment analysis on the processed dataset!")
    else:
        print(f"\n❌ Processing failed - check logs for details")


if __name__ == "__main__":
    main()