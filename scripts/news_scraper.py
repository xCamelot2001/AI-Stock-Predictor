import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime, timedelta
import yfinance as yf
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, quote_plus
import logging
from pathlib import Path
import feedparser
import json

class StockNewsScraper:
    def __init__(self, stock_symbols, delay=2, months_back=3):
        """
        Legal news scraper for specific stocks
        
        Args:
            stock_symbols (list): List of stock symbols (e.g., ['AAPL', 'MSFT'])
            delay (int): Delay between requests (seconds)
            months_back (int): How many months back to get news (default: 3)
        """
        self.stock_symbols = stock_symbols
        self.delay = delay
        self.months_back = months_back
        self.start_date = datetime.now() - timedelta(days=months_back * 30)
        self.end_date = datetime.now()
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Academic Research Bot; University of Southampton)'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Get company names for better search
        self.stock_info = self.get_company_names()
        
        self.logger.info(f"Searching for news from {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
    
    # UTILITY METHODS
    def get_company_names(self):
        """Get company names for stock symbols"""
        stock_info = {}
        for symbol in self.stock_symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                stock_info[symbol] = {
                    'name': info.get('longName', symbol),
                    'short_name': info.get('shortName', symbol)
                }
            except:
                stock_info[symbol] = {'name': symbol, 'short_name': symbol}
        return stock_info
    
    def check_robots_txt(self, base_url):
        """Check if scraping is allowed"""
        try:
            rp = RobotFileParser()
            rp.set_url(urljoin(base_url, '/robots.txt'))
            rp.read()
            return rp.can_fetch('*', base_url)
        except:
            return True
    
    def is_within_date_range(self, article_date):
        """Check if article date is within our target range"""
        if not article_date:
            return True
            
        try:
            if isinstance(article_date, str):
                for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%a, %d %b %Y %H:%M:%S %Z']:
                    try:
                        date_obj = datetime.strptime(article_date[:19], fmt[:19])
                        break
                    except:
                        continue
                else:
                    date_obj = pd.to_datetime(article_date, errors='coerce')
            else:
                date_obj = article_date
                
            if pd.isna(date_obj):
                return True
                
            return self.start_date <= date_obj <= self.end_date
        except:
            return True
    
    # SCRAPING METHODS
    def scrape_yahoo_finance_historical(self, symbol, max_articles=200):
        """Scrape Yahoo Finance with date-based pagination for historical data"""
        articles = []
        
        if not self.check_robots_txt('https://finance.yahoo.com'):
            self.logger.warning("Yahoo Finance robots.txt restricts scraping")
            return articles
            
        try:
            current_date = self.end_date
            
            while current_date >= self.start_date and len(articles) < max_articles:
                timestamp = int(current_date.timestamp())
                url = f'https://finance.yahoo.com/quote/{symbol}/news?offset=0&size=25&tsrc=fin-srch&period1={int(self.start_date.timestamp())}&period2={timestamp}'
                
                response = self.session.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for news items
                news_containers = soup.find_all(['h3', 'div'], class_=lambda x: x and ('news' in x.lower() or 'story' in x.lower()))
                
                page_articles = 0
                for container in news_containers:
                    link_elem = container.find('a') if container.name != 'a' else container
                    if link_elem and link_elem.get('href'):
                        title = link_elem.get_text().strip()
                        href = link_elem.get('href')
                        
                        if href.startswith('/'):
                            href = 'https://finance.yahoo.com' + href
                        
                        # Try to extract date from nearby elements
                        date_elem = container.find_next('time') or container.find_previous('time')
                        article_date = date_elem.get('datetime') if date_elem else current_date.isoformat()
                        
                        if self.is_within_date_range(article_date):
                            articles.append({
                                'symbol': symbol,
                                'title': title,
                                'url': href,
                                'published': article_date,
                                'source': 'Yahoo Finance',
                                'date_scraped': datetime.now().isoformat()
                            })
                            page_articles += 1
                
                if page_articles == 0:
                    current_date -= timedelta(days=7)
                else:
                    current_date -= timedelta(days=1)
                
                time.sleep(self.delay)
                
        except Exception as e:
            self.logger.error(f"Error scraping Yahoo historical for {symbol}: {e}")
            
        return articles
    
    def scrape_google_news_historical(self, symbol, max_articles=100):
        """Scrape Google News for historical financial news"""
        articles = []
        
        try:
            company_name = self.stock_info[symbol]['name']
            
            # Format dates for Google News
            start_str = self.start_date.strftime('%m/%d/%Y')
            end_str = self.end_date.strftime('%m/%d/%Y')
            
            # Google News search with date range
            query = f'"{symbol}" OR "{company_name}" stock earnings financial'
            encoded_query = quote_plus(query)
            
            search_url = f'https://news.google.com/search?q={encoded_query}%20after%3A{start_str}%20before%3A{end_str}&hl=en-US&gl=US&ceid=US%3Aen'
            
            response = self.session.get(search_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Google News article selectors
            news_items = soup.find_all('article') or soup.find_all('div', {'data-n-tid': True})
            
            for item in news_items[:max_articles]:
                try:
                    title_elem = item.find('h3') or item.find('a')
                    if title_elem:
                        title = title_elem.get_text().strip()
                        
                        # Get link (fixed variable name conflict)
                        link_elem = item.find('a')
                        article_url = link_elem.get('href') if link_elem else ''
                        if article_url.startswith('./'):
                            article_url = 'https://news.google.com' + article_url[1:]
                        
                        # Try to find date
                        date_elem = item.find('time') 
                        article_date = date_elem.get('datetime') if date_elem else None
                        
                        if self.is_within_date_range(article_date):
                            articles.append({
                                'symbol': symbol,
                                'title': title,
                                'url': article_url,
                                'published': article_date,
                                'source': 'Google News',
                                'date_scraped': datetime.now().isoformat()
                            })
                except:
                    continue
                    
            time.sleep(self.delay)
            
        except Exception as e:
            self.logger.error(f"Error scraping Google News for {symbol}: {e}")
            
        return articles
    
    def scrape_marketwatch_news(self, symbol, max_articles=30):
        """Scrape MarketWatch news"""
        articles = []
        
        try:
            search_query = f"{symbol} {self.stock_info[symbol]['name']}"
            url = f'https://www.marketwatch.com/search?q={quote_plus(search_query)}&m=Keyword&rpp=25&mp=2005&bd=false&rs=true'
            
            response = self.session.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            news_items = soup.find_all('div', class_='searchresult')
            
            for item in news_items[:max_articles]:
                try:
                    title_elem = item.find('a')
                    if title_elem:
                        title = title_elem.get_text().strip()
                        url = title_elem.get('href')
                        
                        date_elem = item.find('span', class_='datestamp')
                        date = date_elem.get_text().strip() if date_elem else None
                        
                        articles.append({
                            'symbol': symbol,
                            'title': title,
                            'url': url,
                            'source': 'MarketWatch',
                            'published': date,
                            'date_scraped': datetime.now().isoformat()
                        })
                except:
                    continue
                    
            time.sleep(self.delay)
            
        except Exception as e:
            self.logger.error(f"Error scraping MarketWatch for {symbol}: {e}")
            
        return articles
    
    def scrape_reuters_rss(self, symbol):
        """Use Reuters RSS feeds (legal and efficient)"""
        articles = []
        
        try:
            rss_urls = [
                'https://feeds.reuters.com/reuters/businessNews',
                'https://feeds.reuters.com/reuters/companyNews'
            ]
            
            for rss_url in rss_urls:
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries:
                    title = entry.title
                    if (symbol.lower() in title.lower() or 
                        self.stock_info[symbol]['name'].lower() in title.lower() or
                        self.stock_info[symbol]['short_name'].lower() in title.lower()):
                        
                        articles.append({
                            'symbol': symbol,
                            'title': title,
                            'url': entry.link,
                            'summary': entry.get('summary', ''),
                            'published': entry.get('published', ''),
                            'source': 'Reuters RSS',
                            'date_scraped': datetime.now().isoformat()
                        })
                        
            time.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Error with Reuters RSS for {symbol}: {e}")
            
        return articles
    
    def scrape_seeking_alpha_rss(self, symbol):
        """Scrape Seeking Alpha RSS (allows RSS access)"""
        articles = []
        
        try:
            rss_url = f'https://seekingalpha.com/api/sa/combined/{symbol}.xml'
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries:
                articles.append({
                    'symbol': symbol,
                    'title': entry.title,
                    'url': entry.link,
                    'summary': entry.get('summary', ''),
                    'published': entry.get('published', ''),
                    'source': 'Seeking Alpha RSS',
                    'date_scraped': datetime.now().isoformat()
                })
                
            time.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Error with Seeking Alpha RSS for {symbol}: {e}")
            
        return articles
    
    def scrape_financial_times_news(self, symbol, max_articles=25):
        """Scrape Financial Times news (RSS and search)"""
        articles = []
        
        try:
            # FT RSS feeds
            ft_rss_feeds = [
                'https://www.ft.com/companies?format=rss',
                'https://www.ft.com/markets?format=rss',
                'https://www.ft.com/equities?format=rss'
            ]
            
            company_name = self.stock_info[symbol]['name'].lower()
            search_terms = [symbol.lower(), company_name]
            
            # Check RSS feeds first
            for rss_url in ft_rss_feeds:
                try:
                    feed = feedparser.parse(rss_url)
                    
                    for entry in feed.entries:
                        title = entry.title.lower()
                        summary = entry.get('summary', '').lower()
                        
                        if any(term in title or term in summary for term in search_terms):
                            articles.append({
                                'symbol': symbol,
                                'title': entry.title,
                                'url': entry.link,
                                'summary': entry.get('summary', ''),
                                'published': entry.get('published', ''),
                                'source': 'Financial Times RSS',
                                'date_scraped': datetime.now().isoformat()
                            })
                            
                    time.sleep(1)
                except:
                    continue
            
            # Search if RSS insufficient
            if len(articles) < 5 and self.check_robots_txt('https://www.ft.com'):
                try:
                    search_query = f"{symbol} {self.stock_info[symbol]['name']}"
                    search_url = f'https://www.ft.com/search?q={quote_plus(search_query)}'
                    
                    response = self.session.get(search_url)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    search_results = soup.find_all('div', class_='o-teaser__content') or \
                                   soup.find_all('div', {'data-trackable': 'story-package'})
                    
                    for result in search_results[:max_articles-len(articles)]:
                        try:
                            title_elem = result.find('a') or result.find('h3')
                            if title_elem:
                                title = title_elem.get_text().strip()
                                link = title_elem.get('href', '')
                                
                                if link and not link.startswith('http'):
                                    link = 'https://www.ft.com' + link
                                
                                date_elem = result.find('time') or result.find('span', class_='o-date')
                                date = date_elem.get('datetime') if date_elem else None
                                
                                articles.append({
                                    'symbol': symbol,
                                    'title': title,
                                    'url': link,
                                    'published': date,
                                    'source': 'Financial Times Search',
                                    'date_scraped': datetime.now().isoformat()
                                })
                        except:
                            continue
                            
                    time.sleep(self.delay)
                    
                except Exception as e:
                    self.logger.error(f"Error scraping FT search for {symbol}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error with Financial Times for {symbol}: {e}")
            
        return articles
    
    def get_article_content(self, url, max_retries=3):
        """Extract full article content from URL"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                content_selectors = [
                    'div.caas-body',  # Yahoo
                    'div.article__content',  # Reuters
                    'div.paywall-article-content',  # MarketWatch
                    'article',
                    'div[data-module="ArticleBody"]',
                    '.article-wrap',
                    '.entry-content',
                    '[role="main"]'
                ]
                
                content = ""
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        content = ' '.join([elem.get_text().strip() for elem in elements])
                        break
                
                if not content:
                    paragraphs = soup.find_all('p')
                    content = ' '.join([p.get_text().strip() for p in paragraphs])
                
                content = re.sub(r'\s+', ' ', content).strip()
                time.sleep(self.delay)
                
                return content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to get content from {url}: {e}")
                time.sleep(self.delay * (attempt + 1))
                
        return ""
    
    # MAIN METHODS
    def scrape_all_sources(self, include_content=True):
        """Scrape all available sources for all stocks"""
        all_articles = []
        
        for symbol in self.stock_symbols:
            self.logger.info(f"Scraping {self.months_back} months of news for {symbol} ({self.stock_info[symbol]['name']})")
            
            # Scrape all sources
            yahoo_articles = self.scrape_yahoo_finance_historical(symbol)
            google_articles = self.scrape_google_news_historical(symbol)
            marketwatch_articles = self.scrape_marketwatch_news(symbol)
            reuters_articles = self.scrape_reuters_rss(symbol)
            seeking_alpha_articles = self.scrape_seeking_alpha_rss(symbol)
            ft_articles = self.scrape_financial_times_news(symbol)
            
            symbol_articles = yahoo_articles + google_articles + marketwatch_articles + reuters_articles + seeking_alpha_articles + ft_articles
            
            # Filter by date range
            symbol_articles = [article for article in symbol_articles 
                             if self.is_within_date_range(article.get('published'))]
            
            # Get full content if requested
            if include_content:
                for article in symbol_articles:
                    if article.get('url'):
                        content = self.get_article_content(article['url'])
                        article['content'] = content
                        article['content_length'] = len(content)
            
            all_articles.extend(symbol_articles)
            self.logger.info(f"Found {len(symbol_articles)} articles for {symbol}")
        
        return all_articles
    
    def filter_financial_content(self, articles):
        """Filter articles for financial relevance"""
        financial_keywords = [
            'stock', 'share', 'earnings', 'revenue', 'profit', 'loss',
            'quarterly', 'fiscal', 'dividend', 'analyst', 'rating',
            'upgrade', 'downgrade', 'target price', 'valuation',
            'merger', 'acquisition', 'ipo', 'market cap', 'trading',
            'financial', 'investment', 'investor', 'market', 'price',
            'growth', 'sales', 'income', 'report', 'forecast'
        ]
        
        filtered_articles = []
        for article in articles:
            text = f"{article.get('title', '')} {article.get('content', '')} {article.get('summary', '')}".lower()
            
            keyword_count = sum(1 for keyword in financial_keywords if keyword in text)
            
            # Lower threshold and check for stock symbol
            if keyword_count >= 1 or article.get('symbol', '').lower() in text:
                article['financial_relevance_score'] = keyword_count
                filtered_articles.append(article)
        
        return filtered_articles
    
    def save_to_finbert_format(self, articles, output_file='scraped_stock_news.csv'):
        """Save articles in FinBERT-ready format"""
        if not articles:
            self.logger.warning("No articles to save")
            return
        
        df = pd.DataFrame(articles)
        df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
        
        # Create FinBERT input format
        finbert_data = []
        for _, row in df.iterrows():
            # Fix date extraction
            date_val = row.get('published') or row.get('date_scraped', '')
            if not date_val:
                date_val = datetime.now().strftime('%Y-%m-%d')
            
            # Use title as primary text
            finbert_data.append({
                'date': date_val,
                'symbol': row['symbol'],
                'text': row['title'],
                'content': row.get('content', ''),
                'source': row['source'],
                'url': row.get('url', '')
            })
            
            # Add content if substantial
            content = row.get('content', '')
            if content and len(content.split()) > 20:
                finbert_data.append({
                    'date': date_val,
                    'symbol': row['symbol'],
                    'text': content[:500],
                    'content': content,
                    'source': row['source'] + ' (Content)',
                    'url': row.get('url', '')
                })
        
        finbert_df = pd.DataFrame(finbert_data)
        
        # Fix date parsing with explicit format handling
        def parse_date_safely(date_str):
            # print(f"DEBUG: Parsing date: '{date_str}' (type: {type(date_str)})")

            if pd.isna(date_str) or not date_str:
                return datetime.now().strftime('%Y-%m-%d')
            
            try:
                # Try common formats
                for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%a, %d %b %Y %H:%M:%S %Z', '%a, %d %b %Y']:
                    try:
                        return datetime.strptime(str(date_str)[:19], fmt[:19]).strftime('%Y-%m-%d')
                    except:
                        continue
                
                # Fallback to pandas
                parsed = pd.to_datetime(date_str, errors='coerce')
                if pd.notna(parsed):
                    return parsed.strftime('%Y-%m-%d')
                else:
                    return datetime.now().strftime('%Y-%m-%d')
            except:
                return datetime.now().strftime('%Y-%m-%d')
        
        finbert_df['date'] = finbert_df['date'].apply(parse_date_safely)
        finbert_df = finbert_df.sort_values('date').reset_index(drop=True)
        
        # Save both formats
        df.to_csv(output_file, index=False)
        finbert_output = output_file.replace('.csv', '_finbert_ready.csv')
        finbert_df.to_csv(finbert_output, index=False)
        
        self.logger.info(f"Saved {len(df)} articles to {output_file}")
        self.logger.info(f"FinBERT-ready format: {finbert_output} ({len(finbert_df)} entries)")
        
        return finbert_df

# Usage example
if __name__ == "__main__":
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']

    scraper = StockNewsScraper(stock_symbols=stocks, delay=1, months_back=3)

    print("Starting news scraping...")
    articles = scraper.scrape_all_sources(include_content=True)
    
    filtered_articles = scraper.filter_financial_content(articles)
    
    finbert_df = scraper.save_to_finbert_format(filtered_articles)
    
    print(f"\nCompleted! Found {len(filtered_articles)} relevant financial articles")
    print(f"FinBERT-ready dataset has {len(finbert_df)} text entries")
    print("\nSample entries:")
    print(finbert_df[['date', 'symbol', 'text']].head())