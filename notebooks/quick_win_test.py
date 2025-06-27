import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Real AAPL news events with actual dates and impact
real_aapl_events = [
    # Major earnings and product announcements
    {"date": "2024-11-01", "headline": "Apple reports record Q4 revenue of $94.9 billion, beats expectations", "event_type": "earnings"},
    {"date": "2024-10-28", "headline": "Apple unveils new MacBook Pro with M4 chip", "event_type": "product"},
    {"date": "2024-09-09", "headline": "Apple launches iPhone 16 with advanced AI capabilities", "event_type": "product"},
    {"date": "2024-08-01", "headline": "Apple Q3 earnings show strong services growth", "event_type": "earnings"},
    {"date": "2024-06-10", "headline": "Apple announces iOS 18 with major AI features at WWDC", "event_type": "product"},
    {"date": "2024-05-02", "headline": "Apple reports mixed Q2 earnings, iPhone sales decline", "event_type": "earnings"},
    
    # Regulatory and market concerns
    {"date": "2024-03-25", "headline": "EU fines Apple $2 billion for anti-competitive practices", "event_type": "regulatory"},
    {"date": "2024-01-16", "headline": "Apple stock downgraded amid China market concerns", "event_type": "analyst"},
    {"date": "2023-11-02", "headline": "Apple Q4 2023 revenue falls 1% year-over-year", "event_type": "earnings"},
    {"date": "2023-09-12", "headline": "Apple launches iPhone 15 with USB-C port", "event_type": "product"},
    
    # Earlier significant events
    {"date": "2023-05-04", "headline": "Apple reports strong Q2 2023 earnings, services revenue hits record", "event_type": "earnings"},
    {"date": "2023-02-02", "headline": "Apple Q1 2023 revenue declines amid economic uncertainty", "event_type": "earnings"},
    {"date": "2022-10-24", "headline": "Apple reports iPhone 14 strong demand in Q4 2022", "event_type": "earnings"},
    {"date": "2022-09-07", "headline": "Apple announces iPhone 14 series and Apple Watch Ultra", "event_type": "product"},
    {"date": "2022-07-28", "headline": "Apple Q3 2022 earnings beat expectations despite supply constraints", "event_type": "earnings"},
]

def setup_finbert():
    """Set up FinBERT for sentiment analysis"""
    print("🤖 Setting up FinBERT...")
    try:
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        print("✅ FinBERT loaded successfully!")
        return sentiment_pipeline
    except Exception as e:
        print(f"❌ Error loading FinBERT: {e}")
        return None

def analyze_real_events(sentiment_pipeline, events):
    """Analyze sentiment of real historical events"""
    print("\n📰 Analyzing REAL historical news events...")
    
    results = []
    for event in events:
        try:
            sentiment_result = sentiment_pipeline(event['headline'])
            sentiment_label = sentiment_result[0]['label']
            sentiment_score = sentiment_result[0]['score']
            
            # Convert to numerical scale
            if sentiment_label.lower() == 'positive':
                numerical_sentiment = sentiment_score
            elif sentiment_label.lower() == 'negative':
                numerical_sentiment = -sentiment_score
            else:
                numerical_sentiment = 0
            
            results.append({
                'date': event['date'],
                'headline': event['headline'],
                'event_type': event['event_type'],
                'sentiment_label': sentiment_label,
                'sentiment_score': sentiment_score,
                'sentiment_numerical': numerical_sentiment
            })
            
            print(f"📅 {event['date']}: {sentiment_label} ({sentiment_score:.3f})")
            print(f"   📰 {event['headline'][:70]}...")
            print(f"   🏷️  Event type: {event['event_type']}")
            
        except Exception as e:
            print(f"❌ Error analyzing: {event['headline'][:30]}... - {e}")
    
    return pd.DataFrame(results)

def load_aapl_stock_data():
    """Load AAPL stock data with proper date handling"""
    try:
        df = pd.read_csv('data/AAPL_historical.csv')
        
        # Convert dates properly
        df['Date_clean'] = pd.to_datetime(df['Date'], utc=True).dt.date
        df['Date'] = df['Date_clean']
        
        print(f"📊 Loaded {len(df)} days of AAPL stock data")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading stock data: {e}")
        return None

def calculate_price_impact(stock_data, news_data, days_window=5):
    """Calculate actual price impact around news events"""
    print(f"\n📈 Calculating price impact ({days_window} days around each event)...")
    
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    news_data['date'] = pd.to_datetime(news_data['date'])
    
    impact_results = []
    
    for idx, news_row in news_data.iterrows():
        event_date = news_row['date']
        
        # Find stock data around the event date
        before_date = event_date - timedelta(days=days_window)
        after_date = event_date + timedelta(days=days_window)
        
        # Get stock prices before and after
        before_price = stock_data[
            (stock_data['Date'] >= before_date) & 
            (stock_data['Date'] < event_date)
        ]['Close'].tail(1)
        
        after_price = stock_data[
            (stock_data['Date'] > event_date) & 
            (stock_data['Date'] <= after_date)
        ]['Close'].head(1)
        
        if not before_price.empty and not after_price.empty:
            price_change = after_price.iloc[0] - before_price.iloc[0]
            price_change_pct = (price_change / before_price.iloc[0]) * 100
            
            impact_results.append({
                'date': event_date,
                'headline': news_row['headline'],
                'event_type': news_row['event_type'],
                'sentiment_numerical': news_row['sentiment_numerical'],
                'price_before': before_price.iloc[0],
                'price_after': after_price.iloc[0],
                'price_change': price_change,
                'price_change_pct': price_change_pct
            })
            
            print(f"📅 {event_date.date()}: {price_change_pct:+.2f}% price change")
            print(f"   📰 {news_row['headline'][:50]}...")
            print(f"   😊 Sentiment: {news_row['sentiment_numerical']:+.3f}")
    
    return pd.DataFrame(impact_results)

def create_comprehensive_visualization(stock_data, news_data, impact_data):
    """Create a comprehensive visualization showing news impact on stock prices"""
    
    # Prepare data for plotting
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    news_data['date'] = pd.to_datetime(news_data['date'])
    
    # Create the visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('AAPL: Real News Events vs Stock Price Analysis', fontsize=16, fontweight='bold')
    
    # 1. Stock price over time with news events marked
    ax1.plot(stock_data['Date'], stock_data['Close'], 'b-', linewidth=1, alpha=0.7, label='AAPL Price')
    
    # Mark news events on price chart
    for idx, row in news_data.iterrows():
        event_date = row['date']
        # Find closest stock price
        closest_price = stock_data[stock_data['Date'] <= event_date]['Close'].tail(1)
        if not closest_price.empty:
            color = 'green' if row['sentiment_numerical'] > 0 else 'red' if row['sentiment_numerical'] < 0 else 'orange'
            ax1.scatter(event_date, closest_price.iloc[0], c=color, s=100, alpha=0.8, edgecolors='black')
    
    ax1.set_title('Stock Price with News Events', fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Sentiment vs Price Change Correlation
    if not impact_data.empty:
        colors = ['green' if s > 0 else 'red' if s < 0 else 'orange' for s in impact_data['sentiment_numerical']]
        scatter = ax2.scatter(impact_data['sentiment_numerical'], impact_data['price_change_pct'], 
                            c=colors, s=100, alpha=0.7, edgecolors='black')
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('News Sentiment Score')
        ax2.set_ylabel('Price Change (%)')
        ax2.set_title('Sentiment vs Price Impact', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # 3. Event types and their average impact
    if not impact_data.empty:
        event_impact = impact_data.groupby('event_type')['price_change_pct'].mean()
        colors_bar = ['green' if x > 0 else 'red' for x in event_impact.values]
        ax3.bar(event_impact.index, event_impact.values, color=colors_bar, alpha=0.7)
        ax3.set_title('Average Price Impact by Event Type', fontweight='bold')
        ax3.set_ylabel('Average Price Change (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # 4. Timeline of sentiment scores
    ax4.scatter(news_data['date'], news_data['sentiment_numerical'], 
               c=['green' if s > 0 else 'red' if s < 0 else 'orange' for s in news_data['sentiment_numerical']], 
               s=100, alpha=0.7, edgecolors='black')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_title('News Sentiment Over Time', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Sentiment Score')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE ANALYSIS RESULTS")
    print("="*60)
    
    if not impact_data.empty:
        print(f"📈 Events analyzed: {len(impact_data)}")
        print(f"📊 Average price change: {impact_data['price_change_pct'].mean():.2f}%")
        
        # Correlation analysis
        correlation = impact_data['sentiment_numerical'].corr(impact_data['price_change_pct'])
        print(f"🔗 Sentiment-Price correlation: {correlation:.3f}")
        
        # Best and worst events
        best_event = impact_data.loc[impact_data['price_change_pct'].idxmax()]
        worst_event = impact_data.loc[impact_data['price_change_pct'].idxmin()]
        
        print(f"\n🚀 Best performing event (+{best_event['price_change_pct']:.2f}%):")
        print(f"   📅 {best_event['date'].date()}")
        print(f"   📰 {best_event['headline'][:60]}...")
        
        print(f"\n📉 Worst performing event ({worst_event['price_change_pct']:.2f}%):")
        print(f"   📅 {worst_event['date'].date()}")
        print(f"   📰 {worst_event['headline'][:60]}...")
        
        # Event type analysis
        print(f"\n📊 Event Type Performance:")
        for event_type, avg_change in impact_data.groupby('event_type')['price_change_pct'].mean().items():
            print(f"   {event_type}: {avg_change:+.2f}% average change")

def main():
    """Main function to run real news analysis"""
    print("🚀 REAL NEWS + STOCK PRICE ANALYSIS")
    print("="*50)
    print("This will show you ACTUAL relationships between news and price movements!\n")
    
    # Setup
    sentiment_pipeline = setup_finbert()
    if not sentiment_pipeline:
        return
    
    # Analyze real events
    news_df = analyze_real_events(sentiment_pipeline, real_aapl_events)
    
    # Load stock data
    stock_df = load_aapl_stock_data()
    if stock_df is None:
        return
    
    # Calculate actual price impacts
    impact_df = calculate_price_impact(stock_df, news_df)
    
    # Create comprehensive visualization
    create_comprehensive_visualization(stock_df, news_df, impact_df)
    
    print("\n✅ ANALYSIS COMPLETE!")
    print("\nNow you can see:")
    print("1. ✅ Real news events marked on stock price chart")
    print("2. ✅ Correlation between sentiment and price changes") 
    print("3. ✅ Which types of events have biggest impact")
    print("4. ✅ Timeline of sentiment over real events")

if __name__ == "__main__":
    main()