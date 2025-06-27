import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

def download_stock_data():
    """
    Download historical stock data for the project
    Based on your scope: 3-5 major stocks, 2-3 years of data
    """
    
    # Your target stocks (start with these 5 major ones)
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Date range: 3 years back from today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)  # 3 years
    
    print(f"Downloading data from {start_date.date()} to {end_date.date()}")
    
    # Create directory for data
    os.makedirs('data', exist_ok=True)
    
    all_data = {}
    
    for stock in stocks:
        print(f"\nDownloading {stock}...")
        try:
            ticker = yf.Ticker(stock)
            
            # Download daily data
            hist = ticker.history(
                start=start_date,
                end=end_date,
                interval='1d'  # Daily data
            )
            
            if not hist.empty:
                # Add some basic info
                hist['Symbol'] = stock
                hist['Date'] = hist.index
                
                # Save individual stock data
                hist.to_csv(f'data/{stock}_historical.csv')
                all_data[stock] = hist
                
                print(f" {stock}: {len(hist)} days of data")
                print(f"   Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
                print(f"   Price range: ${hist['Close'].min():.2f} - ${hist['Close'].max():.2f}")
            else:
                print(f"No data for {stock}")
                
        except Exception as e:
            print(f"Error downloading {stock}: {e}")
    
    # Combine all data
    if all_data:
        combined_df = pd.concat(all_data.values(), ignore_index=True)
        combined_df.to_csv('data/all_stocks_combined.csv', index=False)
        print(f"\n Combined dataset saved: {len(combined_df)} total records")
        
        # Quick summary
        print("\n Quick Summary:")
        for stock in stocks:
            if stock in all_data:
                stock_data = all_data[stock]
                print(f"{stock}: {len(stock_data)} days, Latest close: ${stock_data['Close'][-1]:.2f}")
    
    return all_data

def explore_data(stock_symbol='AAPL'):
    """
    Quick exploration of downloaded data
    """
    try:
        df = pd.read_csv(f'data/{stock_symbol}_historical.csv')
        print(f"\n Exploring {stock_symbol} data:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print("\nFirst few rows:")
        print(df.head())
        print("\nBasic stats:")
        print(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
        
    except FileNotFoundError:
        print(f"File not found. Run download_stock_data() first.")

if __name__ == "__main__":
    # Download the data
    print(" Starting stock data download...")
    data = download_stock_data()
    
    # Explore one stock as example
    if data:
        explore_data('AAPL')
        
    print("\n Data collection complete!")
