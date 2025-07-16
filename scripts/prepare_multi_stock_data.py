"""
Multi-Stock Data Preparation Script
Combines individual stock files into a unified dataset for multi-stock training
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta

def download_multiple_stocks(symbols, period="2y", save_dir="data"):
    """
    Download historical data for multiple stocks
    """
    os.makedirs(save_dir, exist_ok=True)
    
    all_data = []
    
    for symbol in symbols:
        print(f"📊 Downloading {symbol}...")
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            if not data.empty:
                # Reset index to make Date a column
                data = data.reset_index()
                data['Symbol'] = symbol
                
                # Save individual file
                data.to_csv(f"{save_dir}/{symbol}_historical.csv", index=False)
                
                # Add to combined data
                all_data.append(data)
                print(f"✅ {symbol}: {len(data)} records")
            else:
                print(f"❌ No data for {symbol}")
                
        except Exception as e:
            print(f"❌ Error downloading {symbol}: {e}")
    
    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.sort_values(['Symbol', 'Date'])
        
        # Save combined file
        combined_path = f"{save_dir}/all_stocks_combined.csv"
        combined_df.to_csv(combined_path, index=False)
        
        print(f"\n🎉 Combined data saved to {combined_path}")
        print(f"📊 Total records: {len(combined_df)}")
        print(f"📊 Stocks: {combined_df['Symbol'].unique()}")
        print(f"📅 Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        
        return combined_df
    
    return None

def prepare_multi_stock_features(df):
    """
    Add technical indicators to multi-stock dataset
    """
    print("🔧 Adding technical indicators...")
    
    enhanced_data = []
    
    for symbol in df['Symbol'].unique():
        print(f"   Processing {symbol}...")
        stock_data = df[df['Symbol'] == symbol].copy()
        stock_data = stock_data.sort_values('Date')
        
        try:
            import ta
            
            # Basic returns and moving averages
            stock_data['Return'] = stock_data['Close'].pct_change()
            stock_data['MA_7'] = stock_data['Close'].rolling(window=7).mean()
            stock_data['MA_21'] = stock_data['Close'].rolling(window=21).mean()
            stock_data['MA_50'] = stock_data['Close'].rolling(window=50).mean()
            
            # Technical indicators
            stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close'], window=14).rsi()
            stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
            
            # MACD
            macd = ta.trend.MACD(stock_data['Close'])
            stock_data['MACD'] = macd.macd()
            stock_data['MACD_Signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(stock_data['Close'], window=20)
            stock_data['BB_Upper'] = bb.bollinger_hband()
            stock_data['BB_Lower'] = bb.bollinger_lband()
            stock_data['BB_Width'] = (stock_data['BB_Upper'] - stock_data['BB_Lower']) / stock_data['Close']
            
            # Volatility
            stock_data['Volatility_20'] = stock_data['Return'].rolling(window=20).std()
            
            # Price ratios
            stock_data['High_Low_Ratio'] = stock_data['High'] / stock_data['Low']
            stock_data['Close_Open_Ratio'] = stock_data['Close'] / stock_data['Open']
            
        except ImportError:
            print("⚠️ 'ta' library not found, using basic indicators only")
            stock_data['Return'] = stock_data['Close'].pct_change()
            stock_data['MA_7'] = stock_data['Close'].rolling(window=7).mean()
            stock_data['MA_21'] = stock_data['Close'].rolling(window=21).mean()
            stock_data['Volume_Change'] = stock_data['Volume'].pct_change()
        
        # Remove NaN values
        stock_data = stock_data.dropna()
        enhanced_data.append(stock_data)
    
    if enhanced_data:
        result = pd.concat(enhanced_data, ignore_index=True)
        result = result.sort_values(['Symbol', 'Date'])
        
        print(f"✅ Enhanced data shape: {result.shape}")
        print(f"📊 Features: {list(result.columns)}")
        
        return result
    
    return df

def main():
    """Main execution function"""
    print("🚀 Multi-Stock Data Preparation")
    print("=" * 50)
    
    # Define stocks to download
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Check if combined data already exists
    combined_path = "data/all_stocks_combined.csv"
    if os.path.exists(combined_path):
        print(f"📊 Found existing combined data: {combined_path}")
        df = pd.read_csv(combined_path, parse_dates=['Date'])
        print(f"📊 Loaded {len(df)} records for {df['Symbol'].nunique()} stocks")
    else:
        print("📡 Downloading fresh stock data...")
        df = download_multiple_stocks(stocks)
        
        if df is None:
            print("❌ Failed to download stock data")
            return
    
    # Add technical indicators
    enhanced_df = prepare_multi_stock_features(df)
    
    # Save enhanced data
    enhanced_path = "data/all_stocks_enhanced.csv"
    enhanced_df.to_csv(enhanced_path, index=False)
    
    print(f"\n🎉 Enhanced multi-stock data saved to {enhanced_path}")
    
    # Display summary statistics
    print("\n📈 Data Summary:")
    for symbol in enhanced_df['Symbol'].unique():
        symbol_data = enhanced_df[enhanced_df['Symbol'] == symbol]
        print(f"   {symbol}: {len(symbol_data)} records, "
              f"{symbol_data['Date'].min()} to {symbol_data['Date'].max()}")
    
    print(f"\n🔍 Available features: {len(enhanced_df.columns)}")
    print("✅ Multi-stock data preparation completed!")

if __name__ == "__main__":
    main()
