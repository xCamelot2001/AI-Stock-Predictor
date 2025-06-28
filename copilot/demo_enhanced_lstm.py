"""
Quick Demo: Enhanced LSTM vs Original LSTM
Compare the performance improvements of the enhanced model
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(".")
sys.path.append(project_root)

# Import models
from src.models import ImprovedLSTMModel
from src.sequence_dataset import StockSequenceDataset

# Original simple LSTM (from your notebook)
class OriginalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def prepare_data():
    """Prepare AAPL data for comparison"""
    print("📊 Loading and preparing AAPL data...")
    
    df = pd.read_csv("data/AAPL_historical.csv", parse_dates=["Date"], index_col="Date")
    df.columns = df.columns.str.lower()
    
    # Add technical indicators
    df['return'] = df['close'].pct_change()
    df['ma_7'] = df['close'].rolling(window=7).mean()
    df['ma_21'] = df['close'].rolling(window=21).mean()
    
    try:
        import ta
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['volume_change'] = df['volume'].pct_change()
        
        # Additional features for enhanced model
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
        
    except ImportError:
        print("⚠️ 'ta' library not found, using basic features only")
        df['rsi'] = 50  # Dummy RSI
        df['volume_change'] = df['volume'].pct_change()
        df['macd'] = 0
        df['macd_signal'] = 0
        df['bb_width'] = 0.1
    
    df.dropna(inplace=True)
    
    # Scale data
    feature_cols = [col for col in df.columns if col != 'close']
    scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    df_scaled['close'] = target_scaler.fit_transform(df[['close']])
    
    # Split data
    train_size = int(len(df_scaled) * 0.8)
    train_df = df_scaled.iloc[:train_size]
    test_df = df_scaled.iloc[train_size:]
    
    return train_df, test_df, target_scaler, len(feature_cols)

def train_model(model, train_df, test_df, model_name, epochs=30):
    """Train a model and return test predictions"""
    print(f"\n🚀 Training {model_name}...")
    
    # Create datasets
    seq_len = 20
    train_dataset = StockSequenceDataset(train_df, target_col='close', seq_len=seq_len)
    test_dataset = StockSequenceDataset(test_df.iloc[:-seq_len], target_col='close', seq_len=seq_len)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")
    
    # Test evaluation
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output = model(x_batch).squeeze()
            preds.extend(output.numpy())
            actuals.extend(y_batch.numpy())
    
    return np.array(preds), np.array(actuals)

def compare_models():
    """Compare original vs enhanced LSTM performance"""
    print("🔥 Enhanced LSTM vs Original LSTM Comparison")
    print("=" * 60)
    
    # Prepare data
    train_df, test_df, target_scaler, input_size = prepare_data()
    
    # Create models
    original_model = OriginalLSTM(input_size=input_size)
    enhanced_model = ImprovedLSTMModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        use_attention=True
    )
    
    print(f"📊 Dataset sizes - Train: {len(train_df)}, Test: {len(test_df)}")
    print(f"📊 Input features: {input_size}")
    
    # Train both models
    orig_preds, orig_actuals = train_model(original_model, train_df, test_df, "Original LSTM")
    enh_preds, enh_actuals = train_model(enhanced_model, train_df, test_df, "Enhanced LSTM")
    
    # Convert back to original scale
    orig_preds_scaled = target_scaler.inverse_transform(orig_preds.reshape(-1, 1)).flatten()
    orig_actuals_scaled = target_scaler.inverse_transform(orig_actuals.reshape(-1, 1)).flatten()
    
    enh_preds_scaled = target_scaler.inverse_transform(enh_preds.reshape(-1, 1)).flatten()
    enh_actuals_scaled = target_scaler.inverse_transform(enh_actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    def calculate_metrics(actual, pred, model_name):
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        
        print(f"\n📈 {model_name} Performance:")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   MAE:  ${mae:.2f}")
        print(f"   R²:   {r2:.4f}")
        print(f"   MAPE: {mape:.2f}%")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}
    
    orig_metrics = calculate_metrics(orig_actuals_scaled, orig_preds_scaled, "Original LSTM")
    enh_metrics = calculate_metrics(enh_actuals_scaled, enh_preds_scaled, "Enhanced LSTM")
    
    # Calculate improvements
    print(f"\n🎯 Improvements:")
    rmse_improvement = ((orig_metrics['rmse'] - enh_metrics['rmse']) / orig_metrics['rmse']) * 100
    mae_improvement = ((orig_metrics['mae'] - enh_metrics['mae']) / orig_metrics['mae']) * 100
    r2_improvement = ((enh_metrics['r2'] - orig_metrics['r2']) / orig_metrics['r2']) * 100
    
    print(f"   RMSE: {rmse_improvement:.1f}% better")
    print(f"   MAE:  {mae_improvement:.1f}% better")
    print(f"   R²:   {r2_improvement:.1f}% better")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Predictions comparison
    plt.subplot(1, 3, 1)
    sample_size = min(100, len(orig_actuals_scaled))
    indices = range(sample_size)
    
    plt.plot(indices, orig_actuals_scaled[:sample_size], label='Actual', alpha=0.8, linewidth=2)
    plt.plot(indices, orig_preds_scaled[:sample_size], label='Original LSTM', alpha=0.7)
    plt.plot(indices, enh_preds_scaled[:sample_size], label='Enhanced LSTM', alpha=0.7)
    plt.title('Predictions Comparison (First 100 samples)')
    plt.xlabel('Time Step')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot comparison
    plt.subplot(1, 3, 2)
    plt.scatter(orig_actuals_scaled, orig_preds_scaled, alpha=0.6, label=f'Original (R²={orig_metrics["r2"]:.3f})', s=20)
    plt.scatter(enh_actuals_scaled, enh_preds_scaled, alpha=0.6, label=f'Enhanced (R²={enh_metrics["r2"]:.3f})', s=20)
    
    min_val = min(orig_actuals_scaled.min(), enh_actuals_scaled.min())
    max_val = max(orig_actuals_scaled.max(), enh_actuals_scaled.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
    
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Predicted vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Metrics comparison
    plt.subplot(1, 3, 3)
    metrics = ['RMSE', 'MAE', 'R²', 'MAPE']
    orig_values = [orig_metrics['rmse'], orig_metrics['mae'], orig_metrics['r2'] * 100, orig_metrics['mape']]
    enh_values = [enh_metrics['rmse'], enh_metrics['mae'], enh_metrics['r2'] * 100, enh_metrics['mape']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, orig_values, width, label='Original LSTM', alpha=0.8)
    plt.bar(x + width/2, enh_values, width, label='Enhanced LSTM', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return orig_metrics, enh_metrics

if __name__ == "__main__":
    try:
        orig_metrics, enh_metrics = compare_models()
        print("\n✅ Comparison completed!")
        
        print(f"\n💡 Key Improvements in Enhanced LSTM:")
        print("   • Attention mechanism for better sequence modeling")
        print("   • Batch normalization for stable training")
        print("   • Dropout for regularization")
        print("   • Better weight initialization")
        print("   • Gradient clipping (in full training)")
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        print("Make sure you have the required data files and dependencies installed")
