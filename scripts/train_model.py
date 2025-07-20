"""
Simple LSTM Training Script
==========================
Train LSTM on your processed data - ONE FILE TO RULE THEM ALL
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
import sys

# Import your existing components
from src.models import ImprovedLSTMModel
from src.sequence_dataset import StockSequenceDataset

def main():
    print("🚀 Starting LSTM Training")
    
    # 1. Load your processed data
    print("📊 Loading processed data...")
    data_file = "../data/processed/enhanced_multi_stock_20250716_113531.csv"  # Update this path
    df = pd.read_csv(data_file)
    
    # 2. Focus on one stock first (AAPL)
    aapl_data = df[df['Ticker'] == 'AAPL'].copy()
    aapl_data = aapl_data.sort_values('Date').reset_index(drop=True)
    print(f"✅ Loaded {len(aapl_data)} AAPL samples")
    
    # 3. Prepare features and target
    feature_cols = [col for col in aapl_data.columns if col not in ['Date', 'Ticker', 'Target']]
    
    X = aapl_data[feature_cols].values
    y = aapl_data['Target'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"📊 Features: {len(feature_cols)}")
    print(f"🎯 Target distribution: Up={np.sum(y)}, Down={len(y)-np.sum(y)}")
    
    # 4. Create sequences for LSTM
    seq_len = 30
    X_sequences = []
    y_sequences = []
    
    for i in range(seq_len, len(X_scaled)):
        X_sequences.append(X_scaled[i-seq_len:i])
        y_sequences.append(y[i])
    
    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)
    
    print(f"📦 Created {len(X_seq)} sequences of length {seq_len}")
    
    # 5. Split data (80% train, 20% test - no shuffling for time series!)
    split_idx = int(len(X_seq) * 0.8)
    
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 6. Create PyTorch datasets and dataloaders
    train_dataset = StockSequenceDataset(
        pd.DataFrame(np.concatenate([X_train.reshape(-1, X_train.shape[-1]), 
                                   y_train.reshape(-1, 1)], axis=1)),
        target_col=X_train.shape[-1],  # Last column is target
        seq_len=1  # Already sequenced
    )
    
    # Actually, let's make this simpler - direct PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 7. Create model
    input_size = X_train.shape[2]  # Number of features
    model = ImprovedLSTMModel(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        output_size=1,
        use_attention=False
    )
    
    print(f"🤖 Created model with {input_size} input features")
    
    # 8. Training setup
    criterion = nn.BCEWithLogitsLoss()  # For binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 9. Training loop
    print("🚀 Starting training...")
    model.train()
    
    num_epochs = 50
    batch_size = 32
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 10. Evaluation
    print("📊 Evaluating model...")
    model.eval()
    
    with torch.no_grad():
        test_outputs = model(X_test_tensor).squeeze()
        test_probs = torch.sigmoid(test_outputs)
        test_preds = (test_probs > 0.5).float()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, test_preds.numpy())
    
    print(f"\n✅ RESULTS:")
    print(f"📈 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    if accuracy > 0.55:
        print(f"🎯 Great! Above random chance (50%)")
        if accuracy > 0.60:
            print(f"🚀 Excellent! Very good for stock prediction!")
    
    # Classification report
    print("\n📋 Detailed Results:")
    print(classification_report(y_test, test_preds.numpy(), 
                              target_names=['Down', 'Up']))
    
    # 11. Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/lstm_model.pth')
    print("💾 Model saved to models/lstm_model.pth")
    
    print("\n🎉 Training Complete!")

if __name__ == "__main__":
    main()