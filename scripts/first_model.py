"""
Enhanced Multi-Stock LSTM Training Script
=========================================
Combines ImprovedLSTM features with multi-stock capabilities
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os

class EnhancedMultiStockLSTM(nn.Module):
    """
    Enhanced LSTM combining ImprovedLSTM features with multi-stock capabilities
    """
    def __init__(self, input_size, num_stocks, hidden_size=128, num_layers=2, 
                 dropout=0.2, embedding_dim=16, use_attention=False, output_size=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.embedding_dim = embedding_dim
        
        # Stock embedding layer
        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim)
        
        # Combined input size (features + embedding)
        combined_input_size = input_size + embedding_dim
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            combined_input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Batch normalization for LSTM output
        self.lstm_bn = nn.BatchNorm1d(hidden_size)
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=dropout)
            self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Enhanced fully connected layers with residual connections
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, output_size)
        ])
        
        # Batch normalization for FC layers
        self.fc_bn = nn.ModuleList([
            nn.BatchNorm1d(hidden_size),
            nn.BatchNorm1d(hidden_size // 2)
        ])
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.stock_embedding.weight)
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for fc in self.fc_layers:
            if isinstance(fc, nn.Linear):
                nn.init.xavier_uniform_(fc.weight)
                nn.init.constant_(fc.bias, 0)
    
    def forward(self, x, stock_ids):
        batch_size, seq_len, _ = x.shape
        
        # Get stock embeddings
        stock_emb = self.stock_embedding(stock_ids)  # (batch_size, embedding_dim)
        
        # Expand embeddings to match sequence length
        stock_emb_expanded = stock_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate features with stock embeddings
        x_combined = torch.cat([x, stock_emb_expanded], dim=-1)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x_combined)
        
        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention: (seq_len, batch, hidden_size)
            lstm_out_transposed = lstm_out.transpose(0, 1)
            attn_out, _ = self.attention(
                lstm_out_transposed, 
                lstm_out_transposed, 
                lstm_out_transposed
            )
            # Take the last time step and apply layer norm
            out = self.attention_norm(attn_out[-1])
        else:
            # Take the last time step output
            out = lstm_out[:, -1, :]
        
        # Apply batch normalization
        out = self.lstm_bn(out)
        out = self.dropout(out)
        
        # Fully connected layers with residual connections
        residual = out
        
        # First FC layer
        out = self.fc_layers[0](out)
        out = self.fc_bn[0](out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Add residual connection
        out = out + residual
        
        # Second FC layer
        out = self.fc_layers[1](out)
        out = self.fc_bn[1](out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Final output layer
        out = self.fc_layers[2](out)
        
        return out

class MultiStockDataset(Dataset):
    """Dataset for multi-stock training"""
    def __init__(self, sequences, targets, stock_ids):
        self.sequences = sequences
        self.targets = targets
        self.stock_ids = stock_ids
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.targets[idx]]),
            torch.LongTensor([self.stock_ids[idx]])
        )

def prepare_multi_stock_data(df, seq_len=30, train_split=0.8):
    """Prepare multi-stock data with sequences"""
    
    # Create stock ID mapping
    stocks = sorted(df['Ticker'].unique())
    stock_to_id = {stock: i for i, stock in enumerate(stocks)}
    
    print(f"📊 Found stocks: {stocks}")
    
    # Prepare features (exclude Date, Ticker, Target)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Ticker', 'Target']]
    
    # Scale features per stock
    scalers = {}
    all_sequences = []
    all_targets = []
    all_stock_ids = []
    
    for stock in stocks:
        print(f"🔄 Processing {stock}...")
        
        # Filter stock data
        stock_data = df[df['Ticker'] == stock].copy()
        stock_data = stock_data.sort_values('Date').reset_index(drop=True)
        
        if len(stock_data) < seq_len + 1:
            print(f"⚠️ Skipping {stock} - insufficient data")
            continue
        
        # Scale features
        scaler = StandardScaler()
        X = stock_data[feature_cols].values
        X_scaled = scaler.fit_transform(X)
        scalers[stock] = scaler
        
        # Get targets
        y = stock_data['Target'].values
        
        # Create sequences
        for i in range(seq_len, len(X_scaled)):
            all_sequences.append(X_scaled[i-seq_len:i])
            all_targets.append(y[i])
            all_stock_ids.append(stock_to_id[stock])
    
    # Convert to numpy arrays
    X_seq = np.array(all_sequences)
    y_seq = np.array(all_targets)
    stock_ids = np.array(all_stock_ids)
    
    print(f"📦 Created {len(X_seq)} total sequences")
    
    # Split data (chronological split per stock)
    split_idx = int(len(X_seq) * train_split)
    
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    stock_train, stock_test = stock_ids[:split_idx], stock_ids[split_idx:]
    
    return (X_train, y_train, stock_train), (X_test, y_test, stock_test), scalers, stock_to_id

def main():
    print("🚀 Starting Enhanced Multi-Stock LSTM Training")
    
    # 1. Load processed data
    print("📊 Loading processed data...")
    data_file = "data/processed/enhanced_multi_stock_20250716_113531.csv"
    df = pd.read_csv(data_file)
    
    print(f"✅ Loaded {len(df)} samples for {df['Ticker'].nunique()} stocks")
    
    # 2. Prepare multi-stock data
    (X_train, y_train, stock_train), (X_test, y_test, stock_test), scalers, stock_to_id = prepare_multi_stock_data(df)
    
    print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"🎯 Target distribution - Train Up: {np.sum(y_train)}, Down: {len(y_train)-np.sum(y_train)}")
    
    # 3. Create datasets
    train_dataset = MultiStockDataset(X_train, y_train, stock_train)
    test_dataset = MultiStockDataset(X_test, y_test, stock_test)
    
    # 4. Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. Create enhanced model
    input_size = X_train.shape[2]  # Number of features
    num_stocks = len(stock_to_id)
    
    model = EnhancedMultiStockLSTM(
        input_size=input_size,
        num_stocks=num_stocks,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        embedding_dim=32,
        use_attention=True,  # Enable attention for better performance
        output_size=1
    )
    
    print(f"🤖 Created enhanced model:")
    print(f"   - Input features: {input_size}")
    print(f"   - Number of stocks: {num_stocks}")
    print(f"   - With attention: True")
    print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # 7. Training loop
    print("🚀 Starting training...")
    num_epochs = 100
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    training_losses = []
    validation_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y, batch_stock_ids in train_loader:
            batch_stock_ids = batch_stock_ids.squeeze()
            batch_y = batch_y.squeeze()
            
            optimizer.zero_grad()
            outputs = model(batch_X, batch_stock_ids).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += len(batch_y)
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y, batch_stock_ids in test_loader:
                batch_stock_ids = batch_stock_ids.squeeze()
                batch_y = batch_y.squeeze()
                
                outputs = model(batch_X, batch_stock_ids).squeeze()
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += len(batch_y)
        
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Track losses
        training_losses.append(avg_train_loss)
        validation_losses.append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_enhanced_multistock_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.3f}")
            print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.3f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 8. Final evaluation
    print("\n📊 Final Evaluation...")
    model.load_state_dict(torch.load('models/best_enhanced_multistock_model.pth'))
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y, batch_stock_ids in test_loader:
            batch_stock_ids = batch_stock_ids.squeeze()
            batch_y = batch_y.squeeze()
            
            outputs = model(batch_X, batch_stock_ids).squeeze()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Calculate final metrics
    final_accuracy = accuracy_score(all_targets, all_predictions)
    
    print(f"\n✅ FINAL RESULTS:")
    print(f"📈 Best Validation Loss: {best_loss:.4f}")
    print(f"📈 Final Test Accuracy: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
    
    if final_accuracy > 0.52:
        print(f"🎯 Good! Above random chance (50%)")
        if final_accuracy > 0.58:
            print(f"🚀 Excellent! Very good for multi-stock prediction!")
    
    # Detailed classification report
    print("\n📋 Detailed Results:")
    print(classification_report(all_targets, all_predictions, 
                              target_names=['Down', 'Up']))
    
    # Per-stock analysis
    print("\n📊 Per-Stock Analysis:")
    stock_results = {}
    for stock_id, stock_name in {v: k for k, v in stock_to_id.items()}.items():
        mask = stock_test == stock_id
        if np.sum(mask) > 0:
            stock_acc = accuracy_score(
                np.array(all_targets)[mask], 
                np.array(all_predictions)[mask]
            )
            stock_results[stock_name] = stock_acc
            print(f"   {stock_name}: {stock_acc:.3f} ({np.sum(mask)} samples)")
    
    # Save everything
    os.makedirs('models', exist_ok=True)
    
    # Save scalers and mappings
    import pickle
    with open('models/scalers_and_mappings.pkl', 'wb') as f:
        pickle.dump({
            'scalers': scalers,
            'stock_to_id': stock_to_id,
            'feature_cols': [col for col in df.columns if col not in ['Date', 'Ticker', 'Target']]
        }, f)
    
    print("\n💾 Saved:")
    print("   - Best model: models/best_enhanced_multistock_model.pth")
    print("   - Scalers & mappings: models/scalers_and_mappings.pkl")
    
    print(f"\n🎉 Enhanced Multi-Stock Training Complete!")
    print(f"🎯 Best overall accuracy: {final_accuracy:.1%}")

if __name__ == "__main__":
    main()