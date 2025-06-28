"""
LSTM Training Script - PyTorch Implementation
============================================

Purpose: Train LSTM model on processed stock data using PyTorch
- Load processed data with 37 features
- Create sequences for LSTM
- Train model with realistic expectations (55-65% accuracy)
"""

import pandas as pd
import numpy as np
import os
import glob
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# ML libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StockLSTM(nn.Module):
    """
    PyTorch LSTM model for stock prediction
    
    Architecture:
    - 2 LSTM layers with dropout
    - Dense layers for final prediction
    - Sigmoid output for binary classification
    """
    
    def __init__(self, input_size, hidden_size=50, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 25)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(25, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Dense layers
        out = F.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out

class PyTorchLSTMTrainer:
    """
    PyTorch LSTM trainer for stock prediction
    
    Key features:
    - Creates sequences from processed data
    - Proper time series splitting
    - PyTorch implementation
    - Binary classification (up/down)
    """
    
    def __init__(self, lookback_days=30):
        """
        Initialize trainer
        
        Args:
            lookback_days: How many past days to use for prediction
        """
        self.lookback_days = lookback_days
        self.scaler = MinMaxScaler()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        logger.info(f"🤖 PyTorch LSTM Trainer initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"📅 Lookback: {lookback_days} days")
    
    def load_processed_data(self, symbol='AAPL'):
        """Load the latest processed data"""
        
        # Find latest processed file
        pattern = f"data/processed/{symbol}_processed_*.csv"
        files = glob.glob(pattern)
        
        if not files:
            logger.error(f"❌ No processed data found for {symbol}")
            return None
        
        latest_file = max(files, key=os.path.getctime)
        logger.info(f"📊 Loading: {latest_file}")
        
        # Load data
        df = pd.read_csv(latest_file, index_col='Date', parse_dates=True)
        
        logger.info(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
        logger.info(f"📅 Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        return df
    
    def prepare_features_and_target(self, df):
        """Separate features and target, scale features"""
        logger.info("🔧 Preparing features and target...")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != 'Target']
        
        X = df[feature_cols].values
        y = df['Target'].values
        
        # Scale features (but not target - it's already 0/1)
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"📊 Features: {X_scaled.shape[1]} columns")
        logger.info(f"🎯 Target distribution: {np.bincount(y)}")
        
        return X_scaled, y, feature_cols
    
    def create_sequences(self, X, y):
        """Create sequences for LSTM training"""
        logger.info(f"📦 Creating sequences with {self.lookback_days} day lookback...")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(self.lookback_days, len(X)):
            X_sequences.append(X[i-self.lookback_days:i])
            y_sequences.append(y[i])
        
        X_seq = np.array(X_sequences)
        y_seq = np.array(y_sequences)
        
        logger.info(f"✅ Created {len(X_seq)} sequences")
        logger.info(f"📊 Sequence shape: {X_seq.shape}")
        
        return X_seq, y_seq
    
    def split_data_time_series(self, X_seq, y_seq, train_ratio=0.7, val_ratio=0.15):
        """Split data for time series (NO SHUFFLING!)"""
        logger.info("✂️  Splitting data (time series order preserved)...")
        
        n_samples = len(X_seq)
        
        # Calculate split points
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split data
        X_train = X_seq[:train_end]
        X_val = X_seq[train_end:val_end]
        X_test = X_seq[val_end:]
        
        y_train = y_seq[:train_end]
        y_val = y_seq[train_end:val_end]
        y_test = y_seq[val_end:]
        
        logger.info(f"📊 Data splits:")
        logger.info(f"   Train: {len(X_train)} samples")
        logger.info(f"   Validation: {len(X_val)} samples") 
        logger.info(f"   Test: {len(X_test)} samples")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
        """Create PyTorch data loaders"""
        
        train_dataset = StockDataset(X_train, y_train)
        val_dataset = StockDataset(X_val, y_val)
        test_dataset = StockDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for time series
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader, val_loader, input_size, epochs=50):
        """Train the PyTorch LSTM model"""
        logger.info("🚀 Starting model training...")
        
        # Create model
        self.model = StockLSTM(input_size=input_size).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training history
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}")
                logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_model_temp.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/best_model_temp.pth'))
        
        logger.info("✅ Training completed!")
        
        # Return history for plotting
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accuracies,
            'val_accuracy': val_accuracies
        }
        
        return history
    
    def evaluate_model(self, test_loader):
        """Evaluate model performance"""
        logger.info("📊 Evaluating model...")
        
        self.model.eval()
        y_true = []
        y_pred = []
        y_pred_proba = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X).squeeze()
                predicted = (outputs > 0.5).float()
                
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_pred_proba.extend(outputs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        logger.info(f"📈 Model Performance:")
        logger.info(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed report
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Down', 'Up']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"              Predicted")
        print(f"Actual    Down  Up")
        print(f"Down      {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"Up        {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        return accuracy, y_pred, y_pred_proba
    
    def plot_training_history(self, history):
        """Plot training history"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(history['train_accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        logger.info("📊 Training history plot saved: results/training_history.png")
        logger.info("📁 Open results/training_history.png to view the plot")
    
    def save_model(self, symbol, accuracy):
        """Save trained PyTorch model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_file = f"models/{symbol}_pytorch_lstm_{timestamp}.pth"
        torch.save(self.model.state_dict(), model_file)
        
        # Save scaler
        import joblib
        scaler_file = f"models/{symbol}_scaler_{timestamp}.pkl"
        joblib.dump(self.scaler, scaler_file)
        
        logger.info(f"💾 Model saved: {model_file}")
        logger.info(f"💾 Scaler saved: {scaler_file}")
        
        return model_file, scaler_file
    
    def train_complete_pipeline(self, symbol='AAPL'):
        """Complete training pipeline"""
        logger.info(f"\n🚀 Starting complete PyTorch LSTM training for {symbol}")
        
        # 1. Load data
        df = self.load_processed_data(symbol)
        if df is None:
            return None
        
        # 2. Prepare features and target
        X, y, feature_cols = self.prepare_features_and_target(df)
        
        # 3. Create sequences
        X_seq, y_seq = self.create_sequences(X, y)
        
        # 4. Split data
        (X_train, X_val, X_test), (y_train, y_val, y_test) = self.split_data_time_series(X_seq, y_seq)
        
        # 5. Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # 6. Train model
        input_size = X_seq.shape[2]  # Number of features
        history = self.train_model(train_loader, val_loader, input_size)
        
        # 7. Evaluate model
        accuracy, y_pred, y_pred_proba = self.evaluate_model(test_loader)
        
        # 8. Plot training history
        self.plot_training_history(history)
        
        # 9. Save model
        model_file, scaler_file = self.save_model(symbol, accuracy)
        
        logger.info(f"\n🎉 Training complete!")
        logger.info(f"📊 Final accuracy: {accuracy:.1%}")
        logger.info(f"💾 Model saved: {model_file}")
        
        return {
            'accuracy': accuracy,
            'model_file': model_file,
            'scaler_file': scaler_file,
            'feature_columns': feature_cols
        }


def main():
    """Train PyTorch LSTM model on AAPL data"""
    
    # Create trainer
    trainer = PyTorchLSTMTrainer(lookback_days=30)
    
    # Train model
    result = trainer.train_complete_pipeline('AAPL')
    
    if result:
        print(f"\n✅ Success!")
        print(f"📊 Final Accuracy: {result['accuracy']:.1%}")
        print(f"💾 Model: {result['model_file']}")
        print(f"📁 Results saved in: results/")
        
        # Realistic expectations message
        if result['accuracy'] > 0.55:
            print(f"🎯 Great! {result['accuracy']:.1%} is above random chance (50%)")
            if result['accuracy'] > 0.60:
                print(f"🚀 Excellent! {result['accuracy']:.1%} is very good for stock prediction")
        else:
            print(f"⚠️  {result['accuracy']:.1%} is close to random. Consider more data or different features.")
            
    else:
        print("❌ Training failed")


if __name__ == "__main__":
    main()