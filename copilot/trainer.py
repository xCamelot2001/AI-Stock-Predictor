import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

from .models import ImprovedLSTMModel, MultiStockLSTM
from .sequence_dataset import StockSequenceDataset, MultiStockDataset

class LSTMHyperparameterTuner:
    """
    Hyperparameter tuning class for LSTM models
    """
    
    def __init__(self, base_model_class=ImprovedLSTMModel):
        self.base_model_class = base_model_class
        self.best_params = None
        self.best_score = float('inf')
        self.results = []
        
    def get_param_grid(self):
        """Define hyperparameter search space"""
        return {
            'hidden_size': [64, 128, 256],
            'num_layers': [1, 2, 3],
            'dropout': [0.1, 0.2, 0.3],
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64],
            'seq_len': [10, 20, 30],
            'use_attention': [False, True]
        }
    
    def create_model(self, input_size: int, params: Dict):
        """Create model with given parameters"""
        return self.base_model_class(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            use_attention=params['use_attention']
        )
    
    def train_and_evaluate(self, model, train_loader, val_loader, params: Dict, 
                          num_epochs: int = 50, device: str = 'cpu'):
        """Train model and return validation loss"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )
        
        model.to(device)
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                output = model(x_batch).squeeze()
                loss = criterion(output, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    output = model(x_batch).squeeze()
                    loss = criterion(output, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                break
        
        return best_val_loss
    
    def tune(self, X_train, y_train, input_size: int, max_trials: int = 20,
             validation_split: float = 0.2, device: str = 'cpu'):
        """
        Perform hyperparameter tuning using random search
        """
        param_grid = self.get_param_grid()
        param_combinations = list(ParameterGrid(param_grid))
        
        # Random sample if too many combinations
        if len(param_combinations) > max_trials:
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:max_trials]
        
        print(f"🔍 Starting hyperparameter tuning with {len(param_combinations)} combinations...")
        
        for i, params in enumerate(param_combinations):
            print(f"Trial {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Create dataset with current parameters
                dataset = StockSequenceDataset(
                    pd.concat([X_train, y_train], axis=1),
                    target_col=y_train.name,
                    seq_len=params['seq_len']
                )
                
                # Split into train/validation
                val_size = int(len(dataset) * validation_split)
                train_size = len(dataset) - val_size
                train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                
                # Create data loaders
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=params['batch_size'], 
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=params['batch_size'], 
                    shuffle=False
                )
                
                # Create and train model
                model = self.create_model(input_size, params)
                val_loss = self.train_and_evaluate(
                    model, train_loader, val_loader, params, device=device
                )
                
                # Store results
                result = {**params, 'val_loss': val_loss}
                self.results.append(result)
                
                # Update best parameters
                if val_loss < self.best_score:
                    self.best_score = val_loss
                    self.best_params = params.copy()
                    print(f"✅ New best score: {val_loss:.4f}")
                
            except Exception as e:
                print(f"❌ Trial failed: {e}")
                continue
        
        print(f"🎯 Best parameters found: {self.best_params}")
        print(f"🎯 Best validation loss: {self.best_score:.4f}")
        
        return self.best_params, self.best_score
    
    def save_results(self, filepath: str):
        """Save tuning results to file"""
        results_data = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to {filepath}")

class MultiStockTrainer:
    """
    Trainer class for multi-stock LSTM models
    """
    
    def __init__(self, model_params: Dict = None):
        self.model_params = model_params or {}
        self.model = None
        self.scalers = {}
        self.stock_to_id = {}
        self.id_to_stock = {}
        self.training_history = []
        
    def prepare_multi_stock_data(self, data_path: str, stocks: List[str] = None,
                                seq_len: int = 20, train_split: float = 0.8):
        """
        Prepare data for multi-stock training
        """
        # Read combined stock data
        df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
        
        if stocks is None:
            stocks = df['Symbol'].unique().tolist()
        
        # Create stock ID mapping
        self.stock_to_id = {stock: i for i, stock in enumerate(stocks)}
        self.id_to_stock = {i: stock for stock, i in self.stock_to_id.items()}
        
        # Prepare datasets for each stock
        stock_datasets = []
        
        for stock in stocks:
            stock_data = df[df['Symbol'] == stock].copy()
            stock_data = stock_data.drop(columns=['Symbol'])
            
            # Add technical indicators (similar to your current approach)
            stock_data = self._add_technical_indicators(stock_data)
            
            # Scale features for this stock
            from sklearn.preprocessing import StandardScaler
            
            feature_cols = [col for col in stock_data.columns if col != 'Close']
            target_col = 'Close'
            
            # Store scalers for this stock
            self.scalers[stock] = {
                'features': StandardScaler(),
                'target': StandardScaler()
            }
            
            # Scale data
            stock_data[feature_cols] = self.scalers[stock]['features'].fit_transform(
                stock_data[feature_cols]
            )
            stock_data[target_col] = self.scalers[stock]['target'].fit_transform(
                stock_data[[target_col]]
            )
            
            # Create dataset
            dataset = MultiStockDataset(
                stock_data, 
                stock_id=self.stock_to_id[stock],
                target_col=target_col,
                seq_len=seq_len
            )
            
            stock_datasets.append(dataset)
        
        # Combine all stock datasets
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset(stock_datasets)
        
        # Split into train/test
        total_size = len(combined_dataset)
        train_size = int(total_size * train_split)
        test_size = total_size - train_size
        
        train_dataset, test_dataset = random_split(
            combined_dataset, [train_size, test_size]
        )
        
        return train_dataset, test_dataset
    
    def _add_technical_indicators(self, df: pd.DataFrame):
        """Add technical indicators to stock data"""
        try:
            import ta
            
            # Basic returns and moving averages
            df['return'] = df['Close'].pct_change()
            df['ma_7'] = df['Close'].rolling(window=7).mean()
            df['ma_21'] = df['Close'].rolling(window=21).mean()
            
            # Technical indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            df['volume_change'] = df['Volume'].pct_change()
            
            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(df['Close'], window=20)
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['Close']
            
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Remove NaN values
            df.dropna(inplace=True)
            
        except ImportError:
            print("⚠️ 'ta' library not found, using basic indicators only")
            df['return'] = df['Close'].pct_change()
            df['ma_7'] = df['Close'].rolling(window=7).mean()
            df['ma_21'] = df['Close'].rolling(window=21).mean()
            df['volume_change'] = df['Volume'].pct_change()
            df.dropna(inplace=True)
        
        return df
    
    def create_model(self, input_size: int, num_stocks: int):
        """Create multi-stock LSTM model"""
        self.model = MultiStockLSTM(
            input_size=input_size,
            num_stocks=num_stocks,
            **self.model_params
        )
        return self.model
    
    def train(self, train_dataset, val_dataset=None, num_epochs: int = 100,
              batch_size: int = 32, learning_rate: float = 0.001,
              device: str = 'cpu', save_path: str = None):
        """
        Train the multi-stock model
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.model.to(device)
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 15
        
        print(f"🚀 Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for x_batch, y_batch, stock_ids in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                stock_ids = stock_ids.to(device)
                
                optimizer.zero_grad()
                output = self.model(x_batch, stock_ids).squeeze()
                loss = criterion(output, y_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for x_batch, y_batch, stock_ids in val_loader:
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)
                        stock_ids = stock_ids.to(device)
                        
                        output = self.model(x_batch, stock_ids).squeeze()
                        loss = criterion(output, y_batch)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                else:
                    patience_counter += 1
            
            # Store training history
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss if val_loader else None,
                'lr': optimizer.param_groups[0]['lr']
            }
            self.training_history.append(epoch_info)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                if val_loader:
                    print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
                else:
                    print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping check
            if val_loader and patience_counter >= max_patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
        print("✅ Training completed!")
        
        # Load best model if validation was used
        if save_path and val_loader and os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path))
            print(f"📁 Best model loaded from {save_path}")
    
    def evaluate(self, test_dataset, batch_size: int = 32, device: str = 'cpu'):
        """
        Evaluate the model on test data
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_actuals = []
        all_stock_ids = []
        
        with torch.no_grad():
            for x_batch, y_batch, stock_ids in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                stock_ids = stock_ids.to(device)
                
                output = self.model(x_batch, stock_ids).squeeze()
                
                all_preds.extend(output.cpu().numpy())
                all_actuals.extend(y_batch.cpu().numpy())
                all_stock_ids.extend(stock_ids.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)
        all_stock_ids = np.array(all_stock_ids)
        
        return all_preds, all_actuals, all_stock_ids
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.training_history:
            print("No training history available")
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        val_losses = [h['val_loss'] for h in self.training_history if h['val_loss'] is not None]
        
        plt.figure(figsize=(12, 4))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss', color='blue')
        if val_losses:
            plt.plot(epochs[:len(val_losses)], val_losses, label='Val Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(1, 2, 2)
        lrs = [h['lr'] for h in self.training_history]
        plt.plot(epochs, lrs, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
