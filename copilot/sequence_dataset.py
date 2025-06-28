# Purpose: Enhanced datasets for single and multi-stock LSTM training

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class StockSequenceDataset(Dataset):
    """
    Enhanced dataset for single stock time series prediction
    """
    def __init__(self, df, target_col='close', seq_len=20, return_dates=False):
        self.seq_len = seq_len
        self.target_col = target_col
        self.return_dates = return_dates
        
        # Store dates if requested
        if return_dates and hasattr(df, 'index'):
            self.dates = df.index.values
        else:
            self.dates = None
        
        # Prepare features and targets
        if isinstance(df, pd.DataFrame):
            feature_cols = [col for col in df.columns if col != target_col]
            self.data = df[feature_cols].values
            self.targets = df[target_col].values
        else:
            # Assume numpy array input
            target_idx = -1 if target_col == 'close' else 0  # Simple fallback
            self.data = np.delete(df, target_idx, axis=1)
            self.targets = df[:, target_idx]
        
        self.feature_names = feature_cols if isinstance(df, pd.DataFrame) else None

    def __len__(self):
        return len(self.targets) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        
        if self.return_dates and self.dates is not None:
            date = self.dates[idx + self.seq_len]
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), date
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class MultiStockDataset(Dataset):
    """
    Dataset for multi-stock time series prediction with stock embeddings
    """
    def __init__(self, df, stock_id, target_col='Close', seq_len=20):
        self.seq_len = seq_len
        self.stock_id = stock_id
        
        # Prepare features and targets
        feature_cols = [col for col in df.columns if col != target_col]
        self.data = df[feature_cols].values
        self.targets = df[target_col].values
        self.feature_names = feature_cols

    def __len__(self):
        return len(self.targets) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        stock_id = self.stock_id
        
        return (
            torch.tensor(x, dtype=torch.float32), 
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(stock_id, dtype=torch.long)
        )

class AdvancedStockDataset(Dataset):
    """
    Advanced dataset with support for multiple prediction horizons and features
    """
    def __init__(self, df, target_col='Close', seq_len=20, pred_horizon=1, 
                 include_volume=True, include_technical=True):
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        
        # Feature selection
        base_features = ['Open', 'High', 'Low', 'Close']
        if include_volume and 'Volume' in df.columns:
            base_features.append('Volume')
        
        # Add technical indicators if available
        if include_technical:
            tech_features = [col for col in df.columns 
                           if any(indicator in col.lower() 
                                 for indicator in ['ma_', 'rsi', 'macd', 'bb_', 'return'])]
            base_features.extend(tech_features)
        
        # Remove target from features
        feature_cols = [col for col in base_features if col != target_col]
        
        self.data = df[feature_cols].values
        self.targets = df[target_col].values
        self.feature_names = feature_cols

    def __len__(self):
        return len(self.targets) - self.seq_len - self.pred_horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len + self.pred_horizon - 1]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
