# Purpose: Slice engineered data into sequences for LSTM

import torch
from torch.utils.data import Dataset

class StockSequenceDataset(Dataset):
    def __init__(self, df, target_col='close', seq_len=20):
        self.seq_len = seq_len
        self.target_col = target_col

        self.data = df.drop(columns=[target_col]).values
        self.targets = df[target_col].values

    def __len__(self):
        return len(self.targets) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
