import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedLSTMModel(nn.Module):
    """
    Enhanced LSTM model with dropout, batch normalization, and residual connections
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, 
                 output_size=1, use_attention=False):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size, 
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
        
        # Fully connected layers with residual connections
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
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
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

class MultiStockLSTM(nn.Module):
    """
    LSTM model designed to handle multiple stocks with stock embeddings
    """
    def __init__(self, input_size, num_stocks, hidden_size=128, num_layers=2, 
                 dropout=0.2, embedding_dim=16):
        super().__init__()
        
        self.num_stocks = num_stocks
        self.embedding_dim = embedding_dim
        
        # Stock embedding layer
        self.stock_embedding = nn.Embedding(num_stocks, embedding_dim)
        
        # Combined input size (features + embedding)
        combined_input_size = input_size + embedding_dim
        
        # LSTM with combined input
        self.lstm = nn.LSTM(
            combined_input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.stock_embedding.weight)
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x, stock_ids):
        batch_size, seq_len, _ = x.shape
        
        # Get stock embeddings
        stock_emb = self.stock_embedding(stock_ids)  # (batch_size, embedding_dim)
        
        # Expand embeddings to match sequence length
        stock_emb_expanded = stock_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate features with stock embeddings
        x_combined = torch.cat([x, stock_emb_expanded], dim=-1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x_combined)
        
        # Take last time step and pass through FC layers
        out = self.fc(lstm_out[:, -1, :])
        
        return out
