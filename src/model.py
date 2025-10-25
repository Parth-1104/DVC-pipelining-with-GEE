"""
Transformer-based model for water quality prediction
Based on HydroTransNet architecture (2025)
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Add positional information to input embeddings"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class HydroTransNet(nn.Module):
    """
    Transformer-based water quality prediction model
    """
    
    def __init__(
        self,
        input_dim=7,           # Number of input features (spectral bands + indices)
        d_model=128,           # Embedding dimension
        nhead=8,               # Number of attention heads
        num_encoder_layers=4,  # Number of transformer encoder layers
        dim_feedforward=512,   # Dimension of feedforward network
        dropout=0.1,
        output_dim=3           # Number of output parameters (TSS, Turbidity, Chlorophyll)
    ):
        super(HydroTransNet, self).__init__()
        
        self.d_model = d_model
        
        
        self.embedding = nn.Linear(input_dim, d_model)
        
       
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
       
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        
        self.fc1 = nn.Linear(d_model, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None):
        """
        Args:
            src: Input tensor of shape [seq_len, batch_size, input_dim]
            src_mask: Optional attention mask
        
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        
        
        src = self.pos_encoder(src)
        
       
        output = self.transformer_encoder(src, src_mask)
        
       
        output = output[-1, :, :]
        
        # Pass through output layers
        output = self.fc1(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc_out(output)
        
        return output


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

