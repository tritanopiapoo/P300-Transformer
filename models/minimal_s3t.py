import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class MinimalS3T(nn.Module):
    def __init__(self, n_channels=3, n_timesteps=250, d_model=64, n_classes=16):  # добавили n_classes
        super().__init__()
        self.spatial_proj = nn.Linear(n_channels, d_model)

        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):

        batch_size, seq_len, n_channels, n_timesteps = x.shape

        x = x.view(batch_size * seq_len, n_channels, n_timesteps)
        x = x.permute(0, 2, 1)  
        x = self.spatial_proj(x)  
        x = self.pos_encoding(x)
        x = self.transformer(x)  
        x = x.mean(dim=1)  

        x = x.view(batch_size, seq_len, -1) 
        x = x.mean(dim=1)  
        return self.classifier(x)  


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)
