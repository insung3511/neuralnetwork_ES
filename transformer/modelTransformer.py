import torch.nn as nn
import torch

import numpy as np
import random

device = 'mps'
def set_seed(seed: int = 0):
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def validate_model_output(model, input_x, output_y):
    model.eval()
    with torch.no_grad():
        preds = model(input_x)
        assert preds.shape == output_y.shape, f"Model output shape: {preds.shape}, expected: {output_y.shape}"
    return True

# Transformer model
class TransformerModule(nn.Module):
    def __init__(self, 
                 sequence_length=10,
                 input_dim=64,
                 model_dim=64, 
                 num_heads=4, 
                 num_layers=2, 
                 ff_dim=256, 
                 output_dim=1):
        super(TransformerModule, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, sequence_length, model_dim))
        self.input_fc = nn.Linear(input_dim, model_dim)
        latent_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer = nn.TransformerEncoder(latent_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.input_fc(x)
        x = x + self.positional_encoding
        x = x.permute(1, 0, 2)  # Transformer expects (S, N, E)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to (N, S, E)
        x = self.output_fc(x)
        return x