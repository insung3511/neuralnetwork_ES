# Transformer dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import torch

# This dataset compact for LLM training set, generates random sequences
class RandomSequenceDataset(Dataset):
    def __init__(self, 
                 num_samples=1000, 
                 sequence_length=10, 
                 feature_dim=64):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.data = torch.randn(num_samples, sequence_length, feature_dim)
        self.labels = torch.randn(num_samples, sequence_length, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_dataloader(batch_size=32, num_samples=1000, sequence_length=10, feature_dim=64, dataloader_shuffle=True):
    dataset = RandomSequenceDataset(num_samples=num_samples, sequence_length=sequence_length, feature_dim=feature_dim) 
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=dataloader_shuffle)
    return dataloader