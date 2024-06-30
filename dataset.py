import torch
import numpy as np
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels.reshape(-1,1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return sample, label
