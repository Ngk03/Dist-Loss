import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
    """
    Create custom dataset
    """
    def __init__(self, data, labels, label_density=None):
        self.data = data
        self.labels = labels.reshape(-1,1)
        if label_density is not None:
            self.label_density = label_density.reshape(1,-1)
        else:
            self.label_density = np.empty((1,1))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = torch.tensor(self.data[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        ld = torch.tensor(self.label_density.copy(), dtype=torch.float32)
        return sample, label, ld