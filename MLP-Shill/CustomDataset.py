from torch.utils.data.dataset import Dataset
import torch
import pandas as pd

class custom_dataset_Shill(Dataset):
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        del self.data['Bidder_ID']
        self.data = self.data.apply(pd.to_numeric)
        self.data = self.data.values
        self.x = torch.Tensor(self.data[:, :11]).float()
        self.y = torch.Tensor(self.data[:,  11]).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (self.x[idx,:], self.y[idx])
        return sample
