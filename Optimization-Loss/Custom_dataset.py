from torch.utils.data.dataset import Dataset
import torch

import pandas as pd

class custom_dataset(Dataset):
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)
        # change string value to numeric
        self.data.loc[self.data['species'] == 'Iris-setosa', 'species'] = 0
        self.data.loc[self.data['species'] == 'Iris-versicolor', 'species'] = 1
        self.data.loc[self.data['species'] == 'Iris-virginica', 'species'] = 2
        self.data = self.data.apply(pd.to_numeric)
        # change dataframe to array
        self.data = self.data.values
        self.x = torch.Tensor(self.data[:, :4]).float()
        self.y = torch.Tensor(self.data[:,  4]).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = (self.x[idx,:], self.y[idx])
        return sample