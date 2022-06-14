import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
hl = 1000

#build model
class Net(nn.Module):

    def __init__(self, number_of_features,number_of_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(number_of_features, hl)
        self.fc2 = nn.Linear(hl, number_of_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(4, hl)
        self.fc2 = nn.Linear(hl, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x