import numpy as np
import random

import pandas as pd

datatrain=pd.read_csv("D:\Deep learning\Deep Learning\Optimization\mlp2\Datasets\iris\iris_train.csv")
datatrain.loc[datatrain['species']=='Iris-setosa','species']=0
datatrain.loc[datatrain['species']=='Iris-versicolor','species']=1
datatrain.loc[datatrain['species']=='Iris-virginica','species']=2
# print(type(datatrain))
data=datatrain.apply(pd.to_numeric)
data_array=data.values
# print(type(datatrain))
idx_valid=random.sample(range(120), 20)
idx_valid=np.array(idx_valid)
idx_train = list()
for i in range(120):
    if not i in idx_valid:
        idx_train.append(i)
datavalid_array = data_array[idx_valid,:]
xvalid = datavalid_array[:,:4]
yvalid = datavalid_array[:,4]
datatrain_array = data_array[idx_train,:]
xtrain = datatrain_array[:,:4]
ytrain = datatrain_array[:,4]
#------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
hl = 10
lr = 0.01
num_epoch = 500
torch.manual_seed(1234)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(4,hl)
        self.fc2=nn.Linear(hl,3)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
net=Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
for epoch in range(num_epoch):
    X = torch.Tensor(xtrain).float()
    Y = torch.Tensor(ytrain).long()

    Xv = torch.Tensor(xvalid).float()
    Yv = torch.Tensor(yvalid).long()

    #feedforward - backprop
    optimizer.zero_grad()
    out = net(X)
    loss = criterion(out, Y)
    loss.backward()
    optimizer.step()

    if (epoch) % 50 == 0:
        print ('Epoch [%d/%d] Loss: %.4f' %(epoch+1, num_epoch, loss.item()))
        out = net(Xv)
        _, predicted = torch.max(out.data, 1)
        acc = (100 * torch.sum(Yv == predicted) / 20)
        print('Accuracy of the network in Validation %d %%' % (100 * torch.sum(Yv == predicted) / 20))


datatest = pd.read_csv("D:\Deep learning\Deep Learning\Optimization\mlp2\Datasets\iris\iris_test.csv")

#change string value to numeric
datatest.loc[datatest['species']=='Iris-setosa', 'species']=0
datatest.loc[datatest['species']=='Iris-versicolor', 'species']=1
datatest.loc[datatest['species']=='Iris-virginica', 'species']=2
datatest = datatest.apply(pd.to_numeric)

#change dataframe to array
# datatest_array = datatest.as_matrix()
datatest_array = datatest.values

#split x and y (feature and target)
xtest = datatest_array[:,:4]
ytest = datatest_array[:,4]

#get prediction
X = torch.Tensor(xtest).float()
Y = torch.Tensor(ytest).long()
out = net(X)
_, predicted = torch.max(out.data, 1)
print(predicted)
#get accuration
print('Accuracy of the network %d %%' % (100 * torch.sum(Y==predicted) / 30))


