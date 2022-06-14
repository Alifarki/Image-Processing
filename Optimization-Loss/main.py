# """
# SECTION 1 : Load and setup data for training
# the datasets separated in two files from originai datasets:
# iris_train.csv = datasets for training purpose, 80% from the original data
# iris_test.csv  = datasets for testing purpose, 20% from the original data
# """
# import pandas as pd
# import random
# #load
# datatrain = pd.read_csv("D:\Deep learning\Deep Learning\Optimization\mlp2\Datasets\iris\iris_train.csv")
#
# #change string value to numeric
# datatrain.loc[datatrain['species']=='Iris-setosa', 'species']=0
# datatrain.loc[datatrain['species']=='Iris-versicolor', 'species']=1
# datatrain.loc[datatrain['species']=='Iris-virginica', 'species']=2
# datatrain = datatrain.apply(pd.to_numeric)
#
# #change dataframe to array
# datatrain_array = datatrain.values
#
#
#
# import numpy as np
# idx_valid = np.random.randint(0, 120, 20) #اینجا بخاطر تکراری شدن داده های ایجاد کرده مشکل ایجاد می شود
# idx_train = list()
# for i in range(120):
#     if not i in idx_valid:
#         idx_train.append(i)
# datavalid_array = datatrain_array[idx_valid,:]
# #split x and y (feature and target)
# xvalid = datavalid_array[:,:4]
# yvalid = datavalid_array[:,4]
#
# datatrain_array = datatrain_array[idx_train,:]
# #split x and y (feature and target)
# xtrain = datatrain_array[:,:4]
# ytrain = datatrain_array[:,4]
# """
# SECTION 2 : Build and Train Model
# Multilayer perceptron model, with one hidden layer.
# input layer : 4 neuron, represents the feature of Iris
# hidden layer : 10 neuron, activation using ReLU
# output layer : 3 neuron, represents the class of Iris
# optimizer = stochastic gradient descent with no batch-size
# loss function = categorical cross entropy
# learning rate = 0.01
# epoch = 500
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# torch.manual_seed(1234)
#
# #hyperparameters
# hl = 10
# lr = 0.01
# num_epoch = 500
#
# #build model
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(4, hl)
#         self.fc2 = nn.Linear(hl, 3)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
# net = Net()
#
# #choose optimizer and loss function
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#
# #train
# for epoch in range(num_epoch):
#     X = torch.Tensor(xtrain).float()
#     Y = torch.Tensor(ytrain).long()
#
#     Xv = torch.Tensor(xvalid).float()
#     Yv = torch.Tensor(yvalid).long()
#
#     #feedforward - backprop
#     optimizer.zero_grad()
#     out = net(X)
#     loss = criterion(out, Y)
#     loss.backward()
#     optimizer.step()
#
#     if (epoch) % 50 == 0:
#         print ('Epoch [%d/%d] Loss: %.4f'
#                    %(epoch+1, num_epoch, loss.item()))
#         out = net(Xv)
#         print(out)
#         _, predicted = torch.max(out.data, 1)
#         # print(_)
#         # acc = (100 * torch.sum(Yv == predicted) / 20)
#         print('Accuracy of the network in Validation %d %%' % (100 * torch.sum(Yv == predicted) / 20))

import torch
a=torch.tensor([1,2,3,0.5,0.9])
print(a)
pred=torch.max(a.data)