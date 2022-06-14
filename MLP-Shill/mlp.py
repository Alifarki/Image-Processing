import torch
import torch.nn as nn
from CustomDataset import custom_dataset_Shill
from CustomModel import Net
lr=0.001
num_epoch = 1000
number_of_features=11
number_of_out=2

#(model)
my_model = Net(number_of_features,number_of_out)
mydataset = custom_dataset_Shill('C:/Users/Ali/Desktop/Shill.csv')
number_of_train=(0.7*len(mydataset)).__round__()
number_of_test=(0.1*len(mydataset)).__round__()
number_of_valid=(0.2*len(mydataset)).__round__()
from torch.utils.data import random_split, DataLoader
custom_train, custom_valid, custom_test = random_split(mydataset, [number_of_train,number_of_valid,number_of_test])

dataloader = DataLoader(custom_train, batch_size=50, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(my_model.parameters(), lr=lr)
loss_total = list()
for epoch in range(1000):
    for data_batch, label_batch in dataloader:
        optimizer.zero_grad()
        out = my_model(data_batch)
        loss = criterion(out, label_batch)
        loss.backward()
        optimizer.step()

    if (epoch) % 1 == 0:
        out = my_model(mydataset.x[custom_valid.indices, :number_of_features])
        _, predicted = torch.max(out.data, 1)
        label_v = mydataset.y[custom_valid.indices]
        loss_v = criterion(out, label_v)
        print('Epoch [%d/%d] Train Loss: %.4f' % (epoch + 1, num_epoch, loss.item()))
        print('Epoch [%d/%d] Valid Loss: %.4f' % (epoch + 1, num_epoch, loss_v.item()))
        acc = (100 * torch.sum(label_v == predicted) / number_of_valid)
        print('Accuracy of the network in Validation %.4f %%' % acc)
        loss_total.append([loss, loss_v])

X=mydataset.x[custom_test.indices]
Y=mydataset.y[custom_test.indices]
# calculate out
out = my_model(X)
_, predicted = torch.max(out.data, 1)

#get accuration
print('Accuracy of the network %.4f %%' % (100 * torch.sum(Y==predicted) / number_of_test))
