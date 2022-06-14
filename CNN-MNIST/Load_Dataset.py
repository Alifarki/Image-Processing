import torch
import torchvision
from torchvision import transforms
import cv2
from matplotlib import pyplot as plt
import numpy as np
#Parameters
batch_size=64
#Load Dataset
# train_dataset=torchvision.datasets.MNIST(root='./Dataset/',train=True,
#                                         transform=transforms.Compose([
#                                         transforms.Resize((14,14)),transforms.ToTensor()])
#                                         ,download=True)
# test_dataset=torchvision.datasets.MNIST(root='./Dataset/',train=False,
#                                         transform=transforms.ToTensor()
#                                         ,download=True)
#
# train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
# test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


#Main
# for data , label in train_loader:
    # print(data,label)

#load Dataset Custom

train_dataset_mnist=torchvision.datasets.ImageFolder(root=r'D:\Deep learning\Deep Learning\CIFAR-10-images-master/train',
                                                       transform=transforms.ToTensor())
train_loader_mnist=torch.utils.data.DataLoader(dataset=train_dataset_mnist,batch_size=batch_size,shuffle=True)




