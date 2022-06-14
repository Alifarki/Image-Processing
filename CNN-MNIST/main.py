import torch
import torchvision
from torchvision import transforms
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
#Parameters
n_class=10
batch_size=64
lr=0.01
num_epoch=1
train_dataset=torchvision.datasets.ImageFolder(root=r'D:\Deep learning\Deep Learning\MNIST Dataset JPG format/train',
                                                       transform=transforms.ToTensor())
test_dataset=torchvision.datasets.ImageFolder(root=r'D:\Deep learning\Deep Learning\MNIST Dataset JPG format/test',
                                                       transform=transforms.ToTensor())
valid_dataset=torchvision.datasets.ImageFolder(root=r'D:\Deep learning\Deep Learning\MNIST Dataset JPG format/valid',
                                                       transform=transforms.ToTensor())
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

valid_loader=torch.utils.data.DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True)

#Convolutional neural network
class convnet(nn.Module):
    def __init__(self):
        super(convnet,self).__init__() #انتصاب init به Module
        self.layer1=nn.Sequential(nn.Conv2d(3, 16, 3, 1, 2),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2 ,2)) #راه حل جایگزین برای اینکه همه توایع پشت سرهم و در یک تابع اجرا شوند
        # self.conv1=nn.Conv2d(1, 16, 3, 1, 2)
        # self.BatchN1=nn.BatchNorm2d(16)
        # self.relu1=nn.ReLU()
        # self.maxp1=nn.MaxPool2d(2 ,2)
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2, 2))
        # self.conv2=nn.Conv2d(16, 32, 3, 1, 2)
        # self.BatchN2 = nn.BatchNorm2d(32)
        # self.relu2 = nn.ReLU()
        # self.maxp2 = nn.MaxPool2d(2, 2)
        self.fc=nn.Linear(8*8*32,n_class) #دوتا پولینگ داشتیم که تصویر 7*7 شد و 32 تا کانال داشتیم.
    def forward(self,x):
        out1=self.layer1(x)
        out2=self.layer2(out1) #خروجی این قسمت 8*8*32*64
        #میبایست Reshape شود به عبارتی 64 ثابت باشد و باقی در هم ضرب شوند.به عبارتی 64*2048 شود

        out2=out2.reshape(out2.size(0),-1) #سایز0 در واقع همون 64 یا تعداد بچ هاست.
        #منفی یک یعنی باقی مهم نیست و ضرب کن
        y=self.fc(out2)
        return y
        # a=self.conv1(x)
        # a2=self.BatchN1(a)
        # a3=self.relu1(a2)
        # a4=self.maxp1(a3)
        # a5=self.conv2(a4)
        # a6=self.BatchN2(a5)
        # a7=self.relu2(a6)
        # y=self.maxp2(a7)
        # return y
convmodel=convnet()
#loss
loss_fn=nn.CrossEntropyLoss()
#optimizer
optimizer_fn=torch.optim.Adam(convmodel.parameters(),lr=lr)

#LR
lr_sch=torch.optim.lr_scheduler.StepLR(optimizer_fn,5,gamma=0.1) #هر5 تا step نرخ یادگیری رو تغییر میکند. و گاما در نرخ یادگیری ضرب می شود.تنظیم نرخ یادگیری مختلف
num_steps=len(train_loader)
valid_num_steps = len(valid_loader)
for i in range(num_epoch):
    convmodel.train()
    lr_sch.step() #اینجا شروع میکند به شمردن که وقتی 5 بار تکرار شد ضرب در گاما انجام می شود.
    for j ,(imgs, lbls) in enumerate(train_loader):
        out = convmodel(imgs)
        loss_val = loss_fn(out, lbls)
        optimizer_fn.zero_grad()
        loss_val.backward()
        optimizer_fn.step()
        if (j+1)%2 == 0:
            print('Epoch [{}/{}] Step [{}/{}] Loss {:.4f}'.format(i+1 , num_epoch , j+1 , num_steps ,loss_val.item()))
        if j==36:
            break
        convmodel.eval()
        corrects = 0
        for k, (imgs, lbls) in enumerate(valid_loader):
            out = convmodel(imgs)
            loss_val = loss_fn(out, lbls)
            predicted = torch.argmax(out, 1)
            corrects += torch.sum(predicted == lbls)
            print('Validation, Step [{}/{}] Loss: {:.4f} Acc: {:.4f} '.format(k + 1, valid_num_steps, loss_val.item(),
                                                                              100. * corrects / ((k + 1) * batch_size)))

convmodel.eval()
num_steps=len(test_loader)
correct=0
for j ,(imgs, lbls) in enumerate(test_loader):
    out = convmodel(imgs)
    predict=torch.argmax(out,1)
    correct += torch.sum(predict== lbls)
    print('Step [{}/{}] Acc:{:.4f}'.format(i+1 , num_epoch ,100.*correct/((j+1)*batch_size)))


