import torch
x=torch.randn(10,3,device=torch.device("cpu"),dtype=torch.float)
y=torch.randn(10,2,device=torch.device("cpu"),dtype=torch.float)
fc=torch.nn.Linear(3,2)      #تعریف مقادیر w و b بصورت Backend
print("weights",fc.weight)
print("bias",fc.bias)
loss_func=torch.nn.MSELoss()
optim=torch.optim.SGD(fc.parameters(),lr=0.01)
for t in range(500):
    y_pred=fc(x)
    loss_value=loss_func(y,y_pred)
    print(t,loss_value.item())
    loss_value.backward()
    print(fc.weight.grad)
    print(fc.bias.grad)
    optim.step()
    print(fc.weight)


