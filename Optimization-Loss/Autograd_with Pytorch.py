import torch
x=torch.tensor(1.0,requires_grad=True)
w=torch.tensor(2.0,requires_grad=True)
b=torch.tensor(3.0,requires_grad=True)
y=x*w+b
print(y)
y.backward()
print(x.grad)
print(w.grad)
print(b.grad)
a=torch.tensor(5.0,requires_grad=True)
z=a**3
z.backward()
print(a.grad)

