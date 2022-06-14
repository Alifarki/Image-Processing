import numpy
import torch
N,D_in,H,H1,D_out=10000,100,100,100,10
x=torch.randn(N,D_in,device=torch.device("cpu"),dtype=torch.float)
y=torch.randn(N,D_out,device=torch.device("cpu"),dtype=torch.float)
w1=torch.randn(D_in,H,device=torch.device("cpu"),dtype=torch.float)
w2=torch.randn(H,H1,device=torch.device("cpu"),dtype=torch.float)
w3=torch.randn(H1,D_out,device=torch.device("cpu"),dtype=torch.float)
lr=1e-6
for t in range(500):
    h=x.mm(w1)
    h_relu=h.clamp(min=0)
    h1=h_relu.mm(w2)
    h1_relu=h1.clamp(min=0)
    y_pred=h1_relu.mm(w3)
    loss=(y_pred-y).pow(2).sum().item()
    print(t,loss)
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w3 = h1_relu.t().mm(grad_y_pred)
    grad_h1_relu = grad_y_pred.mm(w3.t())
    grad_h1 = grad_h1_relu.clone()
    grad_h1[h1 < 0] = 0
    grad_w2 = h_relu.t().mm(grad_h1)
    grad_h_relu = grad_h1.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
    w3 -= lr * grad_w3



