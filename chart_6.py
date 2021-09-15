import torch.nn.functional as F
import torch
"""
1step prepare dataset
2step design model using class
3step construct loss and optimizer
4step training cycle
5step print relative info
"""
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])
class LogistRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogistRegressionModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
criterion = torch.nn.BCELoss(size_average=False)

model  = LogistRegressionModel()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
import numpy as np
import  matplotlib.pyplot as plt
x = np.linspace(0,10,200)
x_t = torch.Tensor(x).view((200,1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel("Hours")
plt.ylabel("Probability of passw")
plt.grid()
plt.show()