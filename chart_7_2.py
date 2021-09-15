import numpy as np
import torch
import matplotlib.pyplot as plt
"""
开始读取数据
"""
xy = np.loadtxt('diabetes.csv.gz',delimiter=',',dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])

"""
建立模型
"""
class MOdel(torch.nn.Module):
    def __init__(self):
        super(MOdel, self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmod = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.sigmod(self.linear1(x))
        x = self.sigmod(self.linear2(x))
        x = self.sigmod(self.linear3(x))
        return x
model= MOdel()
"""
构造损失函数和优化器
"""
criterion = torch.nn.BCELoss(size_average=True)
#SGD随机梯度下降
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)

#开始训练模型
for epoch in range(100):
    #forward
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print('epoch= ',epoch,'loss=',loss.item(),'y_pred = ',torch.mean(y_pred),'y_data=',torch.mean(y_data))

    #backward
    optimizer.zero_grad()
    loss.backward()

    #update
    optimizer.step()

