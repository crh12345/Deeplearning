"""
DataLoader：batch_size=2,shuffle = True
shuffle :直接开始打乱数据
batch_size: 设置一个batch的大小
Dataset 是一个抽象类 需要使用类来进行继承实现
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#构造加载数据集的类
class DiabetesDataset(Dataset):
    #进行初始化操作
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])
    #获取相应的下标
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    #得到本身的数据长度
    def __len__(self):
        return self.len
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
#数据进行加载
dataset = DiabetesDataset('diabetes.csv.gz')
train_load = DataLoader(dataset=dataset,batch_size=32,shuffle=True
                        ,num_workers=2)

#training code
for epoch in range(100):
    for i ,data in enumerate(train_load,0):
        #prepare data
        inputs,labels = data
        #forward
        y_pred = model(inputs)
        loss = criterion(y_pred,labels)
        print('epoch = ',epoch ,"i = ",i ,'loss=',loss.item())
        #backward
        optimizer.zero_grad()
        loss.backward()
        #data update
        optimizer.step()
