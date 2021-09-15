import torch
import torch.nn as nn
nn.LSTM()
#prepared dataset and convert one-hots vertor
batch_size = 1
seq_len = 4
input_size  = 4
hideen_size = 4
idx2char = ['e','h','l','o']
x_data = [1,0,2,2,3]
y_data = [3,1,2,3,2]

one_hot_lookup =[
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
]
x_one_hot = [one_hot_lookup[x] for x in x_data]
#-1表示维度自动判断
inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
lables = torch.LongTensor(y_data).view(-1,1)

#design RNN model
class  Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.rnncell = torch.nn.RNNCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size
        )
    def forward(self,inpt,hidden):
        hidden = self.rnncell(inpt,hidden)
        return hidden
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)
net = Model(input_size,hideen_size,batch_size)

#优化器和损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.1)

#开始进行训练
"""
一共十五次迭代训练
每次训练样本里数据的个数
"""
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()#先将梯度归零
    hidden = net.init_hidden() #初始化最开始的h0隐藏层
    print('predicted string:',end='')
    for input,label in zip(inputs,lables):
        hidden = net(input,hidden)# 开始进行训练步骤
        loss += criterion(hidden,label)#计算损失
        x,idx = hidden.max(dim = 1) #x表示的是最大的数值 idx 表示的该位置的下标
        print(idx2char[idx.item()],end='')
    loss.backward()
    optimizer.step()
    print('epoch:',epoch+1,'loss:',loss.item())
#经过上面的输出 这个模型已经是训练ok了
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs,label)
    loss.backward()

    optimizer.step()
    _,idx = outputs.max(dim = 1)
    idx = idx.data.numpy
    print('predictedL',''.join([idx2char[x] for x in idx]),end='')
    print('Epoch[%d/15] loss=%.3f'%(epoch+1,loss.itme()))
#学习lstm 里面相关的结构 遗忘门等