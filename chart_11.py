"""
卷积神经网络RNNCell
"""
import torch
batch_size = 1
seq_len = 3
input_size  = 4
hideen_size = 2
cell = torch.nn.RNNCell(
    #input_size 是输入层的维度，hideen_size 表示的是隐藏层的维度
    input_size=input_size,hidden_size=hideen_size
)

#这里随机
dataset = torch.randn(seq_len,batch_size,input_size)
hidden = torch.zeros(batch_size,hideen_size)


for idx,input in enumerate(dataset):
    print('='*20,idx,'='*20)
    print('inputsize :',input.shape)
    hidden = cell(input,hidden)

    print('outputs size:',hidden.shape)
    print(hidden)