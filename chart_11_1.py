"""
卷积神经网络RNN
"""
import torch
batch_size = 1
seq_len = 3
input_size  = 4
hideen_size = 2

#设置RNN
num_layers = 1
cell = torch.nn.RNN(
    #参数说明 input_size 输入的size hidden 输出的size num_layers表示为
    #共几层这样的RNNCell
    input_size=input_size,hidden_size=hideen_size,num_layers=num_layers
)

inputs = torch.randn(seq_len,batch_size,input_size)
hidden = torch.zeros(num_layers,batch_size,hideen_size)

#RNN的
out,hidden = cell(inputs,hidden)
print('output size:',out.shape)
print('output:',out)
print('Hidden size:',hidden.shape)
print('Hidden:',hidden)

