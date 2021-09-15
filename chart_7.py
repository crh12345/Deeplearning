import torch
import matplotlib.pyplot as plt
import numpy as np
"""
开始搭建基础的模型类 设置结构 激活函数 和前向传播方式
"""
class  Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(8,1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.sigmoid(self.linear(x))
        return x
model = Model()

