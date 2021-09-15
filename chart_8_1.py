"""
作业：地址https://www.kaggle.com/c/titanic/data
给出的是三个数据
gender_submission.csv:保存的是测试集中 人员id的结果y_data
test.csv:保存的所有的数据特征存储
train.csv：里面不仅仅有test.csv保存的数据特征 还有是否survive
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#定义加载类
class DiabetesDataset(Dataset):
    #进行初始化操作
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)

        self.len = xy.shape[0]
        #还得先对数据进行预处理阶段 如名字 什么的可以直接先设置成空或者直接删掉 还有船舱之类的先设置成空 预处理完后
        #需要删除的列有 名字 性别需要替换成 0,或者1 Ticket需要删除 Cabin 也需要删除 因为需要的是 线性模型 所以非线性数据都需要做一次处理操作
        xy = np.delete(xy, 3,axis=1)
        xy = np.rep
    #获取相应的下标
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    #得到本身的数据长度
    def __len__(self):
        return self.len