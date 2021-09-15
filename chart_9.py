import torch
# y = torch.LongTensor([0])
# z = torch.Tensor([[0.2,0.1,-0.1]])
# criterion = torch.nn.CrossEntropyLoss()
# loss = criterion(z,y)
# print(loss)
criterion = torch.nn.CrossEntropyLoss()

Y = torch.LongTensor([2,0,1])

Y_pred1 = torch.Tensor([
    [0.1,0.2,0.9],
    [1.1,0.1,0.2],
    [0.2,2.1,0.1]
])
Y_pred2 = torch.Tensor([
    [0.8,0.2,0.3],
    [0.2,0.3,0.5],
    [0.2,0.2,0.5]
])

l1 = criterion(Y_pred1,Y)
l2 = criterion(Y_pred2,Y)
print("Batch loss1 = ",l1.data,"\nbatch loss2 = ",l2.data)

