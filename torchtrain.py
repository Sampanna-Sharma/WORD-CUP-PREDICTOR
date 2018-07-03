import torch
import numpy as np
from getdata import getfootdata

l_r = 1e-6

Layer = [308, 150,50,9, 3]

x,y = getfootdata()
x = np.array(x).T
y = np.array(y).T
x = torch.autograd.Variable(torch.tensor(x,dtype=torch.float32))
y = torch.autograd.Variable(torch.tensor(y,dtype=torch.float32))
target = torch.tensor(torch.max(y, 1)[1],dtype=torch.long)

print(x.size())
print(y.size())
print(target.size())

model = torch.nn.Sequential(
    torch.nn.Linear(Layer[0],Layer[1],bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(Layer[1],Layer[2],bias=True),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(Layer[2],Layer[3],bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(Layer[3],Layer[4],bias=True),
    torch.nn.Softmax()
    #torch.nn.LogSoftmax()
)
optimizer = torch.optim.Adam(model.parameters(), lr=l_r)
loss_fn = torch.nn.MSELoss()
#loss_fn = torch.nn.CrossEntropyLoss()

for i in range(100000):
    y_predict = model(x)
    loss = loss_fn(y_predict,y)
    #loss = loss_fn(y_predict,target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%100==0:
         print(i,loss.item())

torch.save(model,"model.tar")


