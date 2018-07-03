import torch
import numpy as np
from getdataTest import getfootdata

x = getfootdata("sweden","switzerland")
x = np.array(x).T
x = torch.autograd.Variable(torch.tensor(x,dtype=torch.float32))



model = torch.load("model.tar")
model.eval()
y_predict = model.forward(x)

print(y_predict.data)
