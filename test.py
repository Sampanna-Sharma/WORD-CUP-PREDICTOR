import matplotlib as mt
from NN import *

neural = NN(132,3,0.005)
neural.getdata()
neural.initializeparameter([60,20,5])
for i in range(1000):
    neural.feedforward()
    if(i%10==0):
        print(i,neural.calc_cost())
    neural.backpropagation()
    #print(neural.gradientcheck())
    neural.update_parameters()
    
print(neural.Yl)
# total = 0
# for m,n in zip(neural.Y.argmax(axis=0),neural.Yl.argmax(axis=0)):
#     total = total + 1
#     print(total,m,n)
#     if m == n:
#         print("true")   
    

