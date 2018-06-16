import numpy as np

class NN():
    def __init__(self,x_siz, y_siz, n_epochs, l_r):
        self.X =list()
        self.Y = list()
        self.learning_rate = l_r
        self.x_siz = x_siz
        self.y_siz = y_siz
        self.weight = list()
        self.dweight = list()
        self.bias= list()
        self.dbias= list()
        self.n_epochs = n_epochs
        self.A = list()
        self.Z = list()
        self.Yl = list()

    def initiazeparameter(self,Hlayer_dim):
        Hlayer_dim.insert(0,x_siz)
        Hlayer_dim.append(y_siz)
        for i in range(len(Hlayer_dim)-1):
            self.weight.append(np.random.randn(Hl[i+1], Hl[i])* 0.01)
            self.bias.append(np.random.randn(Hl+1) * 0.01)
        

    def getdata():
        
    def normalizedata():
    
    def sigmoid(x, deri = False):
        if deri is False:
            return 1.0/(1+np.exp(-x))
        else: 
            return x*(1-x)

    def ReLU(x,deri=False):
        if deri is False:
            return x[x<0] = 0
        else:
            x[x<=0] = 0
            x[x>0] = 1
            return x

    def activation(a,w,b,activation):
        z = np.dot(w,a) + b
        self.Z.append(z)
        if activation=="relu":
            return ReLU(z)
        elif activation == "sigmoid":
            return sigmoid(z)
        

    def feedforward(self):
        A_data = self.X
        for l in range(len(self.bias)):
                if(l==len(self.bias)+1):
                     self.A.append(activation(A_data,self.weight[l],self.bias[l]),"sigmoid")
                     self.Yl = self.A[l]
                     continue
            self.A.append(activation(A_data,self.weight[l],self.bias[l],"relu"))
            A_data = self.A[l]
        
        
    def calc_cost():
        m = self.Y.shape[1]
        cost = (-1 / m) * np.sum(np.multiply(self.Y, np.log(self.Yl)) + np.multiply(1 - self.Y, np.log(1 - self.Yl)))
        return cost
    
    def back_activation(x,acti):
        if acti=="sigmoid":
            return sigmoid(x,True)
        else acti=="relu":
            return ReLU(x,True)

    
    def linear_back(self,a,w,b,acti):
        m = a.shape[1]
        dz = back_activation(a,acti)
        dW = np.dot(dz, a.T) / m
        db = np.squeeze(np.sum(dz, axis=1, keepdims=True)) / m
        self.dweight.insert(0,dW)
        self.dbias.insert(0,db)
        dA = np.dot(w.T, dZ)
        return dA


    def backpropagation():
        m = len(self.bias)
        dAL = - (np.divide(self.Y, self.Yl) - np.divide(1 - self.Y, 1 -self.Yl))
        dA = linear_back(dAL,self.weight[m],self.bias[m],"sigmoid")
        for l in reversed(range(m)-1):
            dA = linear_back(dA,self.weight[l],self.bias[l],"relu")


    def update_parameters():
        m = len(self.bias)
        for l in range(m):
            self.weight[l] = self.weight[l] - learning_rate * self.dweight[l]
            self.bias[l] = self.bias[l] - learning_rate * self.dbias[l]
        
        self.Z = []
        self.A = []
        self.dweight = []
        self.dbias = []
        

    def saveweights():

    def loadweights():