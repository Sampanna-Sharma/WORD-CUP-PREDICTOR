import numpy as np
#from getdata import getfootdata
from getdatasmall import getfootdata

class NN():
    def __init__(self,x_siz, y_siz, l_r):
        self.X =list()
        self.Y = list()
        self.numtrain = int 
        self.numtest = int 
        self.cost = float
        self.Hl = list()
        self.learning_rate = l_r
        self.x_siz = x_siz
        self.y_siz = y_siz
        self.weight = list()
        self.dweight = list()
        self.bias= list()
        self.dbias= list()
        #self.n_epochs = n_epochs
        self.A = list()
        self.Z = list()
        self.Yl = list()

    def initializeparameter(self,Hlayer_dim):
        Hl = Hlayer_dim
        Hl.insert(0,self.x_siz)
        Hl.append(self.y_siz)
        for i in range(len(Hlayer_dim)-1):
            self.weight.append(np.random.randn(Hl[i+1], Hl[i])* 0.01)
            self.bias.append(np.random.randn(Hl[i+1]) * 0.01)
        

    def getdata(self):
        self.X, self.Y = getfootdata()
        self.numtrain = self.X.shape[1]
        
    def normalizedata(self):
        pass
    
    def sigmoid(self,x, deri = False):
        if deri is False:
            shiftx = x - np.max(x,axis=0)
            exps = np.exp(shiftx)
            return exps / np.sum(exps,axis=0)
        else: 
            return np.multiply(self.Yl,(1-self.Yl)) 

    def ReLU(self,x,deri=False):
       # if deri is False:
        #    return np.tanh(x)
        #else:
        #    return 1.0 - np.tanh(x)**2
        c = np.zeros_like(x)
        slope = 1e-2
        if deri:
            c[x<=0] = slope
            c[x>0] = 1
        else:
            c[x>0] = x[x>0]
            c[x<=0] = slope*x[x<=0]
        return c

    def activation(self,a,w,b,activation):
        b = b[:, np.newaxis]
        z = np.dot(w,a) + b
        self.Z.append(z)
        if activation=="relu":
            return self.ReLU(z)
        elif activation == "sigmoid":
            return self.sigmoid(z)
        

    def feedforward(self):
        A_data = self.X
        for l in range(len(self.bias)):
            if(l==len(self.bias)-1):
                self.A.append(self.activation(A_data,self.weight[l],self.bias[l],"sigmoid"))
                self.Yl = self.A[l]
                self.A.insert(0,self.X)
                continue
            self.A.append(self.activation(A_data,self.weight[l],self.bias[l],"relu"))
            A_data = self.A[l]
        
        
    def calc_cost(self):
        m = self.Y.shape[1]
        #print((np.multiply(self.Y,np.log(self.Yl))))
        self.cost = (-1 / m) * np.sum(np.multiply(self.Y,np.log(self.Yl)))
        return self.cost
    
    def back_activation(self,x,acti):
        if acti is "sigmoid":
            return self.sigmoid(x,True)
        elif acti is "relu":
            return self.ReLU(x,True)

    
    def linear_back(self,da,a,w,b,acti):
        m = da.shape[1]
        dz = self.back_activation(da,acti)
        dW = np.dot(dz, a.T) / m
        db = np.squeeze(np.sum(dz, axis=1, keepdims=True)) / m
        self.dweight.insert(0,dW)
        self.dbias.insert(0,db)
        dA = np.dot(w.T, dz)
        assert (dA.shape == a.shape)
        assert (dW.shape == w.shape)
        return dA


    def backpropagation(self):
        m = len(self.bias)
        #dAL = - (np.divide(self.Y, self.Yl) - np.divide(1 - self.Y, 1 -self.Yl))
        dAL = self.Yl-self.Y
        dA = self.linear_back(dAL,self.A[m-1],self.weight[m-1],self.bias[m-1],"sigmoid")
        for l in reversed(range(m-1)):
            dA = self.linear_back(dA,self.A[l],self.weight[l],self.bias[l],"relu")


    def update_parameters(self):
        m = len(self.bias)
        #print("w",self.dweight)
        #print("B",self.dbias)
        for l in range(m):
            #assert (self.weight[l].shape == self.dweight[l].shape)
            #assert (self.bias[l].shape == self.dbias[l].shape)
            self.weight[l] = self.weight[l] - self.learning_rate * self.dweight[l]
            self.bias[l] = self.bias[l] - self.learning_rate * self.dbias[l]
        
        self.Z = []
        self.A = []
        self.dweight = []
        self.dbias = []

    def gradientcheck(self):
        epsi = 10^-7
        thetapprox = []
        theta = []
        # print(np.asarray(self.weight).shape)
        # print(np.asarray(self.bias).shape)
        # print(np.asarray(self.dweight).shape)
        # print(np.asarray(self.dbias).shape)
        for i in range(len(self.bias)):
            thetapprox.append(self.weight[i])
            thetapprox.append(self.bias[i])
            theta.append(self.dweight[i])
            theta.append(self.dbias[i])  
        theta = np.asarray(theta)
        theta = theta[:, np.newaxis]
        thetapprox = np.asarray(thetapprox)
        thetapprox = thetapprox[:, np.newaxis]
        print(theta)        
        print(thetapprox)        
        for i in range(len(theta)):
            thetapprox[i] = (thetapprox[i] + epsi) - (thetapprox[i]- epsi)/(2*epsi)
        ans = np.sum(thetapprox - theta,keepdims=True)/(np.sum(thetapprox,keepdims=True)+np.sum(theta,keepdims=True))
        print(ans)
            


    def saveweights(self):
        pass

    def loadweights(self):
        pass