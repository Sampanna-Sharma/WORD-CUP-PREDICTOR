import pandas as pd
import numpy as np
import os

def getfootdata(home_team,away_team):
    X = []
    for year in ["2018"]:
        ef = pd.read_csv("DATA/"+year+"/"+home_team.lower()+".csv", encoding='latin-1')
        ef = ef.drop(ef.index[2])
        ff = pd.read_csv("DATA/"+year+"/"+away_team.lower()+".csv", encoding='latin-1')
        ff = ff.drop(ff.index[2])
        temp1 = (ef.as_matrix(columns = ["RATING","PACE", "SHOOTING" ,"PASSING",  "DRIBBLING",  "DEFENDING",  "PHYSICAL"]).reshape(22,7))#.reshape(154,1))
        temp2 = (ff.as_matrix(columns = ["RATING","PACE", "SHOOTING" ,"PASSING",  "DRIBBLING",  "DEFENDING",  "PHYSICAL"]).reshape(22,7))#.reshape(154,1))
        #print(temp2)
        tempX = np.append(temp1,temp2).reshape(44,7)
        tempX =  (tempX- tempX.min(axis=0))/(tempX.max(axis=0)-tempX.min(axis=0))
        tempX = tempX.reshape(308,1)
        #print(tempX)
        #print(tempX.shape)
        X.append(tempX)
        
        
    X = np.asarray(X,dtype=np.float32).T
    X = X[0]
    return X
