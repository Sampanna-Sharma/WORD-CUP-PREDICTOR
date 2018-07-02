import pandas as pd
import numpy as np
import os

def getfootdata():
    X = []
    Y = []
    for year in ["2010","2014","EURO16"]:
        df = pd.read_csv("DATA/"+year+"/matches.csv")
        for index, row in df.iterrows():
            #print (year,row["home team name"], row["away team name"])
            if(row["Home Team Goals"]>row["Away Team Goals"]):
                Y.append([1,0,0])
            elif(row["Home Team Goals"]<row["Away Team Goals"]):
                Y.append([0,1,0])
            else:
                Y.append([0,0,1])
            
            ef = pd.read_csv("DATA/"+year+"/"+row["home team name"].lower()+".csv", encoding='latin-1')
            ff = pd.read_csv("DATA/"+year+"/"+row["away team name"].lower()+".csv", encoding='latin-1')
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
    Y = np.asarray(Y).T
    #print(X)
    #print(X.shape)
    #print(Y)
    #print(X.shape)
    #print(Y.shape)
    return X,Y
