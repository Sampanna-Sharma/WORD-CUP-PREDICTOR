import pandas as pd
import os
directory = os.fsencode("20")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
        
    data = open("20/"+filename,'r',encoding="utf8")
    comdata = data.read()
    Namearra = comdata.split("\n")
    df = pd.read_csv("20/FIFA14.csv")
    EF = df.loc[df['NAME'].isin(Namearra)].drop_duplicates(subset=['NAME']).drop(['LOADDATE'], axis=1)
    app = pd.DataFrame([['','','','',71.77,71.44,59.85,64.72,67.42,59.436,66.17]],
                        columns=['NAME','CLUB','LEAGUE','POSITION','RATING','PACE','SHOOTING','PASSING','DRIBBLING','DEFENDING','PHYSICAL'])

    for _ in range(23-EF['NAME'].count()):
        EF = EF.append(app,ignore_index=True)

    EF.to_csv(filename[:-4]+'.csv',index = False)