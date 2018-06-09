import os

directory = os.fsencode("2014")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    sentance = ""
    Out = open(filename[3:],'w',encoding="utf8")
    print(filename)
    with open("2014/"+filename,'r',encoding="utf8") as F:
        print(F.readline())
        for line in F:
            sentance = ""
            Name = False
            for alphabet in line:
                if(alphabet == '#'):
                    Name = False;
                if(Name == True):
                    sentance = sentance + alphabet
                if(alphabet == ')'):
                    Name = True
            sentance = sentance[6:].replace("  ","").replace("\n \n","\n")
            Out.write(sentance.replace(" \n",("\n")))
    Out.close();

