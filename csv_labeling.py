import sys
import os
import numpy as np
import pandas as pd



argvs = sys.argv
if len(argvs)!=3:
    print "wrong arg"
    sys.exit()

fp = argvs[1]
csvp = argvs[2]

f = open(fp)
line = f.readline()
print "lebel:{}".format(line)
lbnum = int(line)
lbarr = []

while line:
    line = f.readline()
    line = line.rstrip('\n')
    lbarr.append(line)


#print lbarr
f.close()

csv = pd.read_csv(csvp)

pairs = []
for i in range(len(lbarr)//2):
    j = i*2
    pairs.append([lbarr[j],lbarr[j+1]])

print pairs

print csv.head()
for i in range(len(pairs)):
    col_begin = csv.loc[csv['Time'].str.contains("^{}.*".format(pairs[i][0]))]
    col_end = csv.loc[csv['Time'].str.contains("^{}.*".format(pairs[i][1]))]
    index_begin =  col_begin.index[0]
    index_end = col_end.index[-1]
    
    csv.ix[index_begin:index_end,'Label'] = lbnum

outpath = './labeled/'
if not(os.path.exists(outpath)):
    os.mkdir(outpath)
savepath = outpath+os.path.basename(csvp).replace(".csv","_labeled.csv")
print "saving csv... : "+savepath
csv.to_csv(savepath,index=False)
