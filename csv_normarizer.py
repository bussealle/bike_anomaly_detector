import sys
import os
import numpy as np
import pandas as pd



argvs = sys.argv
if len(argvs)<2:
    print "wrong arg"
    sys.exit()

csvs = argvs[1:]


for csv_path in csvs:
    csv = pd.read_csv(csv_path)
    
    df1 = csv['Time']

    df2 = csv.iloc[:,2:4].apply(lambda x: np.sqrt(x.dot(x)), axis=1)

    new_csv = pd.concat([df1, df2], axis=1)    
    new_csv.columns = ['Time', 'Norm']

    outpath = './normalized/'
    if not(os.path.exists(outpath)):
        os.mkdir(outpath)
    savepath = outpath+os.path.basename(csv_path).replace(".csv","_norm.csv")
    print "saving csv... : "+savepath
    new_csv.to_csv(savepath,index=False)
