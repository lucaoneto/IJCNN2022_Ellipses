import numpy as np
import pandas as pd
import glob, os, sys

path = os.getcwd().replace('\\', '/')

dataset = pd.DataFrame([])
for number, file in enumerate(glob.glob(path+'/' + '*txt')):
    current_df = pd.read_csv(file, sep=';', decimal='.',header=0)
    dataset = pd.concat([dataset,current_df],axis=0)
    sys.stdout.write("\r dataset shape %i, %i" % ((dataset.shape) ))
    sys.stdout.flush()

dataset.to_csv(path+'/data14.csv', header = True, decimal='.', sep=';', index=False)
print('dataset stored correctly')
