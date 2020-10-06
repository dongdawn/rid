#! /usr/bin/python
import glob
import numpy as np
cv_dim=11
dirname='/home/dongdong/SCR/rid_pdz.run02/'
itername=sorted(glob.glob(dirname+'iter.*'))
wf=open('force_std_min.dat','w')
for ii in itername:
    filename=ii+'/02.train/data/data.raw'
    print(filename)
    data=np.loadtxt(filename)
    f_min=np.min(np.std(data[:,cv_dim:],axis=0))
    wf.write(str(f_min)+'  ')
    wf.write('\n')
wf.close()
