#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-31
# @Author  : WDD 
# @Link    : https://github.com/dongdawn
# @Version : v1
import sys
import os
import numpy as np
cut=np.pi/4
def trans1(data):
    trj=[]
    for aa in data:
        if aa < np.pi/2-cut or aa > np.pi/2*3+cut:
            trj.append(1)
        elif aa > np.pi/2+cut and aa < np.pi/2*3-cut:
            trj.append(2)
    aa = np.array([np.argwhere(np.diff(trj)).squeeze()])
    aa = aa.reshape(-1)
    num=len(aa)
    return num
           
num_walkers=12
num_iter=140
filedir='./'
cv_index=7
all_num=[]
for wal in range(int(num_walkers)):
    walker_name='%03d' %wal
    all_cv=[]
    for it in range(int(num_iter)):
        iteration="%06d" %it
        filename=filedir+"iter."+str(iteration)+"/00.enhcMD/"+str(walker_name)+"/plm.out"
        data = np.loadtxt(filename)
        ang = data[:,int(cv_index)]
        all_cv.extend(list(ang))
    all_cv=np.array(all_cv)
    all_cv[all_cv<0]+=2*np.pi
    print(trans1(all_cv))
    all_num.append(trans1(all_cv))
print('sum\n')
print(np.sum(all_num))

