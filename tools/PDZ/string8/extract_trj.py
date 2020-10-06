#!/usr/bin/env python3

import os,glob,shutil
import argparse
import numpy as np
import sklearn.cluster as cluster
from sklearn import preprocessing
cv_dih_dim = 9

def singleSD(nList,refList):
    aa=nList-refList
    bb=np.abs(aa)
    bb[bb>180]=360-bb[bb>180]
    return np.sqrt(np.mean(bb**2))
allsave=[]
data = np.loadtxt ('string1.dat')    
dataR=[]
dataD=[]
dataCry=np.loadtxt('crystal.cv')/3.14*180
for ii in range(len(data)):
    tempR=singleSD(data[ii,0:9]/3.14*180,dataCry[0:9])
    dataR.append(tempR)
    dataD.append((data[ii,9]+data[ii,10]+data[ii,11])/3.0)
dataR=np.array(dataR)
dataD=np.array(dataD)
dataR=dataR.reshape(-1,1)
dataD=dataD.reshape(-1,1)
alldata=np.concatenate((dataR,dataD),axis=1)
for i in range(len(alldata)):
    if alldata[i][1]>3:
        allsave.append(data[i])
        break
for i in range(len(alldata)):
    if alldata[i][1]>2.5 and alldata[i][1]<3:
        allsave.append(data[i])
        break
for i in range(len(alldata)):
    if alldata[i][1]>2.0 and alldata[i][1]<2.5:
        allsave.append(data[i])
        break
rrange=np.arange(2,0.9,-0.1)
for r in range(len(rrange)-1):
    temp=[]
    temp_o=[]
    for i in range(len(alldata)):
        if alldata[i][1]<rrange[r] and alldata[i][1]>rrange[r+1]:
            #print(alldata[i])
            temp.append(alldata[i])
            temp_o.append(data[i])
    temp=np.array([temp])
    if len(temp[0])>0:
        print(temp[0][:,0])
        aa=np.argmax(temp[0][:,0])
        allsave.append(temp_o[aa])
temp=[]
temp_o=[]
for i in range(len(alldata)):
    if alldata[i][1]<0.85:
        temp.append(alldata[i])
        temp_o.append(data[i])
rrange=np.arange(60,0,-5)
for r in range(len(rrange)-1):
    tt=[]
    for i in range(len(temp)):
        if temp[i][0]<rrange[r] and temp[i][0]>rrange[r+1]:
            tt.append(temp_o[i])
    if len(tt)>0:
        print(tt)
        allsave.append(tt[-1])
            
allsave=np.array(allsave)
np.savetxt('trj.dat',allsave,fmt='%.3f')
