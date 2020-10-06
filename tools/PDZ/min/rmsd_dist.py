import sys
import os
import numpy as np
import argparse

n_cv=12
n_dih=9

data = np.loadtxt('./x.cry.out')
ref_cv = np.loadtxt('./crystal.cv')[:n_dih].reshape([1,-1])
dataR = data[:,:n_dih]
dataR = np.mean((np.mod(dataR - ref_cv + np.pi/2, np.pi) - np.pi/2)**2,axis=1)**0.5
dataD = np.mean(data[:,n_dih:n_cv],axis=1).reshape([-1,1])
dataE = data[:,n_cv:]

np.savetxt('cv_ener.cry.dat',np.concatenate([dataR.reshape([-1,1]),dataD,dataE],axis=1))
