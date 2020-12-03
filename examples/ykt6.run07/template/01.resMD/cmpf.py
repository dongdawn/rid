#!/usr/bin/env python3

import numpy as np
import subprocess as sp

tail=0.80
cv_dih_dim = 0
data = np.loadtxt ("plm.res.out")
data = data[:,1:]

mk_kappa_cmd = "grep KAPPA plumed.res.dat  | awk '{print $4}' | cut -d '=' -f 2 > kappa.out"
sp.check_call (mk_kappa_cmd, shell = True)

kk=np.loadtxt('kappa.out'); 
cc=np.loadtxt('centers.out'); 

nframes=data.shape[0]
ndih_values=data.shape[1]
if cv_dih_dim is not None :
    ndih_values = cv_dih_dim

start_f = int(nframes*(1-tail))
avgins = np.average (data[start_f:,:], axis = 0)

diff=np.zeros (avgins.shape)
for ii in range(len(avgins)) :
    diff[ii] = avgins[ii] - cc[ii]
    if (ii < ndih_values) :
        if   diff[ii] >= np.pi :
            diff[ii] -= np.pi * 2.
        elif diff[ii] < -np.pi :
            diff[ii] += np.pi * 2.

ff = np.multiply ( kk, diff ); 
np.savetxt ('force.out',  np.reshape(ff, [1,-1]), fmt='%.10e');
