#!/usr/bin/env python3

import os,glob,shutil
import argparse
import numpy as np
import sklearn.cluster as cluster
from sklearn import preprocessing
cv_dih_dim = 9
weight=np.array([0.3]*cv_dih_dim+[1,1,1])
def parse_cmd () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--idx-file", type=str, default = 'sel.out',
                        help="The sel idx files")
    parser.add_argument("-c","--cv-file", type=str, default = 'sel.angle.out',
                        help="The sel cv files")
    parser.add_argument("-t","--distance_threshold", type=float, default = 0.3,
                        help="distance_threshold")
    parser.add_argument("--output-idx", type=str, default = 'cls.out',
                        help="The output cv idx")
    parser.add_argument("--output-cv", type=str, default = 'cls.angle.out',
                        help="The output cv value")
    args = parser.parse_args()
    return args

def cv_dist (a, b) :
    diff = a - b
    angle_pbc_range = len(diff)
    if cv_dih_dim is not None :
        angle_pbc_range = cv_dih_dim
    for ii in range(len(diff)) :
        value = diff[ii]
        if ii < angle_pbc_range :
            if value < -np.pi :
                value += 2 * np.pi
            elif value >= np.pi :
                value -= 2 * np.pi
        diff[ii] = value
    return np.linalg.norm(diff)

def mk_dist (cv) :
    nframe = cv.shape[0]
    dist = np.zeros([nframe, nframe])
    for ii in range(nframe) :
        for jj in range(ii+1, nframe) :
            dist[ii][jj] = cv_dist(cv[ii], cv[jj])
            dist[jj][ii] = dist[ii][jj]
    return dist

def mk_cluster (dist, distance_threshold) :
    cls = cluster.AgglomerativeClustering(n_clusters = None, 
                                          linkage='average', 
                                          affinity = 'precomputed',
                                          distance_threshold=distance_threshold)
    cls.fit(dist)
    return cls.labels_

def sel_from_cluster (angles, distance_threshold) :
    dist = mk_dist (angles)
    labels = mk_cluster (dist, distance_threshold)
    # make cluster map
    cls_map = []
    for ii in range(len(set(labels))) :
        cls_map.append([])
    for ii in range(len(labels)) :
        cls_idx = labels[ii]
        cls_map[cls_idx].append(ii)
    # randomly select from cluster
    cls_sel = []
    np.random.seed(seed = None)
    for ii in cls_map :
        #_ret = np.random.choice(ii, 1)
        _ret = np.min(ii)
        cls_sel.append (_ret)    
    cls_sel.sort()
    return cls_sel

def singleSD(nList,refList):
    aa=nList-refList
    bb=np.abs(aa)
    bb[bb>180]=360-bb[bb>180]
    return np.sqrt(np.mean(bb**2))

def _main () :
    args = parse_cmd ()
    angidx = np.loadtxt (args.idx_file)
    data = np.loadtxt (args.cv_file)    
    distance_threshold = args.distance_threshold
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
    alldata_scaled=preprocessing.scale(alldata) 
    print(alldata_scaled)  
    cls_sel = sel_from_cluster(alldata_scaled, distance_threshold)
    
    np.savetxt(args.output_idx, angidx[cls_sel], fmt = '%d')
    np.savetxt(args.output_cv,  data[cls_sel], fmt = '%.6f')

if __name__ == '__main__' :
    _main()

