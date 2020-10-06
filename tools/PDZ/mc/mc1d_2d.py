#!/usr/bin/env python3

import re
import os
import sys
import json
import argparse
import pylab
import numpy as np
import tensorflow as tf
from multiprocessing.dummy import Pool as ThreadPool

kbT = (8.617343E-5) * 300 
beta = 1.0 / kbT
f_cvt = 96.485

EPSILON = 1e-8

def load_graph(frozen_graph_filename, 
               prefix = 'load'):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name=prefix, 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

class Walker(object):
    def __init__(self, fd_dih, fd_dist, nw, sess):
        self._full_dim_dih = fd_dih
        self._full_dim_dist = fd_dist
        self._full_dim = self._full_dim_dih + self._full_dim_dist
        self._num_walker = nw
        self._move_scale = 0.5
        self._sess = sess
        self._shape = (self._num_walker, self._full_dim)
        self._shape1 = (self._num_walker, self._full_dim_dih)
        self._shape2 = (self._num_walker, self._full_dim_dist)
        self._shape11 = (self._num_walker-100, self._full_dim_dih)
        self._shape22 = (self._num_walker-100, self._full_dim_dist)
        self._shape_cry = (100, self._full_dim)
        # absolute coordinate
        cry=np.loadtxt('crystal.cv')
        self._position1 = np.append(np.random.uniform(size=self._shape11) * np.pi * 2 , np.random.uniform(size=self._shape22) * 3.6, axis=1)
        self._position2 = np.random.normal(size=self._shape_cry,loc=cry,scale=0.1)
        self._position = np.append(self._position1 , self._position2, axis=0)
        self._energy = np.zeros([self._num_walker])
        self._force = np.zeros([self._num_walker, self._full_dim])
        self._sample_step = 20
        self._acp_ratio_lb = 0.15
        self._acp_ratio_ub = 0.75
        self.max_scale = np.pi
        self.min_scale = 0.01
        self.inc_scale_fac = 1.25

    def sample(self, compute_ef, inter_step=1):
        acp_ratio = []
        self._energy, self._force = compute_ef(self._sess, self._position)
        for _ in range(inter_step):
            position_new =np.append( np.mod(self._position[:,0:self._full_dim_dih] + np.random.normal(scale=self._move_scale,size=self._shape1), 2*np.pi),np.mod(self._position[:,self._full_dim_dih:self._full_dim] + np.random.normal(scale=self._move_scale*0.8,size=self._shape2), 3.6), axis=1)
            energy_new, force_new = compute_ef(self._sess, position_new)
            # in case of overflow
            prob_ratio = np.exp(np.minimum(- beta * (energy_new - self._energy), 0))
            idx = np.random.uniform(size=self._num_walker) < np.reshape(prob_ratio, [-1])
            self._position[idx, :] = position_new[idx, :]
            self._energy[idx] = energy_new[idx]
            self._force[idx] = force_new[idx]
            acp_ratio.append(np.mean(idx))
        acp_ratio = np.mean(acp_ratio)
        if acp_ratio > self._acp_ratio_ub:
            # move_scale is too small
            self._move_scale = min(
                self._move_scale*self.inc_scale_fac,
                self.max_scale)
            print(
                "Increase move_scale to %f due to high acceptance ratio: %f" % (
                    self._move_scale, acp_ratio))
            # print(self._position[:5, :, :])
        elif acp_ratio < self._acp_ratio_lb:
            # move_scale is too large
            self._move_scale = max(
                self._move_scale/self.inc_scale_fac,
                self.min_scale)
            print(
                "Decrease move_scale to %f due to low acceptance ratio: %f" % (
                    self._move_scale, acp_ratio))
        return self._position, self._energy, self._force


def compute_ef (sess, position) :
    graph = sess.graph

    inputs  = graph.get_tensor_by_name ('load/inputs:0')
    o_energy= graph.get_tensor_by_name ('load/o_energy:0')
    o_forces= graph.get_tensor_by_name ('load/o_forces:0')

    zeros = 0.0 * position
    data_inputs = np.concatenate ((position, zeros), axis = 1)
    feed_dict_test = {inputs: data_inputs}

    data_ret = sess.run ([o_energy, o_forces], feed_dict = feed_dict_test)
    return data_ret[0], data_ret[1]

def singleSD(nList,refList):
    aa=nList-refList
    bb=np.abs(aa)
    bb[bb>180]=360-bb[bb>180]
    return np.sqrt(np.mean(bb**2))

def my_hist2d(pp, xx, yy, delta1, delta2, fd1, fd2):
    dataCry=np.loadtxt('crystal.cv')/3.14*180
    my_hist = np.zeros((1, len(xx), len(yy)))
    my_hist1 =np.zeros((1, len(xx)))
    my_hist2 =np.zeros((1, len(yy)))
    for ii in range(pp.shape[0]):
        my_hist[0, np.int(singleSD(pp[ii,0:fd1]/3.14*180,dataCry[0:fd1])//delta1), np.int(np.mean(pp[ii,fd1:fd1+fd2])//delta2)] += 1
        my_hist1[0, np.int(singleSD(pp[ii,0:fd1]/3.14*180,dataCry[0:fd1])//delta1)] += 1
        my_hist2[0, np.int(np.mean(pp[ii,fd1:fd1+fd2])//delta2)] += 1
    my_hist /= (pp.shape[0] * delta1 * delta2)
    my_hist1 /= (pp.shape[0] * delta1)
    my_hist2 /= (pp.shape[0] * delta2)
    return my_hist, my_hist1, my_hist2

def _main () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=[], nargs = '*', type=str, 
                        help="Frozen model file to test")
    parser.add_argument("-fd_dih", "--full_dimension_dih", default = 9, type=int, 
                        help="The dimensionality of FES")
    parser.add_argument("-fd_dist", "--full_dimension_dist", default = 3, type=int, 
                        help="The dimensionality of FES")
    parser.add_argument("-ns", "--num_step", default = 1000000, type=int,
                        help="number of mc step")
    parser.add_argument("-nw", "--num_walker", default=2000, type=int, 
                        help="number of walker")

    args = parser.parse_args()
    model = args.model 
    fd_dih = args.full_dimension_dih
    fd_dist = args.full_dimension_dist
    ns = args.num_step
    nw = args.num_walker
    positons = []
    energies = []
    forces = []
    graph = load_graph (model[0])
    bins=30
    xx = pylab.linspace(0,180, bins)
    yy = pylab.linspace(0,3.6, bins)
    pp_hist1cv1 = np.zeros((1, len(xx)))
    pp_hist1cv2 = np.zeros((1, len(yy)))
    pp_hist2d = np.zeros((1, len(xx), len(yy)))
    delta1 = 180.0 / bins
    delta2 = 3.6 / bins
    
    with tf.Session(graph = graph) as sess:        
        walker = Walker(fd_dih, fd_dist, nw, sess)
        for ii in range(100):
            pp, ee, ff = walker.sample(compute_ef)

        for ii in range(ns+1):
            pp, ee, ff = walker.sample(compute_ef)
            ##all 1d

            ##certain 2d
            pp_hist_new2d,pp_hist_new1cv1,pp_hist_new1cv2 = my_hist2d(pp, xx, yy, delta1, delta2, fd_dih, fd_dist)
            pp_hist2d = (pp_hist2d * ii + pp_hist_new2d) / (ii+1)
            pp_hist1cv1 = (pp_hist1cv1 * ii + pp_hist_new1cv1) / (ii+1)
            pp_hist1cv2 = (pp_hist1cv2 * ii + pp_hist_new1cv2) / (ii+1)

            if np.mod(ii,50000) == 0:
                zz1 = -np.log(pp_hist1cv1+1e-7)/beta
                zz1 *= f_cvt/4.184   ##kcal
                zz1 = zz1 - np.min(zz1)
                fp = open("1CV1_index0.dat", "a")
                for temp in zz1[0]:
                    fp.write(str(temp)+'    ')
                fp.write('\n')
                fp.close()

                zz2 = -np.log(pp_hist1cv2+1e-7)/beta
                zz2 *= f_cvt/4.184   ##kcal
                zz2 = zz2 - np.min(zz2)
                fp = open("1CV2_index1.dat", "a")
                for temp in zz2[0]:
                    fp.write(str(temp)+'    ')
                fp.write('\n')
                fp.close()

                zz2d = np.transpose(-np.log(pp_hist2d+1e-10), (0,2,1))/beta
                zz2d *= f_cvt/4.184
                zz2d = zz2d - np.min(zz2d)
                np.savetxt("2d_step%d.dat" %ii,zz2d[0])

                np.savetxt("position%d.dat" %ii,pp)
    
if __name__ == '__main__':
    _main()

