#!/usr/bin/env python3

import re
import os
import sys
import json
import argparse
import pylab
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
    def __init__(self, fd, nw, sess):
        self._full_dim = fd
        self._num_walker = nw
        self._move_scale = 0.5
        self._sess = sess
        self._shape = (self._num_walker, self._full_dim)
        # absolute coordinate
        self._position = np.random.uniform(size=self._shape) * np.pi * 2
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
            position_new = np.mod(self._position + np.random.normal(scale=self._move_scale,
                                                             size=self._shape), 2*np.pi)
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

def my_hist(pp, xx, yy, delta, fd):
    my_hist = np.zeros((fd, len(xx), len(yy)))
    for ii in range(pp.shape[0]):
        for jj in range(fd):
            my_hist[jj, np.int(pp[ii,jj*2+0]//delta), np.int(pp[ii,jj*2+1]//delta)] += 1
    my_hist /= (pp.shape[0] * delta * delta)
    return my_hist

def my_hist1d(pp, xx, delta, fd):
    my_hist = np.zeros((fd, len(xx)))
    for ii in range(pp.shape[0]):   ###trj_num
        for jj in range(fd):        ###cv_num
            my_hist[jj, np.int(pp[ii,jj]//delta)] += 1
    my_hist /= (pp.shape[0] * delta)
    return my_hist

def my_hist2d(pp, xx, yy, delta, cv1, cv2):
    my_hist = np.zeros((1, len(xx), len(yy)))
    for ii in range(pp.shape[0]):
        my_hist[0, np.int(pp[ii,cv1]//delta), np.int(pp[ii,cv2]//delta)] += 1
    my_hist /= (pp.shape[0] * delta * delta)
    return my_hist

def my_hist1d_1cv(pp, xx, delta, cv):
    my_hist = np.zeros((1, len(xx)))
    for ii in range(pp.shape[0]):   ###trj_num
               ###cv_num
        my_hist[0, np.int(pp[ii,cv]//delta)] += 1
    my_hist /= (pp.shape[0] * delta)
    return my_hist

def _main () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=[], nargs = '*', type=str, 
                        help="Frozen model file to test")
    parser.add_argument("-fd", "--full_dimension", default = 9, type=int, 
                        help="The dimensionality of FES")
    parser.add_argument("-ns", "--num_step", default = 1000000, type=int,
                        help="number of mc step")
    parser.add_argument("-nw", "--num_walker", default=2000, type=int, 
                        help="number of walker")
    parser.add_argument("-cv1", "--cv1_index", default=1, type=int,
                        help="cv1 index")
    parser.add_argument("-cv2", "--cv2_index", default=2, type=int,
                        help="cv2 index")

    args = parser.parse_args()

    model = args.model 
    fd = args.full_dimension
    ns = args.num_step
    nw = args.num_walker
    cv1 = args.cv1_index
    cv2 = args.cv2_index
    positons = []
    energies = []
    forces = []
    graph = load_graph (model[0])
    bins=25
    xx = pylab.linspace(0,2* np.pi, bins)
    yy = pylab.linspace(0,2* np.pi, bins)
    pp_hist = np.zeros((fd, len(xx)))
    pp_hist2d = np.zeros((1, len(xx), len(yy)))
    pp_hist2d2 = np.zeros((1, len(xx), len(yy)))
    pp_hist2d3 = np.zeros((1, len(xx), len(yy)))
    delta = 2.0 * np.pi / bins
    
    with tf.Session(graph = graph) as sess:        
        walker = Walker(fd, nw, sess)
        for ii in range(100):
            pp, ee, ff = walker.sample(compute_ef)

        for ii in range(ns+1):
            pp, ee, ff = walker.sample(compute_ef)
            ##all 1d
            pp_hist_new = my_hist1d(pp, xx, delta, fd)
            pp_hist = (pp_hist * ii + pp_hist_new) / (ii+1)
            ##certain 2d
            pp_hist_new2d = my_hist2d(pp, xx, yy, delta, cv1, cv2)
            pp_hist2d = (pp_hist2d * ii + pp_hist_new2d) / (ii+1)

            pp_hist_new2d2 = my_hist2d(pp, xx, yy, delta, cv1+3, cv2+3)
            pp_hist2d2 = (pp_hist2d2 * ii + pp_hist_new2d2) / (ii+1)

            pp_hist_new2d3 = my_hist2d(pp, xx, yy, delta, cv1+6, cv2+6)
            pp_hist2d3 = (pp_hist2d3 * ii + pp_hist_new2d3) / (ii+1)

            if np.mod(ii,50000) == 0:
                zz = -np.log(pp_hist+1e-7)/beta
                zz *= f_cvt/4.184   ##kcal
                zz = zz - np.min(zz)
                for jj in range(fd):
                    fp = open("1CV_index%d.dat" %jj, "a")
                    for temp in zz[jj]:
                        fp.write(str(temp)+'    ')
                    fp.write('\n')
                    fp.close()

                zz2d = np.transpose(-np.log(pp_hist2d+1e-10), (0,2,1))/beta
                zz2d *= f_cvt/4.184
                zz2d = zz2d - np.min(zz2d)
                np.savetxt("2CV12_step%d.dat" %ii,zz2d[0])
    
                zz2d2 = np.transpose(-np.log(pp_hist2d2+1e-10), (0,2,1))/beta
                zz2d2 *= f_cvt/4.184
                zz2d2 = zz2d2 - np.min(zz2d2)
                np.savetxt("2CV45_step%d.dat" %ii,zz2d2[0])

                zz2d3 = np.transpose(-np.log(pp_hist2d3+1e-10), (0,2,1))/beta
                zz2d3 *= f_cvt/4.184
                zz2d3 = zz2d3 - np.min(zz2d3)
                np.savetxt("2CV78_step%d.dat" %ii,zz2d3[0])

if __name__ == '__main__':
    _main()

