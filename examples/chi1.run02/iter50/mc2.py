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
            my_hist[jj, np.int(pp[ii,jj*2+0]//delta)-50, np.int(pp[ii,jj*2+1]//delta)-50] += 1
    my_hist /= (pp.shape[0] * delta * delta)
    return my_hist

def _main () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=[], nargs = '*', type=str, 
                        help="Frozen model file to test")
    parser.add_argument("-fd", "--full_dimension", default = 4, type=int, 
                        help="The dimensionality of FES")
    parser.add_argument("-ns", "--num_step", default = 1000000, type=int,
                        help="number of mc step")
    parser.add_argument("-nw", "--num_walker", default=2000, type=int, 
                        help="number of walker")
    args = parser.parse_args()

    model = args.model 
    fd = args.full_dimension
    ns = args.num_step
    nw = args.num_walker
    positons = []
    energies = []
    forces = []
    graph = load_graph (model[0])

    with tf.Session(graph = graph) as sess:        
        walker = Walker(fd, nw, sess)
        for ii in range(ns):
            pp, ee, ff = walker.sample(compute_ef)
            for dd in range(fd):
                fp = open("angle%d.dat" %dd, "a")
                for aa in pp[:,dd]:
                    fp.write(str(aa)+'    ')
                fp.write('\n')
                fp.close()

    
if __name__ == '__main__':
    _main()

