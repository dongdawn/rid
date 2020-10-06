#!/usr/bin/env python3

import re
import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

kbT = (8.617343E-5) * 300 
beta = 1.0 / kbT
f_cvt = 96.485
cv_dist_dim = 3 
global_inputs = []
global_o_energy = []
global_o_forces = []
global_sess = []

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

def __val (xx_) :
    global global_sess
    global global_inputs
    global global_o_energy

    zero4 = np.zeros (xx_.shape)
    data_inputs = np.concatenate ((xx_, zero4), axis = 1)
    ee = []
    for ss,ii,oo in zip(global_sess, global_inputs, global_o_energy) :
        ret = ss.run(oo, feed_dict = {ii: data_inputs})
        ret *= f_cvt
        ee.append(ret.reshape([-1,1]))
    return np.mean(np.concatenate(ee, axis=1), axis = 1)
    
def __der (xx_) :
    global global_sess
    global global_inpus
    global global_o_forces
 
    cv_dim = len(xx_) 
    xx = np.concatenate([np.reshape(xx_, [1, cv_dim]), dist],axis=1)   

    zero4 = np.zeros ([xx.shape[0], cv_dim + cv_dist_dim])
    data_inputs = np.concatenate ((xx, zero4), axis = 1)

    ee = []
    for ss,ii,oo in zip(global_sess, global_inputs, global_o_forces) :
        ret = ss.run(oo, feed_dict = {ii: data_inputs})
        ret *= f_cvt
        ee.append(ret[0])
    return np.average(ee, axis = 0)[:-cv_dist_dim]

def _main () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", default=[], nargs = '*', type=str, 
                        help="Frozen models file to test")
    args = parser.parse_args()

    models = args.models    

    global global_sess
    global global_inputs
    global global_o_energy
    global global_o_forces
    
    for ii in models :
        graph = load_graph (ii)
        global_sess.append(tf.Session(graph = graph))
        global_inputs.append(graph.get_tensor_by_name ('load/inputs:0') )
        global_o_energy.append(graph.get_tensor_by_name ('load/o_energy:0'))
        global_o_forces.append(graph.get_tensor_by_name ('load/o_forces:0'))    

    data = np.loadtxt('crystal.md.cv')[:,1:]
    print(np.shape(data))
    fe = __val(data)
    n_cv=12
    n_dih=9
    ref_cv = np.loadtxt('./crystal.cv')[:n_dih].reshape([1,-1])
    dataR = data[:,:n_dih]
    dataR = np.mean((np.mod(dataR - ref_cv + np.pi/2, np.pi) - np.pi/2)**2,axis=1)**0.5
    dataD = np.mean(data[:,n_dih:n_cv],axis=1).reshape([-1,1])

    np.savetxt ('evl.cry.dat', np.concatenate([dataR.reshape([-1,1]),dataD,fe.reshape([-1,1])],axis=1))

if __name__ == '__main__':
    _main()

