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
    global global_inpus
    global global_o_energy
    global dist

    cv_dim = len(xx_) 
    xx = np.concatenate([np.reshape(xx_, [1, cv_dim]), dist],axis=1)

    zero4 = np.zeros ([xx.shape[0], cv_dim + cv_dist_dim])
    data_inputs = np.concatenate ((xx, zero4), axis = 1)

    ee = []
    for ss,ii,oo in zip(global_sess, global_inputs, global_o_energy) :
        ret = ss.run(oo, feed_dict = {ii: data_inputs})
        ret *= f_cvt
        ee.append(ret[0])
    return np.average(np.reshape(ee, [-1]), axis = 0)
    
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
    parser.add_argument("-s", "--start", type=float, nargs="+", default = 0,
                        help="The initial point of optimization")
    parser.add_argument("-o", "--output", type=str, default = "x.cry.out",
                        help="The output file of x")
    args = parser.parse_args()

    models = args.models    

    global global_sess
    global global_inputs
    global global_o_energy
    global global_o_forces
    global dist
    
    for ii in models :
        graph = load_graph (ii)
        global_sess.append(tf.Session(graph = graph))
        global_inputs.append(graph.get_tensor_by_name ('load/inputs:0') )
        global_o_energy.append(graph.get_tensor_by_name ('load/o_energy:0'))
        global_o_forces.append(graph.get_tensor_by_name ('load/o_forces:0'))    
    
    data = np.loadtxt('crystal.md.cv')[:,1:]
    nframe = np.shape(data)[0]
    fe = []
    op = []
    for ii in range(nframe):
        xx0 = data[ii,:12]
        dist = np.reshape(xx0[-cv_dist_dim:],[1,cv_dist_dim])
        ret = minimize(__val, xx0[:-cv_dist_dim], method = 'BFGS', jac = __der, options={'disp': False})
        if not ret.success : 
            raise RuntimeError("failed bfgs opti")
        fe0 = __val(xx0[:-cv_dist_dim])
        fe.append(fe0)
        fe.append(ret.fun)
        op.append(np.concatenate([np.reshape(ret.x, [1,-1]), dist],axis=1))
        print(fe0, ret.fun)
    fe = np.array(fe).reshape([-1,2])
    op = np.concatenate(op, axis = 0)
    np.savetxt (args.output, np.concatenate([op,fe],axis=1))

if __name__ == '__main__':
    _main()

