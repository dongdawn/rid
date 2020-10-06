#!/usr/bin/env python3

import re
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

kbT = (8.617343E-5) * 300 
beta = 1.0 / kbT
f_cvt = 96.485
cv_dim = 12

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

def test_ef (sess, xx) :
    graph = sess.graph

    inputs  = graph.get_tensor_by_name ('load/inputs:0')
    o_energy= graph.get_tensor_by_name ('load/o_energy:0')
    o_forces= graph.get_tensor_by_name ('load/o_forces:0')

    zero4 = np.zeros ([xx.shape[0], cv_dim])
    data_inputs = np.concatenate ((xx, zero4), axis = 1)
    feed_dict_test = {inputs: data_inputs}

    data_ret = sess.run ([o_energy, o_forces], feed_dict = feed_dict_test)
    return data_ret[0], data_ret[1]


def compute_std (forces) :
    nmodels = forces.shape[0]
    nframes = forces.shape[1]
    ncomps = forces.shape[2]
    
    stds = []
    for ii in range (nframes) :
        # print ( forces[0, ii], forces[1, ii], forces[2, ii], forces[3, ii])
        avg_std = 0
        for jj in range (ncomps) :
            mystd = np.std (forces[:, ii, jj])
            avg_std += mystd * mystd
        avg_std = np.sqrt (avg_std / float(ncomps))
        stds.append (avg_std)
    return np.array (stds)
    

def _main () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", default=[], nargs = '*', type=str, 
                        help="Frozen models file to test")
    parser.add_argument("-d", "--data", type=str, 
                        help="The data for test")
    parser.add_argument("-o", "--output", default="free_energy.out", type=str, 
                        help="output selected idx")
    args = parser.parse_args()

    models = args.models    
    data_ = np.loadtxt (args.data)
    data = data_[:,:cv_dim]
    nframes = data.shape[0]
    
    energys = []
    for ii in models :
        graph = load_graph (ii)
        with tf.Session(graph = graph) as sess:        
            ee, ff = test_ef (sess, data)
            energys = np.append (energys, ee)
    energys=np.array(energys)
    energys=energys.reshape(4,-1)
    print(energys)
    ave_ene=np.average(energys,axis=0)
    np.savetxt (args.output, ave_ene*f_cvt/4.184)

if __name__ == '__main__':
    _main()
