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
cv_dim = 9

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

def _main () :
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", default=[], nargs = '*', type=str, 
                        help="Frozen models file to test")
    parser.add_argument("-d", "--data", type=str, 
                        help="The data for test")
    parser.add_argument("-o", "--output", default="fene.out", type=str,
                        help="output free energy")

    args = parser.parse_args()

    models = args.models    
    data_ = np.loadtxt (args.data)
    data = data_[:,:cv_dim]
    nframes = data.shape[0]
    
    forces = []
    energys = []
    for ii in models :
        graph = load_graph (ii)
        with tf.Session(graph = graph) as sess:        
            ee, ff = test_ef (sess, data)
            forces = np.append (forces, ff)
            energys = np.append (energys, ee)

    energys = np.reshape (energys, [len(models), nframes])
    energys *= f_cvt
    
    np.savetxt (args.output, energys)

if __name__ == '__main__':
    _main()
