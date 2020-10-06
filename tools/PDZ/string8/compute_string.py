#!/usr/bin/env python3
# to be converted to python3

import os
import argparse
import numpy as np
import subprocess as sp
import logging
import StringUtils
from scipy.interpolate import interp1d
from subprocess import Popen, PIPE
import tensorflow as tf

#def test_compute_force (xx):
#    yy = np.zeros (np.size(xx))
#    yy[0] = 8*xx[0]*(1-xx[0]*xx[0]-xx[1]*0.5)
#    yy[1] = 2*(1-xx[0]*xx[0]-xx[1])
#    return yy
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


def test_ef (sess, position) :
    graph = sess.graph
    inputs  = graph.get_tensor_by_name ('load/inputs:0')
    o_energy= graph.get_tensor_by_name ('load/o_energy:0')
    o_forces= graph.get_tensor_by_name ('load/o_forces:0')

    zeros = 0.0 * position
    data_inputs = np.concatenate ((position, zeros), axis = 1)
    feed_dict_test = {inputs: data_inputs}

    data_ret = sess.run ([o_energy, o_forces], feed_dict = feed_dict_test)
    return data_ret[0], data_ret[1]

def test_compute_force (models,data) :
    forces=[]
    for ii in models :
        graph = load_graph (ii)
        with tf.Session(graph = graph) as sess:
            ee, ff = test_ef (sess, data)
            forces.append (ff)
    forces=np.array(forces)
    nmodels = forces.shape[0]
    nframes = forces.shape[1]
    ncomps = forces.shape[2]

    forces = np.reshape (forces, [nmodels, nframes, ncomps])
    ave_force=np.average(forces,axis=0)
    return ave_force

def string_compute_force (models, step,
                          string ) :
    numb_node = string.shape[0]
    dim = string.shape[1]
    force=test_compute_force(models,string)
    #force = np.zeros ((numb_node, dim))
    #for jj in range (numb_node):
    #    force[jj] = test_compute_force (models, string[jj])
    return force

def init_linear_string (node_start, node_end, numb_node):
    """ init a linear string between the start and end nodes. """
    dim = np.size(node_start)
    if dim != np.size(node_end):
        raise NameError ('The dimention of starting and ending nodes of the string should match!')
    string = np.zeros ((dim, numb_node))
    for ii in range (dim):
        string [ii] = np.linspace (node_start[ii], node_end[ii], numb_node)
    return string.T

def compute_string_tegent (alpha,
                           string,
                           delta_a = 0.001
                           ):
    """ compute the tangent vector of the string, it is normalized """
    tangent = np.zeros (string.shape)
    numb_node = string.shape[0]
    dim = string.shape[1]
    smooth_str = interp1d (alpha, string, axis=0, kind="linear")
    tangent[0]  = ( smooth_str(alpha[ 0] + delta_a) - smooth_str(alpha[ 0]) ) / delta_a
    tangent[-1] = ( smooth_str(alpha[-1]) - smooth_str(alpha[-1] - delta_a) ) / delta_a
    for ii in range (1, numb_node-1):
        tangent[ii] = ( smooth_str(alpha[ii] + delta_a) - smooth_str(alpha[ii] - delta_a ) ) / (delta_a * 2.)
    norm_t = np.sqrt (np.sum(np.multiply (tangent, tangent), axis=1))
    for ii in range (numb_node):
        tangent[ii] = tangent[ii] / norm_t[ii]
    return tangent

def string_update_rhs (models, compute_force,
                       dt,
                       step,
                       string):
    """ compute the dt * force """
    return dt * compute_force (models, step, string)

def update_string_Euler (models, compute_force,
                         dt,
                         step,
                         string):
    incr = string_update_rhs (models, compute_force, dt, step, string)
    return string + incr

def update_string_RK2 (models, compute_force, dt, step, string):
    my_step = int(step * 2)
    in_k1 = string
    k1 = string_update_rhs (models, compute_force, dt, my_step+0, in_k1)
    in_k2 = string + 0.5 * k1
    k2 = string_update_rhs (models, compute_force, dt, my_step+1, in_k2)
    return string + k2

def update_string_RK4 (models, compute_force, dt, step, string):
    my_step = int(step * 4)
    in_k1 = string
    k1 = string_update_rhs (models, compute_force, dt, my_step+0, in_k1)
    in_k2 = string + 0.5 * k1
    k2 = string_update_rhs (models, compute_force, dt, my_step+1, in_k2)
    in_k3 = string + 0.5 * k2
    k3 = string_update_rhs (models, compute_force, dt, my_step+2, in_k3)
    in_k4 = string + 1.0 * k3
    k4 = string_update_rhs (models, compute_force, dt, my_step+3, in_k4)
    return string + (1./6.) * k1 + (1./3.) * k2 + (1./3.) * k3 + (1./6.) * k4

def compute_string (models, compute_force,              # function for computing the force
                    string,                     # the input string
                    dt = 0.05,                  # artificial time step for updating the string
                    max_iter = 50000,             # maximum allowed number of iterations
                    start_iter = 0,
                    weighting = [[0,1],[1,1]]   # weighting of string discretization
                    ):            
    """ compute the string"""
    factor_Q = 1.1
    numb_node = string.shape[0]
    dim = np.size(string[0])
    # check validity of the inputs
    if dim != np.size(string[-1]):
        raise NameError ('The dimention of starting and ending nodes of the string should match!')
    if numb_node <= 2:
        raise NameError ('The number of nodes on string should be larger than 2')
    # initialize
    alpha_eq    = np.linspace (0, 1, numb_node)
    incr_hist   = [[]]

    conv_file = open ("conv.out", "w")
    # starts the main loop
    for ii in range (start_iter, max_iter):
        # update the string
        string = update_string_Euler (models, compute_force, dt, ii, string)
        # string = update_string_RK4 (compute_force, dt, string)
        # discretize the string
        string = StringUtils.resample_string (string, numb_node, weighting)
        # compute the max norm force as measure of convergence
        if ii != start_iter :
            norm_string = string
            diff_string = norm_string - norm_string_old
            norm_string_old = np.copy (norm_string)
            diff = np.sqrt (np.sum (np.multiply (diff_string, diff_string), axis=1))
            diff_inf = np.max( diff )
            new_item = np.array([ii, diff_inf])
            new_item = new_item[np.newaxis,:]
            if np.size (incr_hist) == 0:
                incr_hist = new_item
            else:
                incr_hist = np.append (incr_hist, new_item, axis=0)
            logging.info ("string %06d: updated with timestep %e . String difference is %e", ii+1, dt, diff_inf)
            print(diff_inf)
            if diff_inf > 2.8*1e-2:
                conv_file.write (str(ii) + " " + str(diff_inf) + "\n")
            else:
                conv_file.close ()
                break
        else :
            norm_string_old = string
            logging.info ("string %06d: updated with timestep %e .", ii+1, dt)
#    print incr_hist
    conv_file.close ()
    return string    

def main ():
    #models=['graph.000.pb','graph.001.pb','graph.002.pb','graph.003.pb']
    models=['graph.001.pb']
    string=np.loadtxt('trj.dat')
    #string = init_linear_string(np.array([-1, 0]), np.array([1, 0]), 10)

    string = compute_string (models, string_compute_force, string)
    np.savetxt('result1.out', string)
    
if __name__ == "__main__":
    main ()
