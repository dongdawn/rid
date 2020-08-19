#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-01
# @Author  : WDD 
# @Link    : https://github.com/dongdawn
# @Version : v1
import sys
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
import pandas as pd
def parse_arg():
    parser = argparse.ArgumentParser(description='This is a program to plot each iteration, python plot_eachITER.py -infiledir ./ -walkers_index 0 -beg_iter 0 -end_iter 3 -outfile iteration0-3.png')
    parser.add_argument('-infiledir', dest='infiledir', help="input file dir ", required=True)
    parser.add_argument('-walkers_index', dest='walkers_index', help="the index of walkers ", default=0)
    parser.add_argument('-beg_iter', dest='beg_iter', help="plot from this iteration ", default=0)
    parser.add_argument('-end_iter', dest='end_index', help="the end of iteration when plotting", default=3)
    parser.add_argument('-outfile', dest='outfile', help="output file", required=True)
    arg = parser.parse_args()
    return arg.infiledir, arg.walkers_index, arg.beg_iter, arg.end_index,  arg.outfile

font_path = '/home/wdd/calibribold.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=19)
leg_prop = font_manager.FontProperties(fname=font_path, size=17)
def plot_iter(filedir,walkers_index,beg_iter,end_iter,figout):
    alldataset=pd.DataFrame()
    fig, ax = plt.subplots(figsize=(6,6))
    for it in np.arange(int(beg_iter),int(end_iter)+1):
        iteration="%06d" %it
        walkers_ind="%03d" %int(walkers_index)
        filename=filedir+"iter."+str(iteration)+"/00.enhcMD/"+str(walkers_ind)+"/plm.out"
        data = np.loadtxt(filename)
        dataset = pd.DataFrame(data[:,1:])
        dataset['iteration']=[it]*len(data)
        alldataset=alldataset.append(dataset)
    g = sns.pairplot(alldataset,vars=alldataset.columns[:-1], hue="iteration")
    #g.set(xlim=(-3.2,3.2),ylim=(-3.2, 3.2))
    #leg=plt.legend(loc=1, labelspacing=0.1, prop=leg_prop, scatterpoints=1, markerscale=1, numpoints=1,handlelength=1.5)
    #leg.get_frame().set_linewidth(0.0)
    #leg.get_frame().set_alpha(0.1)
    #for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #    label.set_fontproperties(font_prop)
    #    label.set_fontsize(16)
    plt.savefig(figout,dpi=300,bbox_inches='tight')
    #plt.show()
if __name__ == "__main__":
    infiledir,walkers_index, beg_iter, end_iter, outfile= parse_arg()
    plot_iter(infiledir,walkers_index, beg_iter, end_iter, outfile)

