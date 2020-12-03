#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-31
# @Author  : WDD 
# @Link    : https://github.com/dongdawn
# @Version : v1
import sys
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.pylab as pl
import seaborn
def parse_arg():
    parser = argparse.ArgumentParser(description='This is a program to plot each CV as time, python plot_eachCV_one.py -infiledir ./ -num_iter 100 -num_walkers 12 -cv_index 8 -time_len 2 -outfile CV8.png')
    parser.add_argument('-infiledir', dest='infiledir', help="input file dir ", required=True)
    parser.add_argument('-num_iter', dest='num_iter', help="total number of iterations ", default=10)
    parser.add_argument('-num_walkers', dest='num_walkers', help="total number of walkers ", default=1)
    parser.add_argument('-cv_index', dest='cv_index', help="the index of CV ploted, start from 1 ", default=1)
    parser.add_argument('-time_len', dest='time_len', help="the length of time of each iteration (ns)", default=2)
    parser.add_argument('-outfile', dest='outfile', help="output file", required=True)
    arg = parser.parse_args()
    return arg.infiledir, arg.num_iter, arg.num_walkers, arg.cv_index, arg.time_len, arg.outfile

font_path = '/home/dongdong/tigress/calibribold.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=19)
leg_prop = font_manager.FontProperties(fname=font_path, size=17)

def plot_iter(filedir,num_iter,num_walkers,cv_index,time_len,figout):
    colors = pl.cm.jet(np.linspace(0,1,num_walkers))
    fig = plt.figure(figsize=(10,8))
    for wal in range(int(num_walkers)):
        walker_name='%03d' %wal
        all_cv=[]
        all_time=[]
        for it in range(int(num_iter)):
            iteration="%06d" %it
            filename=filedir+"iter."+str(iteration)+"/00.enhcMD/"+str(walker_name)+"/plm.out"
            data = np.loadtxt(filename)
            time = data[:,0]/1000+it*float(time_len)
            ang = data[:,int(cv_index)]
            all_cv.extend(list(ang))
            all_time.extend(list(time))
            #if np.max(ang)>4:
            #    print(filename)
        ax=fig.add_subplot(int(num_walkers)/4,4,wal+1)
        ax.plot(all_time,all_cv,alpha=0.8)
        ax.set_ylabel('distance (nm)',fontproperties=font_prop)
        ax.set_xlabel('time (ns)',fontproperties=font_prop)
        plt.ylim(0,6.5)
    #plt.xlim(0,np.max(time))
    #leg=plt.legend(loc=1, labelspacing=0.1, prop=leg_prop, scatterpoints=1, markerscale=1, numpoints=1,handlelength=1.5)
    #leg.get_frame().set_linewidth(0.0)
    #leg.get_frame().set_alpha(0.1)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(font_prop)
            label.set_fontsize(10)
    plt.tight_layout(pad=0.4, w_pad=0.6, h_pad=0.6)
    plt.savefig(figout,dpi=600,bbox_inches='tight')
    plt.show()
#filedir="/home/wdd/reinforcedMD/deep.fe/source/ala-n_test/out/"
if __name__ == "__main__":
    infiledir, num_iter,num_walkers, cv_index, time_len, outfile= parse_arg()
    plot_iter(infiledir,num_iter,num_walkers, cv_index,time_len,outfile)

