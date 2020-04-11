#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Date    : 20200411
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
import matplotlib.patches as patches
def parse_arg():
    parser = argparse.ArgumentParser(description='This is a program to plot distance and rmsd, python plot_dist_rmsd.py -rmsddir . -plmdir . -num_iter 100 -num_walkers 30 -outfile dist_rmsd')
    parser.add_argument('-rmsddir', dest='rmsddir', help="input file dir ", required=True)
    parser.add_argument('-plmdir', dest='plmdir', help="input file dir ", required=True)
    parser.add_argument('-num_iter', dest='num_iter', help="total number of iterations ", default=10)
    parser.add_argument('-num_walkers', dest='num_walkers', help="total number of walkers ", default=1)
    parser.add_argument('-outfile', dest='outfile', help="output file", required=True)
    arg = parser.parse_args()
    return arg.rmsddir, arg.plmdir, arg.num_iter, arg.num_walkers, arg.outfile

font_path = '/home/dongdong/tigress/calibribold.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=12)
leg_prop = font_manager.FontProperties(fname=font_path, size=10)
cmap = plt.cm.get_cmap("jet_r")

def plot_iter(rmsddir,plmdir,num_iter,num_walkers,figout):
    colors = pl.cm.jet(np.linspace(0,1,num_walkers))
    fig = plt.figure(figsize=(10,8))
    for en in range(num_walkers):
        allrmsd=[]
        alldist=[]
        for it in range(num_iter):
            rmsdfile=rmsddir+'/iter.%06d/00.enhcMD/%03d/rmsd.xvg' %(it,en)
            os.system("sed -i 's/^@/#/g' %s " %rmsdfile)
            plmfile=plmdir+'/iter.%06d/00.enhcMD/%03d/plm.out' %(it,en)
            rmsd=np.loadtxt(rmsdfile)[:,1]
            distance=np.mean(np.loadtxt(plmfile)[:,-3:],axis=1)
            allrmsd.extend(list(rmsd))
            alldist.extend(list(distance))
        sub = fig.add_subplot(int(num_walkers)/6,6,en+1)
        sub.scatter(allrmsd,alldist,c=np.arange(len(allrmsd)),cmap=cmap,lw = 0,s=8)
        sub.set_xlabel(r'rmsd (nm)',fontproperties=font_prop)
        sub.set_ylabel(r'distance (nm)',fontproperties=font_prop)
        sub.tick_params(direction="in", length=1)
        sub.add_patch(patches.Rectangle((0, 0.5),0.2,0.5, linewidth=1,edgecolor='r',facecolor='none'))
        plt.ylim(0.5,3)
        plt.xlim(0,0.75)
        #ax.set_yticks(np.linspace(0,2,5))
        #ax.set_yticklabels([0,0.5,1,2])
        #leg=plt.legend(loc=1, labelspacing=0.1, prop=leg_prop, scatterpoints=1, markerscale=1, numpoints=1,handlelength=1.5)
        #leg.get_frame().set_linewidth(0.0)
        #leg.get_frame().set_alpha(0.1)
        for label in (sub.get_xticklabels() + sub.get_yticklabels()):
            label.set_fontproperties(font_prop)
            label.set_fontsize(10)
    plt.savefig('dist_rmsd.png',dpi=300,bbox_inches='tight')
    plt.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.4)
    plt.show()

if __name__ == "__main__":
    rmsddir, plmdir, num_iter,num_walkers, outfile= parse_arg()
    plot_iter(rmsddir, plmdir, int(num_iter), int(num_walkers),outfile)

