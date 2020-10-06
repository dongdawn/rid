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
n=30
colors = pl.cm.jet(np.linspace(0,1,n))
def parse_arg():
    parser = argparse.ArgumentParser(description='This is a program to plot each CV as time, python plot_eachCV.py -infiledir "/home/wdd/reinforcedMD/deep.fe/source/ala-n_test/out/" -num_iter 100 -num_walkers 30 -outfile cry_')
    parser.add_argument('-infiledir', dest='infiledir', help="input file dir ", required=True)
    parser.add_argument('-num_iter', dest='num_iter', help="total number of iterations ", default=10)
    parser.add_argument('-num_walkers', dest='num_walkers', help="total number of walkers ", default=1)
    parser.add_argument('-outfile', dest='outfile', help="output file", required=True)
    arg = parser.parse_args()
    return arg.infiledir, arg.num_iter, arg.num_walkers, arg.outfile

font_path = '/home/dongdong/tigress/calibribold.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=19)
leg_prop = font_manager.FontProperties(fname=font_path, size=17)

def singleSD(nList,refList):
    length=len(nList)
    sumVar=0.0
    for i in range(length):
        sub=nList[i]-refList[i]
        if np.abs(sub)>180:
            sub=360-np.abs(sub)
        sumVar+=sub**2
    return ((float(sumVar)/float((length)))**0.5)

def plot_iter(filedir,num_iter,num_walkers,figout):
    for wal in range(int(num_walkers)):
        fig, ax = plt.subplots(figsize=(4.1,4))
        cmap = plt.cm.get_cmap("jet_r")
        cmap.set_over("white")
        walker_name='%03d' %wal
        dataR=[]
        dataD=[]
        dataCry=np.loadtxt('crystal.cv')/3.14*180
        for it in range(int(num_iter)):
            iteration="%06d" %it
            filename=filedir+"iter."+str(iteration)+"/00.enhcMD/"+str(walker_name)+"/plm.out"
            data = np.loadtxt(filename)
            for ii in range(len(data)):
                tempR=singleSD(data[ii,1:10]/3.14*180,dataCry)
                dataR.append(tempR)
                dataD.append((data[ii,10]+data[ii,11])/2.0)
        ax.scatter(dataR,dataD,c=np.arange(len(dataR)),cmap=cmap,lw = 0,s=20)
        ax.set_ylabel('distance (nm)',fontproperties=font_prop)
        ax.set_xlabel(r'dev ($^\circ$)',fontproperties=font_prop)
        plt.ylim(0.5,3.8)
        plt.xlim(0,180)
    #leg=plt.legend(loc=1, labelspacing=0.1, prop=leg_prop, scatterpoints=1, markerscale=1, numpoints=1,handlelength=1.5)
    #leg.get_frame().set_linewidth(0.0)
    #leg.get_frame().set_alpha(0.1)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontproperties(font_prop)
            label.set_fontsize(16)
        plt.savefig(figout+str(wal)+'.png',dpi=300,bbox_inches='tight')
        plt.show()
#filedir="/home/wdd/reinforcedMD/deep.fe/source/ala-n_test/out/"
if __name__ == "__main__":
    infiledir, num_iter,num_walkers, outfile= parse_arg()
    plot_iter(infiledir,num_iter,num_walkers,outfile)

