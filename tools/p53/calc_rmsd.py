#! /usr/bin/python
# -*- coding: utf-8 -*-
# @Date    : 2020-04-10
# @Author  : WDD 
# @Link    : https://github.com/dongdawn
# @Version : v1
import sys
import os
import numpy as np
import argparse
import pathlib
def parse_arg():
    parser = argparse.ArgumentParser(description='This is a program to calculate rmsd in each enhanced sampling, python calc_rmsd.py -infiledir ./ -outfiledir ./ -iter_start 0 -iter_end 10 -num_walkers 30 -tprfile crystal_nowater.tpr -reference crystal_protein.gro -indexfile index_p.ndx')
    parser.add_argument('-infiledir', dest='infiledir', help="input file dir ", required=True)
    parser.add_argument('-outfiledir', dest='outfiledir', help="input file dir ", required=True)
    parser.add_argument('-iter_start', dest='iter_start', help="start index of iteration ", default=0)
    parser.add_argument('-iter_end', dest='iter_end', help="end index of iteration ", default=10)
    parser.add_argument('-num_walkers', dest='num_walkers', help="total number of walkers ", default=1)
    parser.add_argument('-tprfile', dest='tprfile', help="gromacs tpr file", required=True)
    parser.add_argument('-reference', dest='reference', help="reference for rmsd", required=True)
    parser.add_argument('-indexfile', dest='indexfile', help="gromacs index file", required=True)
    arg = parser.parse_args()
    return arg.infiledir, arg.outfiledir, arg.iter_start, arg.iter_end, arg.num_walkers, arg.tprfile, arg.reference, arg.indexfile

def calc_rmsd(infiledir,outfiledir,iter_start,iter_end,num_walkers,tpr_file,reference_file,index_file):
    for it in range(int(iter_start),int(iter_end)):
        for en in range(int(num_walkers)):
            trjdir=infiledir+'/iter.%06d/00.enhcMD/%03d/' %(it,en)
            outdir=outfiledir+'/iter.%06d/00.enhcMD/%03d/' %(it,en)
            pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
            trjname=trjdir+'traj_comp.xtc'
            outtrj=outdir+'md_nopbc.xtc'
            outrmsd=outdir+'rmsd.xvg'
            os.system('echo -e "1\n" | gmx trjconv -s %s -f %s -o %s -pbc mol -ur compact > %s/trjconv.log 2>&1' %(tpr_file,trjname, outtrj,outdir))
            os.system('echo -e "11\n11\n" | gmx rms -s %s -f %s -o %s -n %s > %s/rmsd.log 2>&1' %(reference_file, outtrj, outrmsd,index_file,outdir))

if __name__ == "__main__":
    trjdir,outdir,iter_start,iter_end,num_walkers,tpr_file,reference_file,index_file=parse_arg()
    calc_rmsd(trjdir,outdir,iter_start,iter_end,num_walkers,tpr_file,reference_file,index_file)
