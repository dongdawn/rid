#!/usr/bin/env python

import os
import sys

import pickle
import numpy as np
import argparse

import mdtraj

import libfeaturizer

WORK_HOME = '/'.join(os.path.abspath(__file__).split("/")[:-2])
TARGETs = ['R0974s1', 'R0986s1', 'R0986s2', 'R1002-D2']
RESs = {'R0986s1': np.arange(14, 97)}
FEATUREs = {'R1002-D2': 'contact'}
LAG = 100 

class Job:
    def __init__(self, name):
        self.name = name
    def set_topology_files(self, top_fn):
        self.ref_fn = '%s/native/%s.pdb'%(WORK_HOME, self.name)
        self.ini_fn = '%s/init/%s.pdb'%(WORK_HOME, self.name)
        self.top_fn = top_fn
    def read_topology(self):
        self.atomSelection = 'name CA'
        #
        # read PDB files
        self.ref_top = mdtraj.load(self.ref_fn)
        self.ini_top = mdtraj.load(self.ini_fn)
        self.md_top = mdtraj.load(self.top_fn)
        #
        # set common residues
        self.ref_res = []
        for i_atm in self.ref_top.topology.select(self.atomSelection):
            self.ref_res.append(self.ref_top.topology.atom(i_atm).residue.resSeq)
        self.ini_res = []
        for i_atm in self.ini_top.topology.select(self.atomSelection):
            self.ini_res.append(self.ini_top.topology.atom(i_atm).residue.resSeq)
        self.md_res = []
        for i_atm in self.md_top.topology.select("protein and %s"%self.atomSelection):
            self.md_res.append(self.md_top.topology.atom(i_atm).residue.resSeq)
        self.res = np.intersect1d(self.ref_res, self.ini_res)
        if self.name in RESs:
            self.res = np.intersect1d(self.res, RESs[self.name])
        for residue in self.res:
            if residue not in self.md_res:
                sys.exit("Residue %d is missing.\n"%residue)
        #
        self.ref_calphaIndex = []
        for i_atm in self.ref_top.topology.select(self.atomSelection):
            if self.ref_top.topology.atom(i_atm).residue.resSeq in self.res:
                self.ref_calphaIndex.append(i_atm)
        self.ini_calphaIndex = []
        for i_atm in self.ini_top.topology.select(self.atomSelection):
            if self.ini_top.topology.atom(i_atm).residue.resSeq in self.res:
                self.ini_calphaIndex.append(i_atm)
        self.md_calphaIndex = []
        for i_atm in self.md_top.topology.select("protein and %s"%self.atomSelection):
            if self.md_top.topology.atom(i_atm).residue.resSeq in self.res:
                self.md_calphaIndex.append(i_atm)
        #
        self.ref = self.ref_top.atom_slice(self.ref_calphaIndex)
        self.ini = self.ini_top.atom_slice(self.ini_calphaIndex)
        self.md = self.md_top.atom_slice(self.md_calphaIndex)
    def read_pdb(self, pdb_fn):
        pdb = mdtraj.load(pdb_fn.path())
        calphaIndex = []
        for i_atm in pdb.topology.select(self.atomSelection):
            if pdb.topology.atom(i_atm).residue.resSeq in self.res:
                calphaIndex.append(i_atm)
        pdb = pdb.atom_slice(calphaIndex)
        return pdb

def main():
    arg = argparse.ArgumentParser(prog='proj_tICA')
    arg.add_argument('--target', dest='name', help='target ID', choices=TARGETs, required=True)
    arg.add_argument('--top', dest='top_fn', help='MD simulation topology file', required=True)
    arg.add_argument('--traj', dest='traj_fn_s', help='trajectory files', nargs='*')
    arg.add_argument('--output', dest='out_fn', help='output NPY file', default=None)

    if len(sys.argv) == 1:
        return arg.print_help()
    arg = arg.parse_args()

    job = Job(arg.name)
    job.set_topology_files(arg.top_fn)
    job.read_topology()
    job.traj_fn_s = arg.traj_fn_s
    #
    use_contact = (FEATUREs.get(job.name, None) == 'contact')
    libfeaturizer.run(job, contact=use_contact)

    tICA_pkl_fn = '%s/tICA_model/%s.tICA.pkl'%(WORK_HOME, job.name)
    with open(tICA_pkl_fn, 'rb') as fp:
        tICA_model = pickle.load(fp, encoding='latin')
    #
    tICA_crd = tICA_model.transform(job.feature)
    if arg.out_fn is None:
        arg.out_fn = '%s.tICA.npy'%arg.name
    np.save(arg.out_fn, tICA_crd)

if __name__ == '__main__':
    main()
