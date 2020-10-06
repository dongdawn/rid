#!/usr/bin/env python

import os
import sys
import numpy as np
import itertools

import mdtraj
import msmbuilder
import msmbuilder.featurizer as featurizer

def ContactFeaturizer(job, traj):
    if 'featurizer' not in dir(job):
        job.featurizer = featurizer.ContactFeaturizer(scheme='ca')
    return job.featurizer.partial_transform(traj)

def SuperposeFeaturizer(job, traj):
    if 'featurizer' not in dir(job):
        atomIndex = np.arange(len(job.ref_calphaIndex))
        job.featurizer = []
        job.featurizer.append(featurizer.SuperposeFeaturizer(np.arange(len(job.ref_calphaIndex)), job.ref))
        job.featurizer.append(featurizer.SuperposeFeaturizer(np.arange(len(job.ini_calphaIndex)), job.ini))
    dij = np.hstack([f.partial_transform(traj) for f in job.featurizer])
    return dij

def run(job, contact=False):
    if contact:
        Featurizer = ContactFeaturizer
    else:
        Featurizer = SuperposeFeaturizer
    #
    job.feature = []
    for traj_fn in job.traj_fn_s:
        traj = mdtraj.load(traj_fn, top=job.md_top, atom_indices=job.md_calphaIndex)
        feature = Featurizer(job, traj)
        job.feature.append(feature)
