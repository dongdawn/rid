#!/usr/bin/env python3
import os
import mdtraj as md
import numpy as np
import pickle
import pathlib
import re
def replace (file_name, pattern, subst) :
    file_handel = open (file_name, 'r')
    file_string = file_handel.read ()
    file_handel.close ()
    file_string = ( re.sub (pattern, subst, file_string) )
    file_handel = open (file_name, 'w')
    file_handel.write (file_string)
    file_handel.close ()

def run_md(pdbname):
    os.system('echo -e "10\n1\n" | gmx pdb2gmx -f %s.pdb -o processed.gro -ignh -heavyh > grompp.log 2>&1' %pdbname)
    os.system('gmx editconf -f processed.gro -o newbox.gro -d 0.9 -c -bt dodecahedron')
    os.system('gmx solvate -cp newbox.gro -cs spc216.gro -o solv.gro -p topol.top > sol.log 2>&1')
    os.system('gmx grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2 > grompp_ion.log 2>&1')
    os.system('echo -e "13\n" | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top -pname NA -nname CL -neutral')
    os.system('gmx grompp -f minim.mdp -c solv_ions.gro -p topol.top -o em.tpr -maxwarn 1 > grompp_em.log 2>&1')
    os.system('gmx mdrun -deffnm em -v -nt 4')
    os.system('gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr -r em.gro -maxwarn 1 > grompp_nvt.log 2>&1')
    os.system('gmx mdrun -deffnm nvt -v -nt 4')
    os.system('gmx grompp -f npt.mdp -c nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -r nvt.gro -maxwarn 1 > grompp_npt.log 2>&1')
    os.system('gmx mdrun -deffnm npt -v -nt 4')

def mk_posre(pdbname):
    window_num=5
    qa=pickle.load(open('/home/dongdong/SCR/jxy/CASP13_Refinement/QA_predict/%s.pkl' %pdbname,'rb'))
    ave=[]
    for i in range(len(qa['pred_lDDT_local'])//window_num):
        ave.append(np.mean(qa['pred_lDDT_local'][i*window_num:i*window_num+window_num]))
    if len(qa['pred_lDDT_local'])%window_num != 0:
        ave.append(np.mean(qa['pred_lDDT_local'][len(qa['pred_lDDT_local'])//window_num*window_num:]))
    normalized = (ave-np.min(ave))/(np.max(ave)-np.min(ave))
    np.savetxt('normalized.txt',normalized)
    biased_ang=[]
    for i in range(len(normalized)-1):
        if normalized[i]<=0.5:
            biased_ang.append(range(i*window_num,(i+1)*window_num))
    if normalized[len(normalized)-1]<=0.5:
        biased_ang.append(range((len(normalized)-1)*window_num,len(qa['pred_lDDT_local'])))
    np.savetxt('biased_res.txt',biased_ang,fmt='%d')
    list_biased_ang=[]
    for ba in biased_ang:
        for aa in ba:
            list_biased_ang.append(aa)
    os.system('cp /home/dongdong/SCR/jxy/source/jsons/phipsi_selected.json ./')
    replace('phipsi_selected.json','.*selected_index.*','    "selected_index":  %s,' %list_biased_ang)
    array_r0=0.5*(1-normalized)+0.1
    structure='nvt.gro'
    #   kappa=0.025      #kcal/mol/A2   *4.184*100 
    #kappa=15             #kj/mol/nm2
    t_ref=md.load(structure,top=structure)
    topology = t_ref.topology
    ca_atoms=topology.select('name CA')+1
    wf=open('posre.itp.templ','w')
    wf.write('[ position_restraints ]\n;  i funct       g         r(nm)       k\n')
    for i in range(len(ca_atoms)):
        wf.write('%d    2        1          %f       TEMP\n' %(ca_atoms[i],array_r0[i//window_num]))
    wf.close()

def mk_rid(pdbname):
    mol_dir='/home/dongdong/SCR/jxy/source/mol/'+pdbname
    pathlib.Path(mol_dir).mkdir(parents=True, exist_ok=True)
    os.system('cp topol.top %s' %mol_dir)
    for i in range(8):
        os.system('cp npt.gro %s/conf00%d.gro' %(mol_dir,i))
    os.system('cp npt.gro %s/conf.gro' %mol_dir)
    os.system('cp posre.itp.templ %s/posre.itp' %mol_dir)
    os.system('cp /home/dongdong/SCR/jxy/source/mol/*.mdp %s' %mol_dir)
    os.chdir('/home/dongdong/SCR/jxy/source/')
    os.system('python gen.py rid ./jsons/default_gen.json /scratch/gpfs/dongdong/jxy/%s/phipsi_selected.json ./mol/%s/ -o /home/dongdong/SCR/jxy/%s.run02' %(pdbname,pdbname,pdbname))
    os.chdir('/home/dongdong/SCR/jxy/%s' %pdbname)

pdbname = ['R0986s1']
pdbname = ['R0986s2']
pdbname = ['R0981-D5']
for pp in pdbname:
    pathlib.Path(pp).mkdir(parents=True, exist_ok=True)
    os.system('cp CASP13_Refinement/Model/%s.pdb %s' %(pp,pp))
    os.system('cp mdp/* %s' %pp)
    os.chdir(pp)
    run_md(pp)
    mk_posre(pp)
    mk_rid(pp)
    os.chdir('..')
