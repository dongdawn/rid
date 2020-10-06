import os
import pathlib
ori_dir='/home/dongdong/SCR/jxy/R0974s1'
tar_dir='/home/dongdong/SCR/jxy/R0974s1'
native_pdb='/home/dongdong/SCR/jxy/CASP13_Refinement/Native/R0974s1.pdb'
for it in range(12,18):
    for en in range(8):
        #print(en)
        trjdir='%s/iter.%06d/00.enhcMD/%03d/' %(ori_dir,it,en)
        outdir='%s/iter.%06d/00.enhcMD/%03d/' %(tar_dir,it,en)
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(outdir+'/pdb').mkdir(parents=True, exist_ok=True)
        trjname=trjdir+'traj_comp.xtc'
        outtrj=outdir+'md_nopbc.xtc'
        #print(trjname)
        os.system('echo -e "1\n" | gmx trjconv -s topol.tpr -f %s -o %s -pbc mol -ur compact' %(trjname, outtrj))
        os.system('echo -e "1\n" | gmx trjconv -sep -f %s -o %s/pdb/conf.pdb' %(outtrj, outdir))
        os.system('rm -f \#*')
        for i in range(1001):
            os.system('./TMscore %s/pdb/conf%d.pdb %s >%s/pdb/conf%d.sc' %(outdir,i,native_pdb,outdir,i))
