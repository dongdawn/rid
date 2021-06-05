import os
import pathlib
TARGETs = ['R0974s1', 'R0986s1', 'R0986s2', 'R1002-D2']
indexs=0

#workdir='/home/dongdong/SCR/jxy/%s.run03/' %TARGETs[indexs]
workdir='/home/dongdong/SCR/jxy/%s/' %TARGETs[indexs]
for it in range(7,18):
    for en in range(8):
        #print(en)
        trjname=workdir+'iter.%06d/00.enhcMD/%03d/md_nopbc.xtc' %(it,en)
        outname=workdir+'iter.%06d/00.enhcMD/%03d/tIC.npy' %(it,en)
        os.system('../scripts/proj_tICA.py --target %s --top ../init/%s.pdb --traj %s --output %s' %(TARGETs[indexs],TARGETs[indexs],trjname,outname))
    
        
