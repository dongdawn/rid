import os
import pathlib
workdir='/home/dongdong/SCR/jxy/R1002-D2.run02/'
for it in range(17,21):
    for en in range(8):
        #print(en)
        trjname=workdir+'iter.%06d/00.enhcMD/%03d/md_nopbc.xtc' %(it,en)
        outname=workdir+'iter.%06d/00.enhcMD/%03d/tIC.npy' %(it,en)
        os.system('../scripts/proj_tICA.py --target R1002-D2 --top ../init/R1002-D2.pdb --traj %s --output %s' %(trjname,outname))
    
        
