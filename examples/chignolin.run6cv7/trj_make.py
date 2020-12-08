import os
import pathlib
for it in range(15,17):
    for en in range(12):
        #print(en)
        trjdir='/home/dongdong/SCR/chignolin.run6cv7/iter.%06d/00.enhcMD/%03d/' %(it,en)
        outdir='iter.%06d/00.enhcMD/%03d/' %(it,en)
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        trjname=trjdir+'traj_comp.xtc'
        outtrj=outdir+'md_nopbc.xtc'
        outrmsd=outdir+'rmsd_cry.xvg'
        #print(trjname)
        os.system('echo -e "1\n" | gmx trjconv -s topol.tpr -f %s -o %s -pbc mol -ur compact' %(trjname, outtrj))
        os.system('echo -e "3\n3\n" | gmx rms -s crystal.pdb -f %s -o %s' %(outtrj, outrmsd))
