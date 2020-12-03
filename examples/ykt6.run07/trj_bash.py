import os
for it in range(1,19):
    for en in range(5,6):
        #print(en)
        trjdir='/home/dongdong/SCR/ykt6.run07/iter.%06d/00.enhcMD/%03d/' %(it,en)
        
        trjname=trjdir+'traj_comp.xtc'
        outtrj='walker%d_iter%d_md_nopbc.xtc' %(en,it)
        outfit='walker%d_iter%d_fit.xtc' %(en,it)
        #print(trjname)
        os.system('echo -e "1\n" | gmx trjconv -s nosol.tpr -f %s -o %s -pbc mol -ur compact' %(trjname, outtrj))
        os.system('echo -e "1\n1\n" | gmx trjconv -s nosol.tpr -f %s -o %s -fit rot+trans' %(outtrj, outfit))
        
