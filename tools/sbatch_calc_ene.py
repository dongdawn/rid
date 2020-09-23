import os
import numpy as np
import pathlib
for it in range(120,141):
    for en in range(12):
        #print(en)
        trjdir='/home/dongdong/SCR/chi3.run01/iter.%06d/00.enhcMD/%03d/' %(it,en)
        outdir='/home/dongdong/SCR/chi3.run01.ene/iter.%06d/00.enhcMD/%03d/' %(it,en)
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        plmf=trjdir+'plm.out'
        data=np.loadtxt(plmf)[:,1:]
        outfile=outdir+'fene.out'
        outdata=outdir+'data.raw'
        np.savetxt(outdata,data)

        os.system('python calc_ene.py -m ./*.pb -d %s -o %s' %(outdata, outfile))

