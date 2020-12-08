import os
import pathlib

ii=30
work_dir='/home/dongdong/SCR/chignolin.run18cv'
for m in range(4):
    print(m)
    outdir='%03d' %m
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    os.system('cp %s/02.train.iter30/%03d/graph.pb %s' %(work_dir,m,outdir))
    os.system('cp sub30_0 %s/sub%d_%d' %(outdir,ii,m))
    os.system('cp mc1d_2d.py %s' %outdir)
    os.chdir(outdir)
    os.system('sbatch sub%d_%d' %(ii,m))
    os.chdir('/home/dongdong/SCR/chignolin.run18cv/mc.iter30')
