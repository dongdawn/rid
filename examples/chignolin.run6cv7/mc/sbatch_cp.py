import os
import pathlib

ii=16
work_dir='/home/dongdong/SCR/chignolin.run6cv7/'
for m in range(4):
    print(m)
    outdir='iter%d/%03d' %(ii,m)
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    os.system('cp %siter.0000%d/02.train/%03d/graph.pb %s' %(work_dir,ii,m,outdir))
    os.system('cp sub30_0 %s/sub%d_%d' %(outdir,ii,m))
    os.system('cp mc1d_2d.py %s' %outdir)
    os.chdir(outdir)
    os.system('sbatch sub%d_%d' %(ii,m))
    os.chdir('/home/dongdong/SCR/chignolin.run6cv7/mc')
