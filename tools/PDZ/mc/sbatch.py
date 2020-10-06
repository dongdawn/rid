import os
for it in [70]:
    dirname='iter%d' %it
    print(dirname)
    for ee in range(4):
        os.system('cp mc1d_2d_pdz3.py mc.sh crystal.cv /home/dongdong/SCR/pdz.run02/mc3/%s/00%d' %(dirname,ee))
        os.chdir('/home/dongdong/SCR/pdz.run02/mc3/%s/00%d' %(dirname,ee))
        os.system('rm -f *.dat *.out')
        os.system('sbatch mc.sh')
    
