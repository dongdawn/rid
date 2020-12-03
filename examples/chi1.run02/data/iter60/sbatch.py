import os

#for i in [10,20,30,40,50,60]:
for i in [60]:
    dirname='iter'+str(i)
    os.system('cp *.py %s' %dirname)
    os.chdir(dirname)
    for j in range(4):
        
        os.mkdir('00%d' %j)
        os.chdir('00%d' %j)
        os.system('cp /home/dongdong/SCR/chi1.run02/data/sub ./')
        os.system('cp -r ../data ./')
        os.system('sbatch sub')
        os.chdir('..')
    os.chdir('..')
