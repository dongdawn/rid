import os

#for i in [10,20,30,40,50,60]:
for i in [50,40,30,20,10]:
    dirname='iter'+str(i)
    os.chdir(dirname)
    os.system('cp ../plot.pp1_2.py ./')
    os.system('python plot.pp1_2.py -m *.pb -o fes1d2.out')
    os.chdir('..')
