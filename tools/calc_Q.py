from MDAnalysis.analysis import contacts
import os
os.chdir('/home/dongdong/SCR/pdz.run02')
chain=['A','B']
t_ref=mda.Universe('begin.pdb',top='begin.pdb')
group1=t_ref.select_atoms('segid A and (name C* or name N* or name O* or name S*)')
num_walkers=16
for en in range(num_walkers):
    for it in range(81,85):
        dirname='iter.%06d/00.enhcMD/%03d/' %(it,en)
        os.system('cp begin.pdb md-nosol.tpr trj.sh index_p.ndx %s' %dirname)
        os.chdir(dirname)
        os.system('sh trj.sh')
        trjname='md_done.xtc'
        u=mda.Universe("begin.pdb",trjname)
        filename='Q_heavyatoms.cs'
        wf=open(filename,'w')
        sel2='segid B and (name C* or name N* or name O* or name S*)'
        group2=t_ref.select_atoms(sel2)
        nc=contacts.Contacts(u,selection=("segid A and (name C* or name N* or name O* or name S*)",sel2),refgroup=(group1,group2),method='soft_cut')
        nc.run()
        bound=nc.timeseries[:,1]
        for b in bound:
            wf.write(str(b)+'\n')
        wf.close()
        os.chdir('/home/dongdong/SCR/pdz.run02')
        
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
font_path = '/home/dongdong/tigress/calibribold.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=16)
leg_prop = font_manager.FontProperties(fname=font_path, size=10)
fig = plt.figure(figsize=(10,9))
num_walkers=16
os.chdir('/home/dongdong/SCR/pdz.run02')
for en in range(num_walkers):
    allQ=[]
    for it in range(85):
        dirname='iter.%06d/00.enhcMD/%03d/' %(it,en)
        os.chdir(dirname)
        filename='Q_heavyatoms.cs'
        Q=np.loadtxt(filename)
        os.chdir('/home/dongdong/SCR/pdz.run02')
        allQ.extend(list(Q))
    #allrmsd=np.reshape(allrmsd,(1,-1))[0]
    #print(np.array(range(len(allrmsd))))
    #print(allrmsd)
    sub = fig.add_subplot(int(num_walkers)/4,4,en+1)
    sub.plot(np.array(range(len(allQ)))/1000.0*2,allQ,lw=0.6)
    sub.set_ylabel(r'Q',fontproperties=font_prop)
    sub.set_xlabel(r'time (ns)',fontproperties=font_prop)
    sub.plot(np.array(range(len(allQ)))/1000.0*2,[0.85]*len(allQ),lw=0.8)
    sub.tick_params(direction="in", length=1)
    plt.ylim(0.,1.)
    #plt.xlim(25,55)
    sub.set_xticks(np.linspace(0,150,4))
    #ax.set_yticklabels([0,0.5,1,2])
    #leg=plt.legend(loc=1, labelspacing=0.1, prop=leg_prop, scatterpoints=1, markerscale=1, numpoints=1,handlelength=1.5)
    #leg.get_frame().set_linewidth(0.0)
    #leg.get_frame().set_alpha(0.1)
    for label in (sub.get_xticklabels() + sub.get_yticklabels()):
        label.set_fontproperties(font_prop)
        label.set_fontsize(15)

plt.tight_layout(pad=3, w_pad=0.8, h_pad=0.4)
plt.savefig('/home/dongdong/SCR/pdz.run02/Q.png',dpi=600,bbox_inches='tight')
plt.show()
