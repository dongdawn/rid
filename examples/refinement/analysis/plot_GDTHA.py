import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
font_path = '/home/dongdong/tigress/calibribold.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=16)
leg_prop = font_manager.FontProperties(fname=font_path, size=10)
fig = plt.figure(figsize=(8,7))
num_walkers=8
os.chdir('/home/dongdong/SCR/jxy/R0974s1')
for en in range(num_walkers):
    allrmsd=[]
    allTM=[]
    allTS=[]
    allHA=[]
    for it in range(12):
        dirname='iter.%06d/00.enhcMD/%03d/' %(it,en)
        for i in range(1000):
            filename=dirname+'pdb/conf%d.sc' %i
            rf=open(filename,'r')
            lines=rf.readlines()
            rmsd=lines[14].strip().split()[-1]
            TM=lines[16].strip().split()[2]
            TS=lines[18].strip().split()[1]
            HA=lines[19].strip().split()[1]
            allrmsd.append(float(rmsd))
            allTM.append(float(TM))
            allTS.append(float(TS))
            allHA.append(float(HA)*100)

    sub = fig.add_subplot(int(num_walkers)/2,3,en+1)
    sub.scatter(np.array(range(len(allHA)))/1000.0*6,allHA,s=0.4,alpha=1,c='red')
    print(np.max(allHA))
    sub.set_ylabel(r'GDT-HA',fontproperties=font_prop)
    sub.set_xlabel(r'time (ns)',fontproperties=font_prop)
    #sub.plot(np.array(range(len(allrmsd)))/1000.0*2,[0.15]*len(allrmsd),lw=0.9,c='black')
    sub.tick_params(direction="in", length=1)
    plt.ylim(40,92)
    #plt.xlim(25,55)
    #ax.set_yticks(np.linspace(0,2,5))
    #ax.set_yticklabels([0,0.5,1,2])
    #leg=plt.legend(loc=1, labelspacing=0.1, prop=leg_prop, scatterpoints=1, markerscale=1, numpoints=1,handlelength=1.5)
    #leg.get_frame().set_linewidth(0.0)
    #leg.get_frame().set_alpha(0.1)
    for label in (sub.get_xticklabels() + sub.get_yticklabels()):
        label.set_fontproperties(font_prop)
        label.set_fontsize(15)
#plt.savefig('rmsd5.png',dpi=600,bbox_inches='tight')
plt.tight_layout(pad=3, w_pad=0.8, h_pad=0.4)
plt.show()