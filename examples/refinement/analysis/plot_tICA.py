import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import os
font_path = '/home/dongdong/tigress/calibribold.ttf'
font_prop = font_manager.FontProperties(fname=font_path, size=16)
leg_prop = font_manager.FontProperties(fname=font_path, size=10)

tic1=0
tic2=1
TARGETs = ['R0974s1', 'R0986s1', 'R0986s2', 'R1002-D2']
indexs=0
num_walkers=8
allx=[]
ally=[]
os.chdir('/home/dongdong/SCR/jxy/%s' %TARGETs[indexs])
for en in range(num_walkers):
    for it in range(18):
        dirname='iter.%06d/00.enhcMD/%03d' %(it,en)
        aaa=np.load('%s/tIC.npy' %dirname)
        onex=list(aaa[0][:,tic1])
        oney=list(aaa[0][:,tic2])
        for i in range(len(onex)):
            allx.append(onex[i])
            ally.append(oney[i])
            
fig = plt.figure(figsize=(4.8,4))
cmap = plt.cm.get_cmap("Spectral")
font_prop = font_manager.FontProperties(fname=font_path, size=22)
leg_prop = font_manager.FontProperties(fname=font_path, size=19)
sub = fig.add_subplot(1,1,1)
H, xedges, yedges = np.histogram2d(allx, ally, bins=[100,100],range=[[-3.5,2.5], [-5,2]])

H = H.T+1e-10
pmf = -np.log(H)*0.6757
pmf = pmf - np.min(pmf)
cmap.set_over("white")
CS = plt.contourf(xedges[1:],yedges[1:],pmf,levels = np.linspace(0,6,13),cmap=cmap,extend="max")
cbar = plt.colorbar(CS)
npz_file='/home/dongdong/SCR/jxy/casp13_analysis-master/tICA_npz/%s.tICA.npz' %TARGETs[indexs]
tICA_crd = np.load(npz_file, allow_pickle=True, encoding='latin1')
sub.scatter(tICA_crd['ini_crd'][0][0],tICA_crd['ini_crd'][0][1],c='black',marker='x')
sub.scatter(tICA_crd['ref_crd'][0][0],tICA_crd['ref_crd'][0][1],color='blue',marker='x')

sub.tick_params(direction="in", length=1)
plt.xlim(-3.2,2.5)
plt.ylim(-4.8,1.8)

#plt.grid(linestyle='--')
plt.grid(alpha=0.3)
#sub.set_xticks(np.linspace(0,150,4))
#sub.set_yticks(np.linspace(1,3,3))
sub.set_xlabel('tIC1',fontproperties=font_prop)
sub.set_ylabel('tIC2',fontproperties=font_prop)
#sub.add_patch(patches.Rectangle((0, 0),40,0.9, linewidth=1,edgecolor='r',facecolor='none'))
#leg=sub.legend(loc=4, labelspacing=0.1, prop=leg_prop, scatterpoints=1, markerscale=1, numpoints=1,handlelength=0)
#leg.get_frame().set_linewidth(0.0)
#leg.get_frame().set_alpha(0.1)
for label in (sub.get_xticklabels() + sub.get_yticklabels()):
    label.set_fontproperties(font_prop)
    label.set_fontsize(20)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(leg_prop)
plt.tight_layout(pad=3, w_pad=0.8, h_pad=0.4)
plt.savefig('rid_pmf.png',dpi=600,bbox_inches='tight')
plt.show()