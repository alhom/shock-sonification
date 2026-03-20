import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import scipy as sp
import sys

def se(B):
   plt.savefig("NSP2_sampler.png",dpi=300)#,bbox_inches='tight')
   # plt.savefig("NSP2_sampler.eps",dpi=300)#,bbox_inches='tight')
   # plt.savefig("NSP2_sampler.pdf",dpi=300)#,bbox_inches='tight')
   np.savetxt("NSP2_samples.txt", B)
   sys.exit()

datafile = "../assets/Magnetic_Field_for_RBF.mat"
Bdata = h5py.File(datafile)["fTS00"]["B"]
print(Bdata['x'].shape)
Bx = np.array(Bdata['x']).transpose((2,1,0))/1e9
By = np.array(Bdata['y']).transpose((2,1,0))/1e9
Bz = np.array(Bdata['z']).transpose((2,1,0))/1e9

Bvec = np.stack((Bx,By,Bz),axis=3)*1e9

# numpy.pad here for periodic boxes?
# https://numpy.org/doc/stable/reference/generated/numpy.pad.html

# assume ~1/cc density and 200km di
di = 200e3 #km

dx = 0.25*di #50e3 km

x = np.arange(Bx.shape[0])*dx
y = np.arange(Bx.shape[1])*dx
z = np.arange(Bx.shape[2])*dx
intpB = sp.interpolate.RegularGridInterpolator((x,y,z), Bvec, method="linear")

v = 100e3 #m/s "shock crossing velocity"
#################
# Reading in the trajectory

# end=18500
# stride=1
# start=16500

# start = 0
# end= 20000


conf="NSP2"

# This can be substituted for Emanuele's virtual spacecraft code!
from basic_constellation import run,fly

run(outerscale=1000e3)
start = np.array([1*di, 30*di, 0])
end = np.array([508*di, 30*di, 0])
dist = np.linalg.norm(end-start)

path = fly(start=start,end=end, steps=20000, time_over_segment=dist/v)

Rs = [] # sc list of SC position tracks



for sc in [0,1,2,3,4,5,6]:
   Rs.append(path[sc::7,2:5])


################### trajectory

ds = (Rs[0][1,0]-Rs[0][0,0])

dt = ds/v
print(dt,"dt, dx/ds", dx/ds)

fig = plt.Figure(figsize=(12,8))

axs=[0,0,0,0,0,0,0,0,0]
nrows = 7
axs[0] = plt.subplot2grid((nrows, 2), (0, 0), colspan=2,rowspan=2)
axs[1] = plt.subplot2grid((nrows, 2), (3, 0))
axs[2] = plt.subplot2grid((nrows, 2), (3, 1))
axs[3] = plt.subplot2grid((nrows, 2), (4, 0))
axs[4] = plt.subplot2grid((nrows, 2), (4, 1))
axs[5] = plt.subplot2grid((nrows, 2), (5, 0))
axs[6] = plt.subplot2grid((nrows, 2), (5, 1))
axs[7] = plt.subplot2grid((nrows, 2), (6, 0))
axs[8] = plt.subplot2grid((nrows, 2), (6, 1))

msize = 0.2
mcolor = 'k'
mcolors = ['k','r','g','b','m','y','c']
bcmap0 = matplotlib.cm.turbo.copy()

bcmap = matplotlib.cm.turbo
bcmap.set_under('white',1.)
xl ='$X/m$'
yl ='$Y/m$'

xx, yy = np.meshgrid(x,y)
zz = np.ones_like(xx)*np.mean(z)

Bintps0 = intpB(Rs[0])
bmags0 = np.linalg.norm(Bintps0, axis=-1)
Bbvec = np.squeeze(intpB((xx,yy,zz)))
# cmappable = axs[0].pcolor(xx,yy,np.linalg.norm(Bbvec,axis=-1), shading='nearest', cmap=bcmap)
cmappable0 = axs[0].imshow(np.linalg.norm(Bbvec,axis=-1), cmap=bcmap0, extent=(np.min(xx),np.max(xx), np.min(yy),np.max(yy)), origin="lower")
# plt.gcf().colorbar(cmappable0, ax=axs[0], label="$B/B_0$", orientation="horizontal")

# axs[0].axis("equal")
axs[0].set_aspect(1)
axs[0].set_xlabel(xl)
axs[0].set_ylabel(yl)
axs[0].streamplot(xx,yy,Bbvec[:,:,0],Bbvec[:,:,1], color='gray',linewidth=0.1, arrowsize=0.1, density=4)
# axs[0].set_xlim((300,400))
clim = cmappable0.get_clim()
# print(clim)
# axs[0].scatter(Rs[0][:,0],Rs[0][:,1], c=mcolor, cmap="turbo",clim=clim, s=msize/10)

# BvecInset = np.squeeze(intpB((X,Y,Z)))
mcolors = ['k','r','g','b','m','y','c']
for sc in [0,1,2,3,4,5,6]:
   axs[0].scatter(Rs[sc][:,0],Rs[sc][:,1], c=mcolors[sc],clim=clim,s=msize)

t_tetras=len(Rs[0])//2
for scs in [[0,1],[0,2],[0,3],[1,2],[2,3],[1,3],[0,4],[0,5],[0,6],[4,5],[5,6],[4,6]]:

   xs = np.array([Rs[scs[0]][t_tetras,0], Rs[scs[1]][t_tetras,0]])
   ys = np.array([Rs[scs[0]][t_tetras,1], Rs[scs[1]][t_tetras,1]])
   axs[0].plot(xs,ys, color='k',linewidth=0.3)

Bscs = [intpB(Rs[sc]) for sc in [0,1,2,3,4,5,6]]
# Smooth the data to get rid of linear intp artefacts at the dx level
kernel_length = 2*int(dx/ds)
kernel_iters = 2
for i in range(kernel_iters):
   Bscs = [np.array([np.convolve(np.squeeze(B[:,i]), np.ones((kernel_length))/kernel_length,mode="same") for i in [0,1,2]]).T for B in Bscs]

Fs = 1/dt
freqlim = 2/(dx/v)
print("freq. limit", freqlim, "freq Fs", Fs)
NFFT = 200

# print(Bscs[0].shape)

Bscsmag = [np.linalg.norm(B, axis=-1) for B in Bscs]

for i,ax in enumerate(axs):
   if i==0:
      print("ax0")
      continue
   elif i==1:
      print("ax1")
      for sc in [0,1,2,3,4,5,6]:
         ax.plot(Rs[sc][:,0],Bscsmag[sc])
         ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            direction="in",
            top=True,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
   elif i>1:
      sc = i - 2
      ax.specgram(Bscsmag[sc], NFFT=NFFT, Fs=Fs)
      ax.set_ylim([0,freqlim])

      ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            direction="in",
            top=True,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

fade = 50

for sc in [0,1,2,3,4,5,6]:
   B = Bscsmag[sc]
   minB = np.min(B)
   maxB = np.max(B)
   B -= minB
   B /= (maxB-minB)
   B *= 2
   B -= 1
   B *= 2147483647
   # print(B[fade:-fade])

   B[:fade] = B[:fade]*np.linspace(0,1,fade)
   B[-fade:] = B[-fade:]*np.linspace(1,0,fade)
   sp.io.wavfile.write("shocking_"+str(sc)+".wav",int(Fs*800),B.astype(np.int32))

Bmagdiffs = []

for scs in [[0,1],[0,2],[0,3],[1,2],[2,3],[1,3],[0,4],[0,5],[0,6],[4,5],[5,6],[4,6]]:

   if (scs[0] != 0):
      # If first SC is not the apex SC, skip - only do diffs against that one now
      continue
   xs = np.array([Rs[scs[0]][t_tetras,0], Rs[scs[1]][t_tetras,0]])
   ys = np.array([Rs[scs[0]][t_tetras,1], Rs[scs[1]][t_tetras,1]])

   Bmagdiff = Bscsmag[scs[1]] - Bscsmag[scs[0]]
   Bmagdiffs.append(Bmagdiff)

for sc,B in enumerate(Bmagdiffs):
   minB = np.min(B)
   maxB = np.max(B)
   B -= minB
   B /= (maxB-minB)
   B *= 2
   B -= 1
   B *= 2147483647

   B[:fade] = B[:fade]*np.linspace(0,1,fade)
   B[-fade:] = B[-fade:]*np.linspace(1,0,fade)
   sp.io.wavfile.write("shocking_diff_0to"+str(sc+1)+".wav",int(Fs*800),B.astype(np.int32))



se(B)


# plt.tight_layout()
# plt.subplots_adjust(hspace=0.0)

