import specplt
import numpy as np
import matplotlib.pyplot as plt

nsite = 8
ttotal = 50.
nt = 500
tau = ttotal/nt

#tarray = np.arange(nt+1)*tau
#gfdiag = np.zeros(nt+1,dtype=np.complex128)
#slst = [0]
#for isite in slst:
#   #gfdiag += np.load('./data/gf'+str(isite)+'.npy')
#   gfdiag += np.load('./gf'+str(isite)+'.npy')
#
#gfr = map(lambda x:x.real,gfdiag)
#gfi = map(lambda x:x.imag,gfdiag)
#plt.plot(tarray,gfr,'ro-')
#plt.plot(tarray,gfi,'bo-')

prefix = "./t500/"
tarray1 = np.loadtxt(prefix+"refR.txt").T[0] 
gfr1 = np.loadtxt(prefix+"refR.txt").T[1]
gfi1 = np.loadtxt(prefix+"refI.txt").T[1]
plt.plot(tarray1,gfr1,'g-')
plt.plot(tarray1,gfi1,'k-')

plt.show()

specplt.run(tarray1,gfr1,gfi1,eta=0.05, rem_add='rem')

