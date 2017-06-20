import numpy
import h5py
import copy
from zmpo_dmrg.source import mpo_dmrg_io
from zmpo_dmrg.source import mpo_dmrg_opers
from zmpo_dmrg.source.misc import mpo_dmrg_model
from zmpo_dmrg.source.mpsmpo import mps_class
from zmpo_dmrg.source.mpsmpo import mpo_class
from zmpo_dmrg.source.mpsmpo.tools import mpslib
import tevol
import os
import shutil

fname='./cal1/lmps'
fmps = h5py.File(fname,'r')

mps,qnum = mpo_dmrg_io.loadMPS(fmps,icase=1)
nsite = len(mps)

# Hamiltonian
fhop = h5py.File('./cal1/hop')
nops = fhop['nops'].value
assert nops == 1
wfacs = [0]*nsite
for isite in range(nsite):
   gname = 'site'+str(isite)
   grp = fhop[gname]
   for iop in range(nops):
      wfacs[isite] = grp['op'+str(iop)].value+0.j
# H-MPO
hmpo = mpo_class.class_mpo(nsite,sites=wfacs)
# DM[i,i]
mps0 = mps_class.class_mps(nsite,sites=mps,iop=1)
e0 = mps0.dot(hmpo.dotMPS(mps0))
print 'e0=',e0

maxM = 100
ttotal = 500.
nstep = 5000
iorder = 4
prefix = './dataOrder'+str(iorder)

tau = ttotal/nstep
print 'nstep=',nstep
slst = [0] #,1,2,3] #range(nsite)

try:
   shutil.rmtree(prefix)
except:
   pass
os.mkdir(prefix)

# To generate approximate exp(-i(H-E0)*t) 
wfacs[0][0,-1] -= e0*numpy.identity(4)
hmpo1 = mpo_class.class_mpo(nsite,sites=wfacs)

tauA = (0.5+0.5j)*tau
tauB = (0.5-0.5j)*tau
empoA = tevol.polyH(hmpo1,xfac=-tauA*1.j)
empoB = tevol.polyH(hmpo1,xfac=-tauB*1.j)

empo = tevol.polyH(hmpo1,xfac=-tau*1.j)
lmpo = tevol.linearH(hmpo1,xfac=-tau*1.j)

def compress(mps_i1,maxM,iprt=0):
   normA = mps_i1.normalize()
   if iprt > 0:
      print '\n[compress]'
      print ' norm of exppsi before normalization =',normA
      print ' bdim0=',mps_i1.bdim()
   # Cast to canonical form
   mps_i1.leftCanon()
   if iprt > 0: print ' bdimL=',mps_i1.bdim()
   # Compress
   mps_i1.rightCanon(Dcut=maxM)
   if iprt > 0: print ' bdimR=',mps_i1.bdim()
   mps_i1.leftCanon(Dcut=maxM)
   if iprt > 0: print ' bdimL=',mps_i1.bdim()
   return normA

def rk4(mps_i,hmpo1,tau,maxM,debug=False):
   # k1
   k1 = hmpo1.dotMPS(mps_i)
   normk1 = compress(k1,maxM)
   k1.mul(tau*normk1)
   k1.sites[0] = -1.j*k1.sites[0]
   if debug: print 'k1norm=',k1.norm()
   # k2
   k1p = k1.copy()
   k1p.mul(0.5)
   k1p = k1p.add(mps_i)
   normk1p = compress(k1p,maxM)
   k2 = hmpo1.dotMPS(k1p)
   normk2 = normk1p*compress(k2,maxM)
   k2.mul(tau*normk2)
   k2.sites[0] = -1.j*k2.sites[0]
   if debug: print 'k2norm=',k2.norm()
   # k3
   k2p = k2.copy()
   k2p.mul(0.5)
   k2p = k2p.add(mps_i)
   normk2p = compress(k2p,maxM)
   k3 = hmpo1.dotMPS(k2p)
   normk3 = normk2p*compress(k3,maxM)
   k3.mul(tau*normk3) 
   k3.sites[0] = -1.j*k3.sites[0]
   if debug: print 'k3norm=',k3.norm()
   # k3
   k3p = k3.copy()
   k3p = k3p.add(mps_i)
   normk3p = compress(k3p,maxM)
   k4 = hmpo1.dotMPS(k3p)
   normk4 = normk3p*compress(k4,maxM)
   k4.mul(tau*normk4) 
   k4.sites[0] = -1.j*k4.sites[0]
   if debug: print 'k4norm=',k4.norm()
   #print 'norms[1,2,3,4]=',(normk1,normk2,normk3,normk4)
   mps_i1 = mps_i.copy()
   k1.mul(1.0/6.0)
   k2.mul(1.0/3.0)
   k3.mul(1.0/3.0)
   k4.mul(1.0/6.0)
   mps_i1 = mps_i1.add(k1)
   mps_i1 = mps_i1.add(k2)
   mps_i1 = mps_i1.add(k3)
   mps_i1 = mps_i1.add(k4)
   norm = compress(mps_i1,maxM)
   return mps_i1,norm 


for isite in slst:

   # create complex MPS
   mpsi = None
   mpsi = [mps[i]+0.j for i in range(nsite)]

   # MPSi = ai*|psi0>
   annA = mpo_dmrg_opers.genElemSpatialMat(2*isite,isite,0)
   tmp = numpy.tensordot(annA,mpsi[isite],axes=([1],[1]))
   mpsi[isite] = tmp.transpose(1,0,2)

   # sign change
   for jsite in range(isite):
      sgn = mpo_dmrg_model.genSaSbSpatialMat()
      tmp = numpy.tensordot(sgn,mpsi[jsite],axes=([1],[1]))
      mpsi[jsite] = tmp.transpose(1,0,2)

   # MPS class 
   mps_i0 = mps_class.class_mps(nsite,sites=mpsi,iop=1)
   mps_i = mps_class.class_mps(nsite,sites=mpsi,iop=1)
   
   # DM[i,i]
   norm0 = mps_i.norm()
   print 'isite=',isite,'DM[i,i]=',norm0

   mps_i.normalize()
   gf = [norm0**2]
   nm = [1.0]
   nt = nstep # 10
   for i in range(nt):

      print '\n'+'='*60
      print 'isite=',isite,' i=',i,' nstep=',nstep,' tau=',tau,' t=',(i+1)*tau

      # first order
      if iorder == 1:

         mps_i1 = empo.dotMPS(mps_i)
         norm = compress(mps_i1,maxM)
    
      elif iorder == 2:

         # second order
         mps_i1 = empoA.dotMPS(mps_i)
         normA = compress(mps_i1,maxM)

         mps_i1 = empoB.dotMPS(mps_i1)
         normB = compress(mps_i1,maxM)

	 norm = normA*normB
 
      elif iorder == 4:

	 mps_i1,norm = rk4(mps_i,hmpo1,tau,maxM)

      #Copy
      mps_i = copy.deepcopy(mps_i1)
      # Gii(t) = (<0|ai^+)[Real] exp(-i(H-E0)t) ai|0>
      prod = mps_i0.dot(mps_i)
      # Note: One should throw away the norm since psi_i is supposed to be 1.
      giit = prod*norm0 
      print '\ngiit=',giit,(norm0,norm)
      nm.append(norm)
      gf.append(giit)

   # Save the complex GF
   numpy.save(prefix+'/nm'+str(isite),numpy.array(nm))
   numpy.save(prefix+'/gf'+str(isite),numpy.array(gf))
   numpy.save(prefix+'/gf'+str(7-isite),numpy.array(gf))

fmps.close()
