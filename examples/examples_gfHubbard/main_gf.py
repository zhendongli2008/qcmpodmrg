import numpy
import h5py
import copy
from zmpo_dmrg.source import mpo_dmrg_io
from zmpo_dmrg.source import mpo_dmrg_opers
from zmpo_dmrg.source import mpo_dmrg_model
from zmpo_dmrg.source.mpsmpo import mps_class
from zmpo_dmrg.source.mpsmpo import mpo_class
from zmpo_dmrg.source.mpsmpo.tools import mpslib
import tevol
import os
import shutil

shutil.rmtree('./data')
os.mkdir('./data')

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
ttotal = 500.0 # 500
nstep = 5000   # 5000
tau = ttotal/nstep
print 'nstep=',nstep

# To generate approximate exp(-i(H-E0)*t) 
wfacs[0][0,-1] -= e0*numpy.identity(4)
hmpo1 = mpo_class.class_mpo(nsite,sites=wfacs)

tauA = (0.5+0.5j)*tau
tauB = (0.5-0.5j)*tau
empoA = tevol.polyH(hmpo1,xfac=-tauA*1.j)
empoB = tevol.polyH(hmpo1,xfac=-tauB*1.j)

empo = tevol.polyH(hmpo1,xfac=-tau*1.j)
lmpo = tevol.linearH(hmpo1,xfac=-tau*1.j)

def compress(mps_i1,maxM):
   normA = mps_i1.normalize()
   print '\n[compress]'
   print ' norm of exppsi before normalization =',normA
   print ' bdim0=',mps_i1.bdim()
   mps_i1.leftCanon(Dcut=maxM)
   print ' bdimL=',mps_i1.bdim()
   mps_i1.rightCanon(Dcut=maxM)
   print ' bdimR=',mps_i1.bdim()
   return normA

slst = [0] #range(nsite)
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

   iorder = 4
   mps_i.normalize()
   gf = [norm0**2]
   nt = nstep # 10
   for i in range(nt):

      print '\n'+'='*60
      print 'isite=',isite,' i=',i,' nstep=',nstep,' tau=',tau,' t=',(i+1)*tau

      # first order
      if iorder == 1:

         mps_i1 = empo.dotMPS(mps_i)
         norm = compress(mps_i1,maxM)
    
      else:

         # second order
         mps_i1 = empoA.dotMPS(mps_i)
         normA = compress(mps_i1,maxM)

         mps_i1 = empoB.dotMPS(mps_i1)
         normB = compress(mps_i1,maxM)

	 norm = normA*normB
 
      #Copy
      mps_i = copy.deepcopy(mps_i1)
      # Gii(t) = (<0|ai^+)[Real] exp(-i(H-E0)t) ai|0>
      prod = mps_i0.dot(mps_i)
      giit = prod*norm0*norm
      print '\ngiit=',giit,(norm0,norm)
      gf.append(giit)

   # Save the complex GF
   numpy.save('./data/gf'+str(isite),numpy.array(gf))

fmps.close()
