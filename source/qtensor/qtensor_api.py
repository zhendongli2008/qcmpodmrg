import numpy
import qtensor
from mpodmrg.source import mpo_dmrg_qphys

# Cast fmps0 to fmps1[Qt form]
def fmpsQt(fmps0,fmps1,status,isym=2):
   print '\n[qtensor_api.fmpsQt] status=',status
   nsite = fmps0['nsite'].value
   fmps1['nsite'] = nsite
   qphys = mpo_dmrg_qphys.initSpatialOrb(nsite,isym)
   # False = In, True = Out
   if status == 'L':
      sta = [False,False,True]
   elif status == 'R': 
      sta = [True,False,False]
   # Check symmetry
   lenSym = len(fmps0['qnum0'].value[0])
   if lenSym != isym:
      print ' error: isym is not consistent!'
      print ' qnum0:',fmps0['qnum0'].value[0]
      print ' lenSym/isym =',lenSym,isym
      exit()
   # Cast into Qt tensor
   for isite in range(nsite):
      ql = fmps0['qnum'+str(isite)].value
      qn = qphys[isite]
      qr = fmps0['qnum'+str(isite+1)].value
      site = fmps0['site'+str(isite)].value
      tmps = qtensor.qtensor(sta)
      tmps.fromDenseTensor(site,[ql,qn,qr])    
      tmps.dump(fmps1,'site'+str(isite))     
   # Save qnums   
   for isite in range(nsite+1):
      ql = fmps0['qnum'+str(isite)].value
      fmps1['qnum'+str(isite)] = ql
   return 0

# Cast mps0 to fmps1[Qt form]
def mpsQt(mps,qnum,status):
   print '\n[qtensor_api.mpsQt] status=',status
   debug=False
   nsite = len(mps)
   qphys = mpo_dmrg_qphys.initSpatialOrb(nsite,2)
   # False = In, True = Out
   if status == 'L':
      sta = [False,False,True]
   elif status == 'R': 
      sta = [True,False,False]
   qtlst = []
   for isite in range(nsite):
      ql = qnum[isite]
      qn = qphys[isite]
      qr = qnum[isite+1]
      site = mps[isite]
      tmps = qtensor.qtensor(sta)
      tmps.fromDenseTensor(site,[ql,qn,qr])    
      qtlst.append(tmps)
      # Test
      if debug:
         print 'isite=',isite
         tsite = tmps.toDenseTensor()
         diffDense = numpy.linalg.norm(tsite-site)
         print ' diffDense=',diffDense
         tmps.prt()
         assert diffDense<1.e-12
   return qtlst

# MPS = List of Qt
def mps_bdim(mps):
   nsite = len(mps)
   bdim = []
   for i in range(nsite-1):
      bdim.append(mps[i].shape[2])
   return bdim

# ||Psi||
def mps_norm(mps):
   norm=math.sqrt(mps_dot(mps,mps))
   return norm

# <Psi1|Psi2>
def mps_dot(bmps,kmps,status='L'):
   nsite1 = len(bmps)
   nsite2 = len(kmps)
   assert nsite1==nsite2
   nsite = nsite1
   if status == 'L':
      tmp = qtensor.tensordot(bmps[0],kmps[0],axes=([1],[1]))
      tmp.status[0] = ~tmp.status[0]
      tmp.status[2] = ~tmp.status[2]
      tmp = tmp.merge([[0,1],[2,3]])
      for i in range(1,nsite):
         # t[l,r]*k[r,n,r'] = t[l,n,r']
	 # b[l,n,l']*t[l,n,r'] = t[l',r']
	 tmp = qtensor.tensordot(tmp,kmps[i],axes=([1],[0]))
	 tmp = qtensor.tensordot(bmps[i],tmp,axes=([0,1],[0,1]))
   elif status == 'R':
      tmp = qtensor.tensordot(bmps[nsite-1],kmps[nsite-1],axes=([1],[1]))
      tmp.status[1] = ~tmp.status[1]
      tmp.status[3] = ~tmp.status[3]
      tmp = tmp.merge([[0,1],[2,3]])
      for i in range(nsite-2,-1,-1):
	 # b[l,n,l']*t[l',r'] = t[l,n,r']
	 # t[l,n,r']*k[r,n,r'] = t[l,r]
	 tmp = qtensor.tensordot(bmps[i],tmp,axes=([2],[0]))
	 tmp = qtensor.tensordot(tmp,kmps[i],axes=([1,2],[1,2]))
   assert len(tmp.value)==1
   ovlp = tmp.value[0]
   return ovlp
