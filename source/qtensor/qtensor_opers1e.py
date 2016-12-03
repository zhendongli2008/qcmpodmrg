#!/usr/bin/env python
#
# MPO operators in Qt form for H1e.
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def genWfacQnums(nsite,isite,isz,status='L'):
# def genHRfacSpatialQt(pindx,nsite,isite,int1e,int2e,qpts,pdic,maxslc=1):
#
import numpy
import itertools
import qtensor
import qtensor_util
import qtensor_opers
from mpodmrg.source import mpo_dmrg_opers1e
from mpodmrg.source.tools import parallel_util

# Spin orbital case: nsite = 2*K
def genWfacQnums(nsite,isite,isz,status='L'):
   spins = [0.5,-0.5]
   sz = spins[isz]
   if isite == 0:
      ql = [[0.,0.]]
      qr = [[0.,0.],[-1.,-sz]]
   elif isite == nsite-1:
      ql = [[0.,0.],[-1.,-sz]]
      qr = [[-1.,-sz]]
   else:
      ql = [[0.,0.],[-1.,-sz]]
      qr = [[0.,0.],[-1.,-sz]]
   # Shifted by the global value (-1,-sz).
   if status == 'R':
      ql = [[-1.-q[0],-sz-q[1]] for q in ql]
      qr = [[-1.-q[0],-sz-q[1]] for q in qr]
   ql = numpy.array(ql)
   qr = numpy.array(qr)
   return ql,qr

# Hx*R(theta)
def genHRfacSpatialQt(pindx,nsite,isite,int1e,qpts,maxslc=1,model_u=0.):
   p,ip = pindx
   iop  = 1
   isym = 2
   status = 'L'
   isz  = p%2
   site = mpo_dmrg_opers1e.genHRfacSpatial(pindx,nsite,isite,int1e,qpts,model_u)
   # Site physical indices
   qu,qd = qtensor_opers.genQphys(isym,isite)
   # Orbital dependent part
   ql1e,qr1e = qtensor_opers.genElemQnums(p,2*isite  ,iop,status)
   ql2e,qr2e = qtensor_opers.genElemQnums(p,2*isite+1,iop,status)
   ql1 = ql1e
   qr1 = qr2e
   ql1w,qr1w = genWfacQnums(nsite,2*isite  ,isz,status)
   ql2w,qr2w = genWfacQnums(nsite,2*isite+1,isz,status)
   ql2 = ql1w
   qr2 = qr2w
   ql = [q1+q2 for q1,q2 in itertools.product(ql1,ql2)]
   qr = [q1+q2 for q1,q2 in itertools.product(qr1,qr2)]
   if abs(model_u)>1.e-12: 
      if isite == 0:
         qr += [[0.,0.]]
      elif isite == nsite//2-1:
         ql += [[0.,0.]]
      else:
         ql += [[0.,0.]]
         qr += [[0.,0.]]
   # Reduce symmetry: local spin rotation do not change particle number
   if ip is not None:
      ql = qtensor_util.reduceQnumsToN(ql)
      qr = qtensor_util.reduceQnumsToN(qr)
      qu = qtensor_util.reduceQnumsToN(qu)
      qd = qtensor_util.reduceQnumsToN(qd)
   #
   # We sort the internal quantum number first
   #
   nql = len(ql)
   nqr = len(qr)
   idxl = qtensor_opers.sortQnums(ql)
   idxr = qtensor_opers.sortQnums(qr)
   ql = numpy.array(ql)[idxl]
   qr = numpy.array(qr)[idxr]
   site = site[numpy.ix_(idxl,idxr)].copy()
   # Slice operators if necessary
   qts = qtensor.Qt(2)
   qts.maxslc[0] = min(maxslc,nql) 
   qts.maxslc[1] = min(maxslc,nqr)
   qts.genDic()
   for islc in range(qts.maxslc[0]): 
      slc0 = parallel_util.partitionSites(nql,qts.maxslc[0],islc)
      for jslc in range(qts.maxslc[1]):
         slc1 = parallel_util.partitionSites(nqr,qts.maxslc[1],jslc)
	 ijdx = qts.ravel([islc,jslc])
         qts.dic[ijdx] = qtensor.qtensor([False,True,False,True])
         # Note that the qsyms for the last two dimensions are not collected
         qts.dic[ijdx].fromDenseTensor(site[numpy.ix_(slc0,slc1)],[ql[slc0],qr[slc1],qu,qd],ifcollect=[1,1,0,0])
	 qts.size[ijdx] = qts.dic[ijdx].size_allowed
   return qts
