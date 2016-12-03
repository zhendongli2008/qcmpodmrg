#!/usr/bin/env python
#
# MPO operators in Qt form.
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def sortQnums(qnums):
# def case(porb,isite,ncsite=1):
# def genQphys(isym,isite=None):
# def genElemQnums(p,isite,iop,status='L'):
# def genElemProductQnums(oplst,isite):
# def genWfacSpinQnumsRight(nsite,isite,sz):
# def genWfacQnums(nsite,isite,isz,status='L'):
#
# def linkTwoOpers(qt1,qt2):
# def genElemSpinQt(p,isite,iop=1):
# def genElemSpatialQt(p,isite,iop):
# def genElemSpatialQt0(p,isite,iop):
# def genElemProductSpatialQt(oplst,isite):
#
# def genWfacSpinQt(nsite,isite,hq,vqrs,isz):
# def genWfacSpatialQt(nsite,isite,hq,vqrs,isz):
# def genHfacSpinQt(p,nsite,isite,hq,vqrs):
# def genHfacSpatialQt(p,nsite,isite,hq,vqrs):
#
# def genHRfacSpatialQt(pindx,nsite,isite,int1e,int2e,qpts,pdic,maxslc=1):
# def genHenRfacSpatialQt(dmrg,pindx,isite,iop):
# def genExpISyPhiQt(phi):
# def genElemProductRSpatialQt(oplst,isite,phi):
#
# Test:
#
# def genHfacSpinQt0(p,nsite,isite,hq,vqrs):
# def genHfacSpatialQt0(p,nsite,isite,hq,vqrs):
# def testElem():
# def testElemSingle():
# def testWfacSpin():
# def testWfacSpatial():
# def testHfacSpin():
# def testHfacSpatial():
# def testHRfacSpatial():
# def testQnumHRfacSpatialQt():
# 
import time
import numpy
import itertools
import qtensor
import qtensor_util
from qcmpodmrg.source import mpo_dmrg_opers
from qcmpodmrg.source import mpo_dmrg_ptopers
from qcmpodmrg.source import mpo_dmrg_qphys
from qcmpodmrg.source.tools import parallel_util

def sortQnums(qnums):
   idx = sorted(range(len(qnums)), key=lambda k: str(qnums[k]))
   return numpy.array(idx)

#
# To reduce the cost of processing symmetry information,
# we borrow the idea of operator types in the complementary
# based approach. For a given dot configuration, in the spin-orbital 
# based SUM of MPO representation, we classify the set of MPO as 
# a sum of six/eight classes of operators for nsite=1,2, respectively, 
# such that within each class the symmetry information is of the same type.
#
#    H = \sum_p Hp 
#      = H[<pA] + H[<pB] + H[pA] + H[pB] + H[>pA] + H[>pB]
#      = H[<pA] + H[<pB] + H[pA] + H[pB] + H[p'A] + H[p'B] H[>p'A] + H[>p'B]
#
def case(porb,isite,ncsite=1):
   if porb//2 < isite: 
      icase = 0
      if porb%2 == 1: icase = 1
   elif porb//2 == isite: 
      icase = 2
      if porb%2 == 1: icase = 3
   elif porb//2 > isite:
      if ncsite == 1:
         icase = 4
         if porb%2 == 1: icase = 5
      elif ncsite == 2:
         if porb//2 == isite+1:
            icase = 4
            if porb%2 == 1: icase = 5
         elif porb//2 > isite+1:
            icase = 6
            if porb%2 == 1: icase = 7
   return icase

#
# Generation of quantum numbers
#
def genQphys(isym,isite=None):
   if isym == 1:
      # Site physical indices
      if isite%2 == 0:
         qu = [[0.,0.],[1.,0.5]]
         qd = [[0.,0.],[1.,0.5]]
      else:
         qu = [[0.,0.],[1.,-0.5]]
         qd = [[0.,0.],[1.,-0.5]]
   elif isym == 2:
      qu = [[0.,0.],[1.,-0.5],[1.,0.5],[2.,0.]]
      qd = [[0.,0.],[1.,-0.5],[1.,0.5],[2.,0.]]
   qu = numpy.array(qu)
   qd = numpy.array(qd)
   return qu,qd

# p and iop defines the spin and cre/ann types
# for spin orbital site
def genElemQnums(p,isite,iop,status='L'):
   # Spin
   if p%2 == 0:
      spin = 0.5
   else:
      spin = -0.5
   # Cre/Ann
   if iop == 1:
      sgn = 1.0
   else:
      sgn = -1.0
   nchange = sgn 
   schange = sgn*spin
   # Set
   if isite < p:
      ql = [[0.,0.]]
      qr = [[0.,0.]]
   elif isite == p:
      ql = [[0.,0.]]
      qr = [[nchange,schange]]
   elif isite > p:
      ql = [[nchange,schange]]
      qr = [[nchange,schange]]
   # Shifted by the global value (sgn,sgn*spin).
   if status == 'R':
      ql = [[nchange-q[0],schange-q[1]] for q in ql]
      qr = [[nchange-q[0],schange-q[1]] for q in qr]
   ql = numpy.array(ql)
   qr = numpy.array(qr)
   return ql,qr

# Products of Qnums
def genElemProductQnums(oplst,isite):
   p0,iop0 = oplst[0]
   ql1,qr1 = genElemQnums(p0,2*isite  ,iop0,status='L')
   ql2,qr2 = genElemQnums(p0,2*isite+1,iop0,status='L')
   ql = ql1
   qr = qr2
   for p1,iop1 in oplst[1:]:
      ql1,qr1 = genElemQnums(p1,2*isite  ,iop1,status='L')
      ql2,qr2 = genElemQnums(p1,2*isite+1,iop1,status='L')
      ql = [q1+q2 for q1,q2 in itertools.product(ql,ql1)]
      qr = [q1+q2 for q1,q2 in itertools.product(qr,qr2)]
   ql = numpy.array(ql)
   qr = numpy.array(qr)
   return ql,qr

# quantum numbers of right bond of site k for the left direction. 
def genWfacSpinQnumsRight(nsite,isite,sz):
   spins = [0.5,-0.5]
   q0 = [[0.,0.]]
   q1 = [[-2.,-sz-spins[i%2]] for i in range(isite+1,nsite)]
   q2 = [[ 0.,-sz+spins[i%2]] for i in range(isite+1,nsite)] 
   q3 = [[-1.,   -spins[i%2]] for i in range(isite+1)]
   q4 = [[ 1.,    spins[i%2]] for i in range(isite+1)]
   q5 = [[-1.,-sz]]
   qnums = q0+q1+q2+q3+q4+q5
   assert len(qnums) == 2*(nsite+1)
   qnums = numpy.array(qnums)
   return qnums 

# Spin orbital case: nsite = 2*K
def genWfacQnums(nsite,isite,isz,status='L'):
   spins = [0.5,-0.5]
   sz = spins[isz]
   if isite == 0:
      ql = [[0.,0.]]
      qr = genWfacSpinQnumsRight(nsite,isite,sz)
   elif isite == nsite-1:
      ql = genWfacSpinQnumsRight(nsite,isite-1,sz) 
      qr = [[-1.,-sz]]
   else:
      ql = genWfacSpinQnumsRight(nsite,isite-1,sz) 
      qr = genWfacSpinQnumsRight(nsite,isite,sz)
   # Shifted by the global value (-1,-sz).
   if status == 'R':
      ql = [[-1.-q[0],-sz-q[1]] for q in ql]
      qr = [[-1.-q[0],-sz-q[1]] for q in qr]
   ql = numpy.array(ql)
   qr = numpy.array(qr)
   return ql,qr

#
# Combine site tensor to form Qtensor
#
def genElemSpinQt(p,isite,iop=1):
   isym = 1
   # (1,1,2,2)
   site = mpo_dmrg_opers.genElemSpin(p,isite,iop)
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = genQphys(isym,isite)
   # Orbital dependent part
   ql,qr = genElemQnums(p,isite,iop,status='L')
   # Combine
   qt.fromDenseTensor(site,[ql,qr,qu,qd])
   return qt

def genElemSpatialQt(p,isite,iop):
   isym = 2
   # (1,1,4,4)
   site = mpo_dmrg_opers.genElemSpatial(p,isite,iop)
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = genQphys(isym,isite)
   # Orbital dependent part
   ql1,qr1 = genElemQnums(p,2*isite  ,iop,status='L')
   ql2,qr2 = genElemQnums(p,2*isite+1,iop,status='L')
   ql = ql1
   qr = qr2
   qt.fromDenseTensor(site,[ql,qr,qu,qd])
   return qt

def genElemProductSpatialQt(oplst,isite):
   isym = 2
   # (1,1,4,4)
   site = mpo_dmrg_opers.genElemProductSpatial(oplst,isite)
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = genQphys(isym,isite)
   # Orbital dependent part
   ql,qr = genElemProductQnums(oplst,isite)
   # Final conversion 
   qt.fromDenseTensor(site,[ql,qr,qu,qd])
   return qt

def linkTwoOpers(qt1,qt2):
   # (abij),(bckl)-> (aij,ckl)
   qt12 = qtensor.tensordot(qt1,qt2,axes=([1],[0]))
   qt12 = qt12.transpose(0,3,1,4,2,5)
   qt12 = qt12.merge([[0],[1],[2,3],[4,5]])
   return qt12

# A slow version based on contraction
def genElemSpatialQt0(p,isite,iop):
   qt1 = genElemSpinQt(p,2*isite  ,iop)
   qt2 = genElemSpinQt(p,2*isite+1,iop)
   qt12 = linkTwoOpers(qt1,qt2)
   return qt12 

# Spin orbital case: nsite=2*K
def genWfacSpinQt(nsite,isite,hq,vqrs,isz):
   isym = 1
   status = 'L'
   site = mpo_dmrg_opers.genWfacSpin(nsite,isite,hq,vqrs)
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = genQphys(isym,isite)
   # Quantum numbers
   ql,qr = genWfacQnums(nsite,isite,isz,status)
   # Combine
   qt.fromDenseTensor(site,[ql,qr,qu,qd])
   return qt

# Spatial orbital case: nsite=2*K
def genWfacSpatialQt(nsite,isite,hq,vqrs,isz):
   isym = 2
   status = 'L'
   site = mpo_dmrg_opers.genWfacSpatial(nsite,isite,hq,vqrs)
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = genQphys(isym,isite)
   # Quantum numbers
   ql1,qr1 = genWfacQnums(nsite,2*isite  ,isz,status)
   ql2,qr2 = genWfacQnums(nsite,2*isite+1,isz,status)
   ql = ql1
   qr = qr2
   qt.fromDenseTensor(site,[ql,qr,qu,qd])
   return qt

# ap^+Vp: we can assume isz is determined by the spin-orbital index p,
#	  which is the case for alpha-beta orderings.
def genHfacSpinQt0(p,nsite,isite,hq,vqrs):
   isz = p%2 
   elem = genElemSpinQt(p,isite,iop=1)
   wfac = genWfacSpinQt(nsite,isite,hq,vqrs,isz)
   # lrij,abjk-> lriabk ->larbik
   hfac = qtensor.tensordot(elem,wfac,axes=([3],[2]))
   hfac = hfac.transpose(0,3,1,4,2,5)
   hfac = hfac.merge([[0,1],[2,3],[4],[5]])
   return hfac

# Note that the ordering of Qsyms generated by
# these two subroutines are different due to the
# change of ordering for qsyms in converting to
# dictionary [partially fixed at the current stage].
def genHfacSpinQt(p,nsite,isite,hq,vqrs):
   iop  = 1
   isym = 1
   status = 'L'
   isz  = p%2
   site = mpo_dmrg_opers.genHfacSpin(p,nsite,isite,hq,vqrs)
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = genQphys(isym,isite)
   # Orbital dependent part
   ql1,qr1 = genElemQnums(p,isite,iop,status)
   ql2,qr2 = genWfacQnums(nsite,isite,isz,status)
   # Only if we convert qnums into numpy.array
   # List addition will lead to a longer list!
   ql = [q1+q2 for q1,q2 in itertools.product(ql1,ql2)]
   qr = [q1+q2 for q1,q2 in itertools.product(qr1,qr2)]
   # Combine
   qt.fromDenseTensor(site,[ql,qr,qu,qd])
   return qt

def genHfacSpatialQt(p,nsite,isite,hq,vqrs):
   iop  = 1
   isym = 2
   status = 'L'
   isz  = p%2
   site = mpo_dmrg_opers.genHfacSpatial(p,nsite,isite,hq,vqrs)
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = genQphys(isym,isite)
   # Orbital dependent part
   ql1e,qr1e = genElemQnums(p,2*isite  ,iop,status)
   ql2e,qr2e = genElemQnums(p,2*isite+1,iop,status)
   ql1 = ql1e
   qr1 = qr2e
   ql1w,qr1w = genWfacQnums(nsite,2*isite  ,isz,status)
   ql2w,qr2w = genWfacQnums(nsite,2*isite+1,isz,status)
   ql2 = ql1w
   qr2 = qr2w
   ql = [q1+q2 for q1,q2 in itertools.product(ql1,ql2)]
   qr = [q1+q2 for q1,q2 in itertools.product(qr1,qr2)]
   # Combine
   qt.fromDenseTensor(site,[ql,qr,qu,qd])
   return qt

# Simple implementation via linking
def genHfacSpatialQt0(p,nsite,isite,hq,vqrs):
   qt12e = genHfacSpinQt(p,nsite,2*isite  ,hq,vqrs)
   qt12o = genHfacSpinQt(p,nsite,2*isite+1,hq,vqrs)
   qt12b = linkTwoOpers(qt12e,qt12o)
   return qt12b

# 
# Hx*R(theta)
#
def genHRfacSpatialQt(pindx,nsite,isite,int1e,int2e,qpts,pdic,maxslc=1):
   p,ip = pindx
   iop  = 1
   isym = 2
   status = 'L'
   isz  = p%2
   site = mpo_dmrg_opers.genHRfacSpatial(pindx,nsite,isite,int1e,int2e,qpts,pdic)
   # Site physical indices
   qu,qd = genQphys(isym,isite)
   # Orbital dependent part
   ql1e,qr1e = genElemQnums(p,2*isite  ,iop,status)
   ql2e,qr2e = genElemQnums(p,2*isite+1,iop,status)
   ql1 = ql1e
   qr1 = qr2e
   ql1w,qr1w = genWfacQnums(nsite,2*isite  ,isz,status)
   ql2w,qr2w = genWfacQnums(nsite,2*isite+1,isz,status)
   ql2 = ql1w
   qr2 = qr2w
   ql = [q1+q2 for q1,q2 in itertools.product(ql1,ql2)]
   qr = [q1+q2 for q1,q2 in itertools.product(qr1,qr2)]
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
   idxl = sortQnums(ql)
   idxr = sortQnums(qr)
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

# 
# H[EN]*R(theta): the operator Hen has the feature that
#		  the quantum number is not changed.
#
def genHenRfacSpatialQt(dmrg,pindx,isite,iop):
   p,ip = pindx
   isym = 2
   site = mpo_dmrg_ptopers.genHenRfacSpatial(dmrg,pindx,isite,iop)
   if iop == 0:
      nbond = 3
   elif iop == 1 or iop == 2:
      nbond = 5
   # Site physical indices
   qu,qd = genQphys(isym,isite)
   ql = [[0.,0.]]*nbond
   qr = [[0.,0.]]*nbond
   if isite == 0: ql = [[0.,0.]]
   if isite == dmrg.nsite-1: qr = [[0.,0.]]
   ql = numpy.array(ql)
   qr = numpy.array(qr)
   # Reduce symmetry: local spin rotation do not change particle number
   if ip is not None:
      ql = qtensor_util.reduceQnumsToN(ql)
      qr = qtensor_util.reduceQnumsToN(qr)
      qu = qtensor_util.reduceQnumsToN(qu)
      qd = qtensor_util.reduceQnumsToN(qd)
   nql = len(ql)
   nqr = len(qr)
   # Slice operators if necessary
   qts = qtensor.Qt(2)
   qts.maxslc[0] = 1
   qts.maxslc[1] = 1
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

#
# R(theta)
#
def genExpISyPhiQt(phi):
   # (1,1,4,4)
   site = mpo_dmrg_opers.genExpISyPhi(phi)
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Spin rotation R(theta) is always particle number conserving.
   qu = [[0.],[1.],[1.],[2.]]
   qd = [[0.],[1.],[1.],[2.]]
   ql = [[0.]]
   qr = [[0.]]
   # Note that the qsyms for the last two dimensions are not collected
   qt.fromDenseTensor(site,[ql,qr,qu,qd],ifcollect=[0,0,0,0])
   return qt

#
# O*R(theta)
#
def genElemProductRSpatialQt(oplst,isite,phi):
   isym = 2
   # (1,1,4,4)
   cop = mpo_dmrg_opers.genElemProductSpatial(oplst,isite)
   rop = mpo_dmrg_opers.genExpISyPhiMat(phi)
   site = numpy.tensordot(cop,rop,axes=([3],[0]))
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = genQphys(isym,isite)
   # Orbital dependent part
   ql,qr = genElemProductQnums(oplst,isite)
   # Reduce symmetry: local spin rotation do not change particle number
   qu = qtensor_util.reduceQnumsToN(qu)
   qd = qtensor_util.reduceQnumsToN(qd)
   ql = qtensor_util.reduceQnumsToN(ql)
   qr = qtensor_util.reduceQnumsToN(qr)
   # Final conversion 
   qt.fromDenseTensor(site,[ql,qr,qu,qd],ifcollect=[0,0,0,0])
   return qt

##################################################################
# Test
##################################################################
def testElem():
   nsite = 4
   ta = 0.
   tb = 0.
   for iop in [1,0]:
      for p in range(nsite*2):
         for isite in range(nsite):
            t0 = time.time()
            qt12a = genElemSpatialQt(p,isite,iop)
            qt12b = genElemSpatialQt0(p,isite,iop)
	    qt12a.equalSym(qt12b)
	    site0 = qt12a.toDenseTensor()
	    t1 = time.time()
	    site1 = mpo_dmrg_opers.genElemSpatial(p,isite,iop)
	    t2 = time.time()
	    ta += t1-t0
	    tb += t2-t1
	    diff  = numpy.linalg.norm(site0-site1)
	    assert diff < 1.e-10
	    print 'iop,p,isite,diff=',iop,p,isite,diff
	    #assert diff < 1.e-10
   print 'ta =',ta
   print 'tb =',tb
   return 0 

def testElemSingle():
   p = 0
   isite = 0
   iop = 1 # creation
   for iop in [1,0]:
      qt1 = genElemSpinQt(p,isite,iop)
      qt1.prt()
      print qt1.qsyms
      print qt1.ndims
      print qt1.idlst
      print qt1.shape
      print qt1.toDenseTensor()
      ref = numpy.zeros((1,1,2,2))
      if iop == 1:
	 ref[0,0,1,0] = 1.0
      else:
	 ref[0,0,0,1] = 1.0
      assert numpy.linalg.norm(ref-qt1.toDenseTensor())<1.e-10
   for iop in [1,0]:
      qt1 = genElemSpatialQt(p,isite,iop)
      qt2 = genElemSpatialQt0(p,isite,iop)
      qt1.prt()
      qt2.prt()
      t1 = qt1.toDenseTensor()
      t2 = qt1.toDenseTensor()
      print 't1',t1.shape
      print 't1',t1
      print 't2',t2.shape
      print 't2',t2
      ref = numpy.zeros((1,1,4,4))
      if iop == 1:
	 ref[0,0,2,0] = 1.
         ref[0,0,3,1] = 1.
      else:
	 ref[0,0,0,2] = 1.
         ref[0,0,1,3] = 1.
      print 'ref',ref.shape
      print 'ref',ref
      assert numpy.linalg.norm(t1-ref)<1.e-10
      assert numpy.linalg.norm(t2-ref)<1.e-10
#
# Even sparsity at the matrix element level is used!
#
#  Basic information:
#   rank = 4  shape= [1 1 4 4]  nsyms= [1 1 4 4]
#   nblks_allowed = 2  nblks = 16
#   size_allowed  = 2  size = 16  savings= 0.125
#  Basic information:
#   rank = 4  shape= [1 1 4 4]  nsyms= [1 1 4 4]
#   nblks_allowed = 2  nblks = 16
#   size_allowed  = 2  size = 16  savings= 0.125
#
   return 0

def testWfacSpin():
#
# Time for ERI (requires 12G) for transpose,
# saving for size is about 1/10 for Wsite.
#
# ta = 7.88540029526 s
# tb = 9.84655189514 s
# 
# isz,isite= 1 197
# wop= (402, 402, 2, 2)
# Basic information:
#  rank = 4  shape= [402 402   2   2]  nsyms= [8 8 2 2]
#  nblks_allowed = 26  nblks = 256
#  size_allowed  = 80114  size = 646416  savings= 0.123935669909
# isz,isite,diff= 1 197 0.0
#  t1= 0.0034439563751 
#  t2= 0.0159950256348 (Efficient enough for generation of Wsite!)
# 
   nsite = 40
   nbas = nsite*2
   hmo = numpy.random.uniform(-1,1,(nbas,nbas))
   hmo[::2,1::2]=hmo[1::2,::2]=0.
   # [ij|kl]
   eri = numpy.random.uniform(-1,1,(nbas,nbas,nbas,nbas))
   eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
   # The spin symmetry is essential.
   #   <ij|kl>=[ik|jl]
   eri = eri.transpose(0,2,1,3)
   # Antisymmetry is not since only r<s is taken.
   #   # Antisymmetrize V[pqrs]=-1/2*<pq||rs> 
   #   eri = -0.5*(eri-eri.transpose(0,1,3,2))
   for isz in [0,1]:
      hq = hmo[isz]
      vqrs = eri[isz]
      ta = 0.
      tb = 0.
      for isite in range(nbas):
         print '\nisz,isite=',isz,isite
         t0 = time.time()
         site0 = mpo_dmrg_opers.genWfacSpin(nbas,isite,hq,vqrs)
         t1 = time.time()
         print 'wop=',site0.shape
	 # Wsite
	 qt12 = genWfacSpinQt(nbas,isite,hq,vqrs,isz)
	 qt12.prt()
	 site1 = qt12.toDenseTensor()
         t2 = time.time()
         ta += t1-t0
         tb += t2-t1
         assert site0.shape == site1.shape
         diff  = numpy.linalg.norm(site0-site1)
         print 'isz,isite,diff=',isz,isite,diff
         assert diff < 1.e-10
	 print ' t1=',t1-t0
	 print ' t2=',t2-t1
      print
      print 'ta =',ta
      print 'tb =',tb
   return 0 

def testWfacSpatial():
#
# Time for tc becomes smaller for nsite=100,
# due to the saving in contraction two sites.
#	
# ta = 14.3027746677
# tb = 16.9804520607 averaged(/100) = 0.16s.
# tc = 13.7098605633
# 
# isz,isite= 0 46
# wop= (402, 402, 4, 4)
# Basic information:
#  rank = 4  shape= [402 402   4   4]  nsyms= [8 8 4 4]
#  nblks_allowed = 84  nblks = 1024
#  size_allowed  = 212956  size = 2585664  savings= 0.0823602757357
# isz,isite,diff= 0 46 0.0
# Basic information:
#  rank = 4  shape= [402 402   4   4]  nsyms= [8 8 4 4]
#  nblks_allowed = 84  nblks = 1024
#  size_allowed  = 212956  size = 2585664  savings= 0.0823602757357
# isz,isite,diff= 0 46 0.0
#  equalSym= True
#  t1= 0.189389944077
#  t2= 0.212503194809
#  t3= 0.176101922989
# 
   nsite = 40
   nbas = nsite*2
   hmo = numpy.random.uniform(-1,1,(nbas,nbas))
   hmo[::2,1::2]=hmo[1::2,::2]=0.
   # [ij|kl]
   eri = numpy.random.uniform(-1,1,(nbas,nbas,nbas,nbas))
   eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
   # The spin symmetry is essential.
   #   <ij|kl>=[ik|jl]
   eri = eri.transpose(0,2,1,3)
   # Antisymmetry is not since only r<s is taken.
   #   # Antisymmetrize V[pqrs]=-1/2*<pq||rs> 
   #   eri = -0.5*(eri-eri.transpose(0,1,3,2))
   for isz in [0,1]:
      hq = hmo[isz]
      vqrs = eri[isz]
      ta = 0.
      tb = 0.
      tc = 0.
      for isite in range(nsite):
         print '\nisz,isite=',isz,isite
         t0 = time.time()
         site0 = mpo_dmrg_opers.genWfacSpatial(nbas,isite,hq,vqrs)
         t1 = time.time()
         print 'wop=',site0.shape
	 # Wsite
	 qt12 = genWfacSpatialQt(nbas,isite,hq,vqrs,isz)
	 qt12.prt()
	 site1 = qt12.toDenseTensor()
         t2 = time.time()
         ta += t1-t0
         tb += t2-t1
         assert site0.shape == site1.shape
         diff  = numpy.linalg.norm(site0-site1)
         print 'isz,isite,diff=',isz,isite,diff
         assert diff < 1.e-10
	 
	 t3 = time.time()
	 qt12a = genWfacSpinQt(nbas,2*isite  ,hq,vqrs,isz)
	 qt12b = genWfacSpinQt(nbas,2*isite+1,hq,vqrs,isz)
	 qt12l = linkTwoOpers(qt12a,qt12b)
	 qt12l.prt()
	 site2 = qt12l.toDenseTensor()
	 t4 = time.time()
	 tc += t4-t3
         assert site1.shape == site2.shape
         diff  = numpy.linalg.norm(site1-site2)
         print 'isz,isite,diff=',isz,isite,diff
         assert diff < 1.e-10

	 print ' equalSym=',qt12l.equalSym(qt12)
	 print ' t1=',t1-t0
	 print ' t2=',t2-t1
	 print ' t3=',t4-t3
      print
      print 'ta =',ta
      print 'tb =',tb
      print 'tc =',tc
   return 0 

def testHfacSpin():
#
# ta = 17.9673910141
# tb = 22.5408022404
# tc = 20.9652657509 Contraction based becomes slightly better for large nsite (n=100).
#
# p,isz,isite= 1 1 77
# hop= (402, 402, 2, 2)
# site0.shape= (402, 402, 2, 2)  sum= 251.870705764
# Basic information:
#  rank = 4  shape= [402 402   2   2]  nsyms= [8 8 2 2]
#  nblks_allowed = 26  nblks = 256
#  size_allowed  = 66554  size = 646416  savings= 0.102958466375
# site1a.shape= (402, 402, 2, 2)  sum= 251.870705764
# isz,isite,diffa= 1 77 0.0
# Basic information:
#  rank = 4  shape= [402 402   2   2]  nsyms= [8 8 2 2]
#  nblks_allowed = 26  nblks = 256
#  size_allowed  = 66554  size = 646416  savings= 0.102958466375
# site1b.shape= (402, 402, 2, 2)  sum= 251.870705764
# isz,isite,diffb= 1 77 0.0
# isz,isite,diffab= 1 77 0.0
#  t1[mat]= 0.0578351020813
#  t2[Qt ]= 0.0677578449249
#  t3[Qt0]= 0.0642321109772
#  equalSym= True
#
   numpy.random.seed(10) 
   nsite = 10
   nbas = nsite*2
   hmo = numpy.random.uniform(-1,1,(nbas,nbas))
   hmo[::2,1::2]=hmo[1::2,::2]=0.
   # [ij|kl]
   eri = numpy.random.uniform(-1,1,(nbas,nbas,nbas,nbas))
   eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
   # The spin symmetry is essential.
   #   <ij|kl>=[ik|jl]
   eri = eri.transpose(0,2,1,3)
   # Antisymmetry is not since only r<s is taken.
   #   # Antisymmetrize V[pqrs]=-1/2*<pq||rs> 
   #   eri = -0.5*(eri-eri.transpose(0,1,3,2))
   ta = 0.
   tb = 0.
   tc = 0.
   for p in range(nbas):
      isz = p%2
      hq = hmo[isz]
      vqrs = eri[isz]
      for isite in range(nbas):
         print '\np,isz,isite=',p,isz,isite
         t0 = time.time()
         site0 = mpo_dmrg_opers.genHfacSpin(p,nbas,isite,hq,vqrs)
         t1 = time.time()
         print 'hop=',site0.shape
         print 'site0.shape=',site0.shape,' sum=',numpy.sum(site0)
	 # Wsite
	 qt12a = genHfacSpinQt(p,nbas,isite,hq,vqrs)
	 qt12a.prt()
	 site1a = qt12a.toDenseTensor()
	 t2 = time.time()
	 print 'site1a.shape=',site1a.shape,' sum=',numpy.sum(site1a)
	 assert site0.shape == site1a.shape
	 diffa  = numpy.linalg.norm(site0-site1a)
         print 'isz,isite,diffa=',isz,isite,diffa

	 t3 = time.time()
	 qt12b = genHfacSpinQt0(p,nbas,isite,hq,vqrs)
	 qt12b.prt()
	 site1b = qt12b.toDenseTensor()
         t4 = time.time()
	 print 'site1b.shape=',site1b.shape,' sum=',numpy.sum(site1b)
	 assert site0.shape == site1b.shape
	 diffb  = numpy.linalg.norm(site0-site1b)
         print 'isz,isite,diffb=',isz,isite,diffb
	 
	 ta += t1-t0
         tb += t2-t1
         tc += t4-t3
	 diffab  = numpy.linalg.norm(site1a-site1b)
         print 'isz,isite,diffab=',isz,isite,diffab
	 print ' t1[mat]=',t1-t0
	 print ' t2[Qt ]=',t2-t1
	 print ' t3[Qt0]=',t4-t3
	 assert diffa < 1.e-10
         assert diffb < 1.e-10
         assert diffab < 1.e-10
	 print ' equalSym=',qt12a.equalSym(qt12b)
         assert qt12a.equalSym(qt12b)

      print
      print 'ta =',ta
      print 'tb =',tb
      print 'tc =',tc
   return 0 

def testHfacSpatial():
#
# ta = 29.8947856426
# tb = 34.9535117149
# tc = 27.4478466511 [again for large nsite=100, contraction based is better.]
#
# p,isz,isite= 1 1 50
# hop= (402, 402, 4, 4)
# site0.shape= (402, 402, 4, 4)  sum= 763.35358861
# Basic information:
#  rank = 4  shape= [402 402   4   4]  nsyms= [8 8 4 4]
#  nblks_allowed = 84  nblks = 1024
#  size_allowed  = 212812  size = 2585664  savings= 0.0823045840449
# site1a.shape= (402, 402, 4, 4)  sum= 763.35358861
# isz,isite,diffa= 1 50 0.0
# Basic information:
#  rank = 4  shape= [402 402   4   4]  nsyms= [8 8 4 4]
#  nblks_allowed = 84  nblks = 1024
#  size_allowed  = 212812  size = 2585664  savings= 0.0823045840449
# site1b.shape= (402, 402, 4, 4)  sum= 763.35358861
# isz,isite,diffb= 1 50 0.0
# isz,isite,diffab= 1 50 0.0
#  t1[mat]= 0.183460950851
#  t2[Qt ]= 0.20934510231
#  t3[Qt0]= 0.174911975861
#  equalSym= True
# 
   numpy.random.seed(10) 
   nsite = 10
   nbas = nsite*2
   hmo = numpy.random.uniform(-1,1,(nbas,nbas))
   hmo[::2,1::2]=hmo[1::2,::2]=0.
   # [ij|kl]
   eri = numpy.random.uniform(-1,1,(nbas,nbas,nbas,nbas))
   eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
   # The spin symmetry is essential.
   #   <ij|kl>=[ik|jl]
   eri = eri.transpose(0,2,1,3)
   # Antisymmetry is not since only r<s is taken.
   #   # Antisymmetrize V[pqrs]=-1/2*<pq||rs> 
   #   eri = -0.5*(eri-eri.transpose(0,1,3,2))
   ta = 0.
   tb = 0.
   tc = 0.
   for p in range(nbas):
      isz = p%2
      hq = hmo[isz]
      vqrs = eri[isz]
      for isite in range(nsite):
         print '\np,isz,isite=',p,isz,isite
         t0 = time.time()
         site0 = mpo_dmrg_opers.genHfacSpatial(p,nbas,isite,hq,vqrs)
         t1 = time.time()
         print 'hop=',site0.shape
         print 'site0.shape=',site0.shape,' sum=',numpy.sum(site0)
	 # Wsite
	 qt12a = genHfacSpatialQt(p,nbas,isite,hq,vqrs)
	 qt12a.prt()
	 site1a = qt12a.toDenseTensor()
	 t2 = time.time()
	 print 'site1a.shape=',site1a.shape,' sum=',numpy.sum(site1a)
	 assert site0.shape == site1a.shape
	 diffa  = numpy.linalg.norm(site0-site1a)
         print 'isz,isite,diffa=',isz,isite,diffa

	 t3 = time.time()
	 qt12e = genHfacSpinQt(p,nbas,2*isite  ,hq,vqrs)
	 qt12o = genHfacSpinQt(p,nbas,2*isite+1,hq,vqrs)
	 qt12b = linkTwoOpers(qt12e,qt12o)
	 qt12b.prt()
	 site1b = qt12b.toDenseTensor()
         t4 = time.time()
	 print 'site1b.shape=',site1b.shape,' sum=',numpy.sum(site1b)
	 assert site0.shape == site1b.shape
	 diffb  = numpy.linalg.norm(site0-site1b)
         print 'isz,isite,diffb=',isz,isite,diffb
	 
	 ta += t1-t0
         tb += t2-t1
         tc += t4-t3
	 diffab  = numpy.linalg.norm(site1a-site1b)
         print 'isz,isite,diffab=',isz,isite,diffab
	 print ' t1[mat]=',t1-t0
	 print ' t2[Qt ]=',t2-t1
	 print ' t3[Qt0]=',t4-t3
	 assert diffa < 1.e-10
         assert diffb < 1.e-10
         assert diffab < 1.e-10
	 print ' equalSym=',qt12a.equalSym(qt12b)
         assert qt12a.equalSym(qt12b)

      print
      print 'ta =',ta
      print 'tb =',tb
      print 'tc =',tc
   return 0 

def testHRfacSpatial():
#
# Savings are slightly reduced due to the nonzero terms 
# that mix blocks with different Sz values. [2 times]
#    
#   	   [ 1   0   0   0 ]
#   R(a) = [ 0   c  -s   0 ]
#   	   [ 0  -s   c   0 ]
#  	   [ 0   0   0   1 ]
#  
# p,isz,isite= 9 1 39
# hop= (162, 1, 4, 4)
# site0.shape= (162, 1, 4, 4)  sum= 19.4722422084
# Basic information:
#  rank = 4  shape= [162   1   4   4]  nsyms= [4 1 3 3]
#  nblks_allowed = 8  nblks = 36
#  size_allowed  = 572  size = 2592  savings= 0.220679012346
# site1a.shape= (162, 1, 4, 4)  sum= 19.4722422084
# isz,isite,diffa= 1 39 0.0
# 
   numpy.random.seed(10) 
   nsite = 10
   nbas = nsite*2
   hmo = numpy.random.uniform(-1,1,(nbas,nbas))
   hmo[::2,1::2]=hmo[1::2,::2]=0.
   # [ij|kl]
   eri = numpy.random.uniform(-1,1,(nbas,nbas,nbas,nbas))
   eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
   # The spin symmetry is essential.
   #   <ij|kl>=[ik|jl]
   eri = eri.transpose(0,2,1,3)
   # Antisymmetry is not since only r<s is taken.
   #   # Antisymmetrize V[pqrs]=-1/2*<pq||rs> 
   #   eri = -0.5*(eri-eri.transpose(0,1,3,2))
   ta = 0.
   tb = 0.
   for p in range(nbas):
      isz = p%2
      for isite in range(nsite):
         print '\np,isz,isite=',p,isz,isite
         t0 = time.time()
	 pindx = (p,0)
	 qpts = numpy.array([0.3])
         site0 = mpo_dmrg_opers.genHRfacSpatial(pindx,nbas,isite,hmo,eri,qpts)
         t1 = time.time()
         print 'hop=',site0.shape
         print 'site0.shape=',site0.shape,' sum=',numpy.sum(site0)
	 # Wsite
	 pdic = None #fake
	 qt12a = genHRfacSpatialQt(pindx,nbas,isite,hmo,eri,pdic,qpts)
	 qt12a.prt()
	 site1a = qt12a.toDenseTensor()
	 t2 = time.time()
	 print 'site1a.shape=',site1a.shape,' sum=',numpy.sum(site1a)
	 assert site0.shape == site1a.shape
	 diffa  = numpy.linalg.norm(site0-site1a)
         print 'isz,isite,diffa=',isz,isite,diffa
	 ta += t1-t0
         tb += t2-t1
      print
      print 'ta =',ta
      print 'tb =',tb
      qt12a.prt()
      print 'qsyms=',qt12a.qsyms
      print 'nsyms=',qt12a.nsyms
      print 'idlst=',qt12a.idlst
      print 'vals =',qt12a.value
   return 0 

def testQnumHRfacSpatialQt():
   numpy.random.seed(10) 
   nsite = 5
   nbas = nsite*2
   hmo = numpy.random.uniform(-1,1,(nbas,nbas))
   hmo[::2,1::2]=hmo[1::2,::2]=0.
   # [ij|kl]
   eri = numpy.random.uniform(-1,1,(nbas,nbas,nbas,nbas))
   eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
   # The spin symmetry is essential.
   #   <ij|kl>=[ik|jl]
   eri = eri.transpose(0,2,1,3)
   # Antisymmetry is not since only r<s is taken.
   #   # Antisymmetrize V[pqrs]=-1/2*<pq||rs> 
   #   eri = -0.5*(eri-eri.transpose(0,1,3,2))
   ta = 0.
   tb = 0.
   for isite in range(nsite):
      for p in range(nbas):
         isz = p%2
	 pindx = (p,0)
	 qpts = numpy.array([0.3])
	 pdic = None #fake
	 qt12a = genHRfacSpatialQt(pindx,nbas,isite,hmo,eri,pdic,qpts)
         print '\n>>> isite/p=',isite,p
	 print 'ql=',qt12a.qsyms[0]
	 print 'qr=',qt12a.qsyms[1]
   return 0 


if __name__ == '__main__':
   #testElemSingle()
   #testElem()
   #testWfacSpin() 
   #testWfacSpatial() 
   #testHfacSpin() 
   #testHfacSpatial()
   #testHRfacSpatial()
   #testQnumHRfacSpatialQt()
   pass
