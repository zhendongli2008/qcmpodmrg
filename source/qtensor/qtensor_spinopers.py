#!/usr/bin/env python
#
# MPO operators in Qt form for property operators.
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
# 
# def genSpinOpersQnums(nsite,isite,key):
#
# def genS2GlobalSpatialQt(nsite,isite):	# for normal MPS
# def genGlobalSpatialQt(nsite,isite,key):
# def genLocalSpatialQt(nsite,isite,ig,key):
# def genLocal2SpatialQt(nsite,isite,ig,jg,ikey,jkey):
# 
# def genS2GlobalRSpatialQt(nsite,isite,phi):   # for spin projected MPS
# def genGlobalRSpatialQt(nsite,isite,key,phi):
# def genLocalRSpatialQt(nsite,isite,ig,key,phi):
# def genLocal2RSpatialQt(nsite,isite,ig,jg,ikey,jkey,fac,phi):
#
import numpy
import itertools
import qtensor
import qtensor_util
import qtensor_opers
from mpodmrg.source import mpo_dmrg_opers
from mpodmrg.source import mpo_dmrg_spinopers

def genSpinOpersQnums(nsite,isite,key):
   if key in ['N','Sz','Omega','NaNb']: 
      if isite == 0:
	 ql = [[0.,0.]]
	 qr = [[0.,0.],[0.,0.]]
      elif isite == nsite-1:
	 ql = [[0.,0.],[0.,0.]]
	 qr = [[0.,0.]]
      else:
	 ql = [[0.,0.],[0.,0.]]
	 qr = [[0.,0.],[0.,0.]]
   elif key in ['Sp']:
      if isite == 0:
	 ql = [[0.,0.]]
	 qr = [[0.,0.],[0.,1.0]]
      elif isite == nsite-1:
	 ql = [[0.,0.],[0.,1.0]]
	 qr = [[0.,1.0]]
      else:
	 ql = [[0.,0.],[0.,1.0]]
	 qr = [[0.,0.],[0.,1.0]]
   elif key in ['Sm']:
      if isite == 0:
	 ql = [[0.,0.]]
	 qr = [[0.,0.],[0.,-1.0]]
      elif isite == nsite-1:
	 ql = [[0.,0.],[0.,-1.0]]
	 qr = [[0.,-1.0]]
      else:
	 ql = [[0.,0.],[0.,-1.0]]
	 qr = [[0.,0.],[0.,-1.0]]
   elif key in ['S2']:
      if isite == 0:
	 ql = [[0.,0.]]
	 qr = [[0.,0.],[0.,-1.0],[0.,1.0],[0.,0.],[0.,0.]]
      elif isite == nsite-1:
	 ql = [[0.,0.],[0.,-1.0],[0.,1.0],[0.,0.],[0.,0.]]
	 qr = [[0.,0.]]    
      else:                
	 ql = [[0.,0.],[0.,-1.0],[0.,1.0],[0.,0.],[0.,0.]]
	 qr = [[0.,0.],[0.,-1.0],[0.,1.0],[0.,0.],[0.,0.]]
   ql = numpy.array(ql)
   qr = numpy.array(qr)
   return ql,qr

#
#        |
#       \|/
#        |
# --->---W--->--- : status of the MPO site
#  in    |   out
#       \|/
#        |
#
def genS2GlobalSpatialQt(nsite,isite):
   isym = 2 
   # (5,5,4,4)
   site = mpo_dmrg_spinopers.genS2GlobalSpatial(nsite,isite)
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = qtensor_opers.genQphys(isym,isite)
   ql,qr = genSpinOpersQnums(nsite,isite,'S2')
   qt.fromDenseTensor(site,[ql,qr,qu,qd])
   return qt

def genGlobalSpatialQt(nsite,isite,key):
   isym = 2 
   # (D,D,4,4) [D<=2] 
   site = mpo_dmrg_spinopers.genGlobalSpatial(nsite,isite,key)
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = qtensor_opers.genQphys(isym,isite)
   ql,qr = genSpinOpersQnums(nsite,isite,key)
   qt.fromDenseTensor(site,[ql,qr,qu,qd])
   return qt

def genLocalSpatialQt(nsite,isite,ig,key):
   isym = 2 
   # (D,D,4,4) [D<=2] 
   site = mpo_dmrg_spinopers.genLocalSpatial(nsite,isite,ig,key)
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = qtensor_opers.genQphys(isym,isite)
   ql,qr = genSpinOpersQnums(nsite,isite,key)
   qt.fromDenseTensor(site,[ql,qr,qu,qd])
   return qt

def genLocal2SpatialQt(nsite,isite,ig,jg,ikey,jkey,fac):
   isym = 2 
   # (D,D,4,4) [D<=2] 
   site = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,ikey,jkey,fac)
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = qtensor_opers.genQphys(isym,isite)
   qli,qri = genSpinOpersQnums(nsite,isite,ikey)
   qlj,qrj = genSpinOpersQnums(nsite,isite,jkey)
   # Direct product of quantum numbers
   ql = [q1+q2 for q1,q2 in itertools.product(qli,qlj)]
   qr = [q1+q2 for q1,q2 in itertools.product(qri,qrj)]
   qt.fromDenseTensor(site,[ql,qr,qu,qd])
   return qt

#########
# O*Rpt #
#########

def genS2GlobalRSpatialQt(nsite,isite,phi):
   isym = 2 
   # (D,D,4,4) [D<=2] 
   cop = mpo_dmrg_spinopers.genS2GlobalSpatial(nsite,isite)
   rop = mpo_dmrg_opers.genExpISyPhiMat(phi)
   site = numpy.tensordot(cop,rop,axes=([3],[0]))
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = qtensor_opers.genQphys(isym,isite)
   ql,qr = genSpinOpersQnums(nsite,isite,'S2')
   # Reduce symmetry
   qu = qtensor_util.reduceQnumsToN(qu)
   qd = qtensor_util.reduceQnumsToN(qd)
   ql = qtensor_util.reduceQnumsToN(ql)
   qr = qtensor_util.reduceQnumsToN(qr)
   qt.fromDenseTensor(site,[ql,qr,qu,qd],ifcollect=[0,0,0,0])
   return qt

def genGlobalRSpatialQt(nsite,isite,key,phi):
   isym = 2 
   # (D,D,4,4) [D<=2] 
   cop = mpo_dmrg_spinopers.genGlobalSpatial(nsite,isite,key)
   rop = mpo_dmrg_opers.genExpISyPhiMat(phi)
   site = numpy.tensordot(cop,rop,axes=([3],[0]))
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = qtensor_opers.genQphys(isym,isite)
   ql,qr = genSpinOpersQnums(nsite,isite,key)
   # Reduce symmetry
   qu = qtensor_util.reduceQnumsToN(qu)
   qd = qtensor_util.reduceQnumsToN(qd)
   ql = qtensor_util.reduceQnumsToN(ql)
   qr = qtensor_util.reduceQnumsToN(qr)
   qt.fromDenseTensor(site,[ql,qr,qu,qd],ifcollect=[0,0,0,0])
   return qt

def genLocalRSpatialQt(nsite,isite,ig,key,phi):
   isym = 2 
   # (D,D,4,4) [D<=2] 
   cop = mpo_dmrg_spinopers.genLocalSpatial(nsite,isite,ig,key)
   rop = mpo_dmrg_opers.genExpISyPhiMat(phi)
   site = numpy.tensordot(cop,rop,axes=([3],[0]))
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = qtensor_opers.genQphys(isym,isite)
   ql,qr = genSpinOpersQnums(nsite,isite,key)
   # Reduce symmetry
   qu = qtensor_util.reduceQnumsToN(qu)
   qd = qtensor_util.reduceQnumsToN(qd)
   ql = qtensor_util.reduceQnumsToN(ql)
   qr = qtensor_util.reduceQnumsToN(qr)
   qt.fromDenseTensor(site,[ql,qr,qu,qd],ifcollect=[0,0,0,0])
   return qt

def genLocal2RSpatialQt(nsite,isite,ig,jg,ikey,jkey,fac,phi):
   isym = 2 
   # (D,D,4,4) [D<=2] 
   cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,ikey,jkey,fac)
   rop = mpo_dmrg_opers.genExpISyPhiMat(phi)
   site = numpy.tensordot(cop,rop,axes=([3],[0]))
   # Status
   qt = qtensor.qtensor([False,True,False,True])
   # Site physical indices
   qu,qd = qtensor_opers.genQphys(isym,isite)
   qli,qri = genSpinOpersQnums(nsite,isite,ikey)
   qlj,qrj = genSpinOpersQnums(nsite,isite,jkey)
   # Direct product of quantum numbers
   ql = [q1+q2 for q1,q2 in itertools.product(qli,qlj)]
   qr = [q1+q2 for q1,q2 in itertools.product(qri,qrj)]
   # Reduce symmetry
   qu = qtensor_util.reduceQnumsToN(qu)
   qd = qtensor_util.reduceQnumsToN(qd)
   ql = qtensor_util.reduceQnumsToN(ql)
   qr = qtensor_util.reduceQnumsToN(qr)
   qt.fromDenseTensor(site,[ql,qr,qu,qd],ifcollect=[0,0,0,0])
   return qt
