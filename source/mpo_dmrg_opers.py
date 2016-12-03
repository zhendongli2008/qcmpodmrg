#!/usr/bin/env python
#
# Each MPO is a collection of site operators [nsite=2*K]
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def prodTwoOpers(op1,op2):
# def linkTwoOpers(op1,op2):
#
# def genElemSpinMat(p,isite,iop=1):
# def genElemSpatialMat(p,isite,iop):
# def genElemSpin(p,isite,iop=1):
# def genElemSpatial(p,isite,iop=1):
# def genElemProductSpatial(oplst,isite):
#
# def genWfacSpin(nsite,isite,hq,vqrs):
# def genHfacSpin(p,nsite,isite,hq,vqrs):
# def genWfacSpatial(nsite,isite,hq,vqrs):
# def genHfacSpatial(p,nsite,isite,hq,vqrs):
#
# def genExpISyPhiMat(phi):
# def genExpISyPhi(phi):
# def genElemProductRSpatial(oplst,isite):
# def genHRfacSpatial(pindx,nsite,isite,int1e,int2e,qpts,pdic):
# 
import math
import numpy
import mpo_dmrg_const
from sysutil_include import dmrg_dtype,dmrg_mtype

# Product vertically
def prodTwoOpers(op1,op2):
   # lrud,LRdD->lruLRD->lLrRuD
   op12 = numpy.tensordot(op1,op2,axes=([3],[2]))
   op12 = op12.transpose(0,3,1,4,2,5)
   s = op12.shape
   op12 = op12.reshape((s[0]*s[1],s[2]*s[3],s[4],s[5]))
   return op12

# Join MPO sites horizontally: ---W1---W2---
def linkTwoOpers(op1,op2):
   # Target: abij,bckl->ac(ik)(jl)
   op12 = numpy.tensordot(op1,op2,axes=([1],[0])) # abij,bckl->aijckl
   op12 = op12.transpose(0,3,1,4,2,5) # aijckl->acikjl
   s = op12.shape
   op12 = op12.reshape((s[0],s[1],s[2]*s[3],s[4]*s[5]))
   return op12

#####################
# Spin-orbital sites 
#####################
 
# Creation or annhilation
def genElemSpinMat(p,isite,iop=1):
   if isite<p:
      op = mpo_dmrg_const.sgn
   elif isite==p:
      if iop == 1:
         op = mpo_dmrg_const.cre
      else:
         op = mpo_dmrg_const.ann
   elif isite>p:
      op = mpo_dmrg_const.idn
   return op

def genElemSpatialMat(p,isite,iop):
   mat1 = genElemSpinMat(p,2*isite  ,iop)
   mat2 = genElemSpinMat(p,2*isite+1,iop)
   mat12 = numpy.kron(mat1,mat2)
   return mat12

def genElemSpin(p,isite,iop=1):
  mat = genElemSpinMat(p,isite,iop)
  return mat.reshape(1,1,2,2)

def genElemSpatial(p,isite,iop=1):
  mat = genElemSpatialMat(p,isite,iop)
  return mat.reshape(1,1,4,4)

# [[p0,iop0],...]
def genElemProductSpatial(oplst,isite):
   p0,iop0 = oplst[0] 
   mat = genElemSpatialMat(p0,isite,iop0)
   for p1,iop1 in oplst[1:]:
      mat = mat.dot(genElemSpatialMat(p1,isite,iop1))
   return mat.reshape(1,1,4,4)

# On-the-fly generatation of W[isite,iop] based on three index integrals
def genWfacSpin(nsite,isite,hq,vqrs):
   ndim1 = 2*(nsite+1)
   ndim2 = 2*(nsite+1)
   if isite==0: ndim1=1
   if isite==nsite-1: ndim2=1
   wop = numpy.zeros((ndim1,ndim2,2,2),dtype=dmrg_dtype)
   diml = isite
   dimr = nsite-isite-1
   if isite == 0:
      # blk[0,0]
      wop[0,0] = mpo_dmrg_const.idnt
      # blk[0,2]
      off = 1+dimr
      rindx = 0
      for rsite in range(isite+1,nsite):
         wop[0,off+rindx] = vqrs[isite,isite,rsite]*mpo_dmrg_const.niit
	 rindx += 1
      # blk[0,4]
      off = 1+2*dimr+diml
      wop[0,off] = -mpo_dmrg_const.ann
      # blk[0,6]
      off = 2+2*dimr+2*diml
      wop[0,off] = mpo_dmrg_const.cre
      # blk[0,7]
      wop[0,ndim2-1] = hq[isite]*mpo_dmrg_const.ann 
   elif isite == nsite-1:
      # blk[1,0]
      wop[0,0] = hq[isite]*mpo_dmrg_const.ann 
      # blk[1,0]
      wop[1,0] = mpo_dmrg_const.cre
      # blk[2,0]
      wop[2,0] = mpo_dmrg_const.ann
      # blk[3,0]
      for jsite in range(isite):
         wop[3+jsite,0] = vqrs[isite,jsite,isite]*mpo_dmrg_const.nii
      # blk[5,0]	 
      wop[ndim1-1,0] = mpo_dmrg_const.idn
   else:
      # ROW-1:
      # blk[0,0]
      wop[0,0] = mpo_dmrg_const.idnt
      # blk[0,2]
      off = 1+dimr
      rindx = 0
      for rsite in range(isite+1,nsite):
         wop[0,off+rindx] = vqrs[isite,isite,rsite]*mpo_dmrg_const.niit
	 rindx += 1
      # blk[0,4]
      off = 1+2*dimr+diml
      wop[0,off] = -mpo_dmrg_const.ann
      # blk[0,6]
      off = 2+2*dimr+2*diml
      wop[0,off] = mpo_dmrg_const.cre
      # blk[0,7]
      wop[0,ndim2-1] = hq[isite]*mpo_dmrg_const.ann 
      # ROW-2:
      wop[1,ndim2-1] = mpo_dmrg_const.cre
      # ROW-3:
      rindx = 0
      for rsite in range(isite+1,nsite):
	 wop[2+rindx,1+rindx] = mpo_dmrg_const.idnt
	 rindx += 1
      # ROW-4:
      offl = 2+dimr
      wop[offl,ndim2-1] = mpo_dmrg_const.ann
      # ROW-5:
      offl = 3+dimr
      offr = 1+dimr
      rindx = 0
      for rsite in range(isite+1,nsite):
         wop[offl+rindx,offr+rindx] = mpo_dmrg_const.idnt
	 rindx += 1
      # ROW-6:
      offl = 3+2*dimr
      for lsite in range(isite):
        rindx = 0
        for rsite in range(isite+1,nsite):
	   wop[offl+lsite,1+rindx] = -vqrs[rsite,lsite,isite]*mpo_dmrg_const.annt
	   rindx += 1
      offr = 1+dimr
      for lsite in range(isite):
        rindx = 0
        for rsite in range(isite+1,nsite):
	   wop[offl+lsite,offr+rindx] = vqrs[isite,lsite,rsite]*mpo_dmrg_const.cret
	   rindx += 1
      offr = 1+2*dimr
      for lsite in range(isite):
	  wop[offl+lsite,offr+lsite] = mpo_dmrg_const.idn
      for lsite in range(isite):
          wop[offl+lsite,ndim2-1] = vqrs[isite,lsite,isite]*mpo_dmrg_const.nii
      # ROW-7:
      offl = 3+2*dimr+diml
      offr = 1+dimr
      for lsite in range(isite):
        rindx = 0
	for rsite in range(isite+1,nsite):
	   wop[offl+lsite,offr+rindx] = vqrs[lsite,isite,rsite]*mpo_dmrg_const.annt
	   rindx += 1
      offr = 2+2*dimr+diml
      for lsite in range(isite):
	  wop[offl+lsite,offr+lsite] = mpo_dmrg_const.idn
      # ROW-8: 
      wop[ndim1-1,ndim2-1] = mpo_dmrg_const.idn
   return wop

# Hp = ap^+*Vp
def genHfacSpin(p,nsite,isite,hq,vqrs):
   wfac = genWfacSpin(nsite,isite,hq,vqrs)
   elem = genElemSpinMat(p,isite)
   #hfac = numpy.einsum('ij,abjk->abik',elem,wfac)
   hfac = numpy.tensordot(wfac,elem,axes=([2],[1])) # abjk,ij-> abki
   hfac = hfac.transpose(0,1,3,2).copy() 	    # abki->abik
   return hfac

########################
# Spatial-orbital sites
########################

# Three-index operator Vqrs, nsite=2*K
def genWfacSpatial(nsite,isite,hq,vqrs):
   op1 = genWfacSpin(nsite,2*isite  ,hq,vqrs)
   op2 = genWfacSpin(nsite,2*isite+1,hq,vqrs)
   return linkTwoOpers(op1,op2)

# Hp = ap^+Vqrs
def genHfacSpatial(p,nsite,isite,hq,vqrs):
   op1 = genHfacSpin(p,nsite,2*isite  ,hq,vqrs)
   op2 = genHfacSpin(p,nsite,2*isite+1,hq,vqrs)
   return linkTwoOpers(op1,op2)

########################
# Spin-rotation version
########################

# Integrand for exp(-i*phi*Sy)
# [  c  s ]
# [ -s  c ]
def genExpISyPhiMat(phi):
   expm = numpy.zeros((4,4),dtype=dmrg_dtype)
   expm[0,0] = 1.0
   c = math.cos(0.5*phi)
   s = math.sin(0.5*phi)
   expm[1,1] = c 
   expm[1,2] = s 
   expm[2,1] = -s 
   expm[2,2] = c 
   expm[3,3] = 1.0
   return expm

def genExpISyPhi(phi):
   tmp = genExpISyPhiMat(phi)
   tmp = tmp.reshape(1,1,4,4)
   return tmp 

# [[p0,iop0],...]
def genElemProductRSpatial(oplst,isite,phi):
   cop = genElemProductSpatial(oplst,isite)
   pop = genExpISyPhiMat(phi)
   wop = numpy.tensordot(cop,pop,axes=([3],[0]))
   return wop

# Note that int1e,int2e are now full integrals!
def genHRfacSpatial(pindx,nsite,isite,int1e,int2e,qpts,pdic):
   porb,ipop = pindx
   idx = pdic[porb]
   wop = genHfacSpatial(porb,nsite,isite,int1e[idx],int2e[idx])
   if ipop is not None:
      pop = genExpISyPhiMat(qpts[ipop])
      wop = numpy.tensordot(wop,pop,axes=([3],[0]))
   return wop
