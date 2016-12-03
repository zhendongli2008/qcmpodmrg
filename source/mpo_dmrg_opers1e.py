#!/usr/bin/env python
#
# MPO for 1e operators [nsite=2*K]
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def genHRfacSpatial(pindx,nsite,isite,int1e,qpts,model_u):
# def addLocalOpers(p,nsite,isite,op12,model_u):
# def genHfacSpatial(p,nsite,isite,hq,model_u):
# def genHfacSpin(p,nsite,isite,hq):
# def genWfacSpin(nsite,isite,hq):
# 
import math
import numpy
import mpo_dmrg_const
import mpo_dmrg_opers
import mpo_dmrg_spinopers
from sysutil_include import dmrg_dtype,dmrg_mtype

# Note that int1e are now full integrals!
# porb  - spin orbital
# int1e - integrals in spin orbital
# nsite - no. of spin orbitals
def genHRfacSpatial(pindx,nsite,isite,int1e,qpts,model_u):
   porb,ipop = pindx
   wop = genHfacSpatial(porb,nsite,isite,int1e[porb],model_u)
   if ipop is not None:
      pop = mpo_dmrg_opers.genExpISyPhiMat(qpts[ipop])
      wop = numpy.tensordot(wop,pop,axes=([3],[0]))
   return wop

#============================================================================

def addLocalOpers(p,nsite,isite,op12,model_u):
   psite = p//2
   if isite == 0:
      op12new = numpy.zeros((1,3,4,4),dtype=dmrg_dtype)
      op12new[0,0] = op12[0,0].copy()
      op12new[0,1] = op12[0,1].copy()
      if psite == 0:
	 op12new[0,2] = mpo_dmrg_spinopers.genNaNbMat()*model_u*0.5
      else:
	 op12new[0,2] = mpo_dmrg_spinopers.genIpMat()
   elif isite == nsite//2-1:
      op12new = numpy.zeros((3,1,4,4),dtype=dmrg_dtype)
      op12new[0,0] = op12[0,0].copy()
      op12new[1,0] = op12[1,0].copy()
      if psite == nsite//2-1:
	 op12new[2,0] = mpo_dmrg_spinopers.genNaNbMat()*model_u*0.5
      else:
	 op12new[2,0] = mpo_dmrg_spinopers.genIpMat() 
   else:
      op12new = numpy.zeros((3,3,4,4),dtype=dmrg_dtype)
      op12new[0,0] = op12[0,0].copy()
      op12new[0,1] = op12[0,1].copy()
      op12new[1,0] = op12[1,0].copy()
      op12new[1,1] = op12[1,1].copy()
      if psite == isite:
	 op12new[2,2] = mpo_dmrg_spinopers.genNaNbMat()*model_u*0.5
      else:
	 op12new[2,2] = mpo_dmrg_spinopers.genIpMat() 
   return op12new

#============================================================================

# Hp = ap^+Vqrs: p - spin orbitals
def genHfacSpatial(p,nsite,isite,hq,model_u):
   op1 = genHfacSpin(p,nsite,2*isite  ,hq)
   op2 = genHfacSpin(p,nsite,2*isite+1,hq)
   op12 = mpo_dmrg_opers.linkTwoOpers(op1,op2) # [l,r,u,d]
   if abs(model_u)>1.e-12: 
      op12 = addLocalOpers(p,nsite,isite,op12,model_u)
   return op12

# Hp = ap^+*Vp
def genHfacSpin(p,nsite,isite,hq):
   wfac = genWfacSpin(nsite,isite,hq)
   elem = mpo_dmrg_opers.genElemSpinMat(p,isite)
   hfac = numpy.tensordot(wfac,elem,axes=([2],[1])) # abjk,ij-> abki
   hfac = hfac.transpose(0,1,3,2).copy() 	    # abki->abik
   return hfac

# On-the-fly generatation of W[isite,iop] based on three index integrals
#
#  	       [ Ik'  hk*ak ]  [ hk*ak ]
# [Ik' hk*ak]  [  	    ]  [       ] 
# 	       [ 0k     Ik  ]  [   Ik  ]
#
# isite is the spin-orbital site here.
def genWfacSpin(nsite,isite,hq):
   if isite == 0:
      wop = numpy.zeros((1,2,2,2),dtype=dmrg_dtype)
      wop[0,0] = mpo_dmrg_const.idnt
      wop[0,1] = hq[isite]*mpo_dmrg_const.ann 
   elif isite == nsite-1:
      wop = numpy.zeros((2,1,2,2),dtype=dmrg_dtype)
      wop[0,0] = hq[isite]*mpo_dmrg_const.ann 
      wop[1,0] = mpo_dmrg_const.idn
   else:
      wop = numpy.zeros((2,2,2,2),dtype=dmrg_dtype)
      wop[0,0] = mpo_dmrg_const.idnt
      wop[0,1] = hq[isite]*mpo_dmrg_const.ann 
      wop[1,1] = mpo_dmrg_const.idn
   return wop
