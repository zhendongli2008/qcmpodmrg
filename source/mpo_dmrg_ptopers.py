#!/usr/bin/env python
#
# Generation of MPO for H0 used in perturbation theory
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
# 
# def genHenRfacSpatial(dmrg,pindx,isite,icase):
# def genHDiagfacSpatial(dmrg,porb,isite):
# def genHBlkDiagfacSpatial(dmrg,porb,isite):
# def genHDiagSFfacSpatial(dmrg,porb,isite):
# 
import h5py
import time
import numpy
import mpo_dmrg_opers
import mpo_dmrg_spinopers
from sysutil_include import dmrg_dtype,dmrg_mtype

# EN-like H0
def genHenRfacSpatial(dmrg,pindx,isite,icase):
   porb,ipop = pindx
   if icase == 0:
      cop = genHDiagfacSpatial(dmrg,porb,isite)
   elif icase == 1:
      cop = genHBlkDiagfacSpatial(dmrg,porb,isite)
   elif icase == 2:
      cop = genHDiagSFfacSpatial(dmrg,porb,isite)
   # H[spinfree]*R 
   if ipop is not None:
      if icase == 1 or icase == 2:
	 pop = mpo_dmrg_opers.genExpISyPhiMat(dmrg.qpts[ipop])
         cop = numpy.tensordot(cop,pop,axes=([3],[0]))
      else:
         print '\nerror: icase must be 1 or 2 for ifs2proj = True!'
	 exit()
   return cop

# Diagonal in the Fock space: O(3*K)
def genHDiagfacSpatial(dmrg,porb,isite):
   psite = porb//2
   pspin = porb%2
   iden = mpo_dmrg_spinopers.genIpMat()
   matp = mpo_dmrg_spinopers.genNpSpinMat(pspin)
   # Term-1: hpp*Epps
   if isite == psite:
      mat1 = dmrg.sint1e[psite,psite]*matp
      mats = matp.copy()
   else:
      mat1 = iden.copy()
      mats = iden.copy()
   # Term-2: \sum_q 1/2*gppqq*Eqq - 1/2*gpqqp*Eqqs
   mat2 = mpo_dmrg_spinopers.genNpMat() 
   mat2 = 0.5*dmrg.sint2e[psite,psite,isite,isite]*mat2
   mat2 -= 0.5*dmrg.sint2e[psite,isite,isite,psite]*matp
   # Set up elements
   if isite == 0:
      # [ X   I   Y ]
      cop = numpy.zeros((1,3,4,4),dtype=dmrg_dtype)
      cop[0,0] = mat1 
      cop[0,1] = mats.dot(iden)
      cop[0,2] = mats.dot(mat2)
   elif isite == dmrg.nsite-1:
      # [ X   Y   I ]^T
      cop = numpy.zeros((3,1,4,4),dtype=dmrg_dtype)
      cop[0,0] = mat1
      cop[1,0] = mats.dot(mat2)
      cop[2,0] = mats.dot(iden)
   else:
      # [ X   0   0 ]
      # [ 0   I   Y ]
      # [ 0   0   I ]
      cop = numpy.zeros((3,3,4,4),dtype=dmrg_dtype)
      cop[0,0] = mat1
      cop[1,1] = mats.dot(iden)
      cop[2,2] = mats.dot(iden)
      cop[1,2] = mats.dot(mat2)
   return cop

# Block diagonal: D=O(5*K)
def genHBlkDiagfacSpatial(dmrg,porb,isite):
   psite = porb//2
   pspin = porb%2
   qspin = (~pspin)+2
   iden = mpo_dmrg_spinopers.genIpMat()
   matp = mpo_dmrg_spinopers.genNpSpinMat(pspin)
   # Term-1: kpp*Epps + (pp|pp)Epps - 0.5*(pp|pp)Epps*Epp
   if isite == psite:
      #########################
      ### TO BE SIMPLIFIED! ###
      #########################
      pss = mpo_dmrg_spinopers.genNpXYMat(pspin,pspin)
      pos = mpo_dmrg_spinopers.genNpXYMat(pspin,qspin)
      tpp = pss.dot(pss)+pos.dot(pos.T)
      mat1 = (dmrg.sint1e[psite,psite] + \
          - 0.5*dmrg.sint2e[psite,psite,psite,psite])*matp \
	  + 0.5*dmrg.sint2e[psite,psite,psite,psite]*tpp
      mats = matp.copy()
   else:
      mat1 = iden.copy()
      mats = iden.copy()
   # Term-2: Epp * (\sum_q 1/2*gppqq*Eqq) - 1/2*gpqqp*Eqqs (SS) 
   mat2 = mpo_dmrg_spinopers.genNpMat() 
   mat2 = 0.5*dmrg.sint2e[psite,psite,isite,isite]*mat2
   mat2 -= 0.5*dmrg.sint2e[psite,isite,isite,psite]*matp
   #
   # Additional term: Opposite spin part
   # Term-3: \sum_t Epss' * (\sum_q -1/2*gpqqp*Eqs's)
   #
   if isite == psite:
      pos = mpo_dmrg_spinopers.genNpXYMat(pspin,qspin)
   else:
      pos = iden.copy()
   qos = mpo_dmrg_spinopers.genNpXYMat(qspin,pspin)
   qos = -0.5*dmrg.sint2e[psite,isite,isite,psite]*qos
   # Set up elements
   if isite == 0:
      # [ X  I  Y  I  Y1 ]
      cop = numpy.zeros((1,5,4,4),dtype=dmrg_dtype)
      cop[0,0] = mat1 
      cop[0,1] = mats.dot(iden)
      cop[0,2] = mats.dot(mat2)
      cop[0,3] = pos.dot(iden)
      cop[0,4] = pos.dot(qos)
   elif isite == dmrg.nsite-1:
      # [ X  Y  I  Y1  I ]^T
      cop = numpy.zeros((5,1,4,4),dtype=dmrg_dtype)
      cop[0,0] = mat1
      cop[1,0] = mats.dot(mat2)
      cop[2,0] = mats.dot(iden)
      cop[3,0] = pos.dot(qos)
      cop[4,0] = pos.dot(iden)
   else:
      # [ X  0  0  0  0  ]
      # [ 0  I  Y  0  0  ]
      # [ 0  0  I  0  0  ]
      # [ 0  0  0  I  Y1 ]
      # [ 0  0  0  0  I  ]
      cop = numpy.zeros((5,5,4,4),dtype=dmrg_dtype)
      cop[0,0] = mat1
      cop[1,1] = mats.dot(iden)
      cop[2,2] = mats.dot(iden)
      cop[1,2] = mats.dot(mat2)
      cop[3,3] = pos.dot(iden)
      cop[4,4] = pos.dot(iden)
      cop[3,4] = pos.dot(qos)
   return cop

# Spin-free version for singlet space: Hd[p]
def genHDiagSFfacSpatial(dmrg,porb,isite):
   psite = porb//2
   pspin = porb%2
   qspin = (~pspin)+2
   iden = mpo_dmrg_spinopers.genIpMat()
   matp = mpo_dmrg_spinopers.genNpSpinMat(pspin)
   # Term-1: hpp*Epps
   if isite == psite:
      mat1 = dmrg.sint1e[psite,psite]*matp
      mats = matp.copy()
   else:
      mat1 = iden.copy()
      mats = iden.copy()
   # Term-2: Epss * (\sum_q jpq*Eqq - kpq*Eqqs)
   mat2 = mpo_dmrg_spinopers.genNpMat() 
   mat2 = (1.0/2.0*dmrg.sint2e[psite,psite,isite,isite]
          -1.0/6.0*dmrg.sint2e[psite,isite,isite,psite])*mat2
   mat2 -= 1.0/6.0*dmrg.sint2e[psite,isite,isite,psite]*matp
   #
   # Additional term: Opposite spin part
   # Term-3: Epss' * (\sum_q -kpq*Eqs's)
   #
   if isite == psite:
      pos = mpo_dmrg_spinopers.genNpXYMat(pspin,qspin)
   else:
      pos = iden.copy()
   qos = mpo_dmrg_spinopers.genNpXYMat(qspin,pspin)
   qos = -1.0/6.0*dmrg.sint2e[psite,isite,isite,psite]*qos
   # Set up elements
   if isite == 0:
      # [ X  I  Y  I  Y1 ]
      cop = numpy.zeros((1,5,4,4),dtype=dmrg_dtype)
      cop[0,0] = mat1 
      cop[0,1] = mats.dot(iden)
      cop[0,2] = mats.dot(mat2)
      cop[0,3] = pos.dot(iden)
      cop[0,4] = pos.dot(qos)
   elif isite == dmrg.nsite-1:
      # [ X  Y  I  Y1  I ]^T
      cop = numpy.zeros((5,1,4,4),dtype=dmrg_dtype)
      cop[0,0] = mat1
      cop[1,0] = mats.dot(mat2)
      cop[2,0] = mats.dot(iden)
      cop[3,0] = pos.dot(qos)
      cop[4,0] = pos.dot(iden)
   else:
      # [ X  0  0  0  0  ]
      # [ 0  I  Y  0  0  ]
      # [ 0  0  I  0  0  ]
      # [ 0  0  0  I  Y1 ]
      # [ 0  0  0  0  I  ]
      cop = numpy.zeros((5,5,4,4),dtype=dmrg_dtype)
      cop[0,0] = mat1
      cop[1,1] = mats.dot(iden)
      cop[2,2] = mats.dot(iden)
      cop[1,2] = mats.dot(mat2)
      cop[3,3] = pos.dot(iden)
      cop[4,4] = pos.dot(iden)
      cop[3,4] = pos.dot(qos)
   return cop
