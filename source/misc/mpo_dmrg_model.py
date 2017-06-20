#!/usr/bin/env python
#
# Generation of MPO for 1D Hubbard model
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def genSaSpatialMat():
# def genSbSpatialMat():
# def genSaSbSpatialMat():
# def genHubbardSpatial(dmrg,isite):
# 
import numpy
from qcmpodmrg.source import mpo_dmrg_opers
from qcmpodmrg.source import mpo_dmrg_spinopers

def genSaSpatialMat():
   # signAlpha*idenBeta	
   # [ 1    ]  [ 1    ]
   # [   -1 ]  [    1 ] 
   mat = numpy.identity(4)
   mat[2,2] = -1
   mat[3,3] = -1
   return mat

def genSbSpatialMat():
   # idenAlpha*signBeta
   # [ 1    ]  [ 1    ]
   # [    1 ]  [   -1 ] 
   mat = numpy.identity(4)
   mat[1,1] = -1
   mat[3,3] = -1
   return mat

def genSaSbSpatialMat():
   # signAlpha*signBeta
   # [ 1    ]  [ 1    ]
   # [   -1 ]  [   -1 ] 
   mat = numpy.identity(4)
   mat[1,1] = -1
   mat[2,2] = -1
   return mat

# Hubbard:
def genHubbardSpatial(dmrg,isite):
   iden = mpo_dmrg_spinopers.genIpMat()
   nanb = mpo_dmrg_spinopers.genNaNbMat()
   creA = mpo_dmrg_opers.genElemSpatialMat(2*isite  ,isite,1)
   creB = mpo_dmrg_opers.genElemSpatialMat(2*isite+1,isite,1)
   annA = mpo_dmrg_opers.genElemSpatialMat(2*isite  ,isite,0)
   annB = mpo_dmrg_opers.genElemSpatialMat(2*isite+1,isite,0)
   # ai^+[alpha]*aj[alpha]   =  creA*sgnA*sgnB * annA 
   # -ai[alpha]*aj^+[alpha]  = -annA*sgnA*sgnB * creA
   # ai^+[beta]*aj[beta]     =  sgnA*creB*sgnB * annB
   # -ai[beta]*aj^+[beta]   =  -sgnA*annB*sgnB * creB
   sa = genSaSpatialMat()
   sb = genSbSpatialMat()
   sasb = genSaSbSpatialMat()
   creAs = numpy.dot(creA,sasb)
   annAs = numpy.dot(annA,sasb)
   # Note that creB already contains a sign factor sgnA, so
   # an additional sgnA is needed to cancel it.
   creBs = numpy.dot(sa,numpy.dot(creB,sb))
   annBs = numpy.dot(sa,numpy.dot(annB,sb))
   # Set up elements
   if isite == 0:
      # [I C D] = [ I (-t)au^+ (-t)ad^+ (-t)au (-t)ad U*niA*niB ]
      cop = numpy.zeros((1,6,4,4))
      cop[0,0] = iden
      cop[0,1] = -dmrg.model_t * creAs
      cop[0,2] = -dmrg.model_t * creBs
      cop[0,3] = -dmrg.model_t * annAs
      cop[0,4] = -dmrg.model_t * annBs
      cop[0,5] = dmrg.model_u * nanb 
   elif isite == dmrg.nsite-1:
      # [D B I]^T 	   
      cop = numpy.zeros((6,1,4,4))
      cop[0,0] = dmrg.model_u * nanb 
      cop[1,0] = annA  
      cop[2,0] = annB
      cop[3,0] = -creA
      cop[4,0] = -creB
      cop[5,0] = iden 
   else:
      # [ I  -t*a^+ -t*a D  ]
      # [ 0    0     0   a  ]
      # [ 0    0     0  -a^+]
      # [ 0    0     0   I  ]
      cop = numpy.zeros((6,6,4,4))
      cop[0,0] = iden
      cop[0,1] = -dmrg.model_t * creAs
      cop[0,2] = -dmrg.model_t * creBs
      cop[0,3] = -dmrg.model_t * annAs
      cop[0,4] = -dmrg.model_t * annBs
      cop[0,5] = dmrg.model_u * nanb
      cop[1,5] = annA 
      cop[2,5] = annB
      cop[3,5] = -creA
      cop[4,5] = -creB
      cop[5,5] = iden 
   return cop
