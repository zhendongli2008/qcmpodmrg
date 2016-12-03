#!/usr/bin/env python
#
# Matrix Representation for On-Site Operators 
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def genIpMat():
# def genNpMat():
# def genOmegaMat():
# def genNaNbMat():  
# def genNpSpinMat(ispin):
# def genNpXYMat(ispin,jspin):
# def genSzMat():
# def genSpMat():
# def genSmMat():
# def genS2Mat():
#
# def genS2GlobalSpatial(nsite,isite):
# def genGlobalSpatial(nsite,isite,key):
# def genLocalSpatial(nsite,isite,ig,key):
# def genLocal2Spatial(nsite,isite,ig,jg,ikey,jkey,fac=1.0):
# 
import numpy
import mpo_dmrg_opers
from sysutil_include import dmrg_dtype,dmrg_mtype

# Op = I
# [[1 0 0 0],
#  [0 1 0 0],
#  [0 0 1 0],
#  [0 0 0 1]]
def genIpMat():
   return numpy.identity(4,dtype=dmrg_dtype)

# Op = N = Na+Nb
# [[0 0 0 0],
#  [0 1 0 0],
#  [0 0 1 0],
#  [0 0 0 2]]
def genNpMat():
   tmp = numpy.zeros((4,4),dtype=dmrg_dtype)
   tmp[1,1] = 1.
   tmp[2,2] = 1.
   tmp[3,3] = 2.
   return tmp

# Op = Omega for seniority
# [[ 0.    0.    0.    0.  ]
#  [ 0.    1.    0.    0.  ]
#  [ 0.    0.    1.    0.  ]
#  [ 0.    0.    0.    0.  ]]
def genOmegaMat():
   tmp = numpy.zeros((4,4),dtype=dmrg_dtype)
   tmp[1,1] = 1.0
   tmp[2,2] = 1.0
   return tmp

# Op = Na*Nb, and this is for doubly occupancy!
# [[0 0 0 0],
#  [0 0 0 0],
#  [0 0 0 0],
#  [0 0 0 1]]
def genNaNbMat():
   tmp = numpy.zeros((4,4),dtype=dmrg_dtype)
   tmp[3,3] = 1.
   return tmp

# Op = Na or Nb
def genNpSpinMat(ispin):
   tmp = numpy.zeros((4,4),dtype=dmrg_dtype)
   # Alpha
   if ispin == 0:
      tmp[2,2] = 1.
      tmp[3,3] = 1.
   # Beta
   elif ispin == 1:
      tmp[1,1] = 1.
      tmp[3,3] = 1.
   return tmp

# Op = a^+[ispin]*a[jspin]
def genNpXYMat(ispin,jspin):
   tmp = numpy.zeros((4,4),dtype=dmrg_dtype)
   # a^+*a or b^+*b
   if ispin == jspin:
      tmp = genNpSpinMat(ispin)
   else:
      # a^+*b	   
      if ispin == 0 and jspin == 1:
         tmp[2,1] = 1.
      elif ispin == 1 and jspin == 0:
	 tmp[1,2] = 1.
      else:
	 print 'error: no such spin case in genNpXYMat! ispin,jspin=',(ispin,jspin)
	 exit(1)
   return tmp

# Op = Sz = 0.5*(pa^+pa-pb^+pb) 
# [[ 0.   0.   0.   0. ]
#  [ 0.  -0.5  0.   0. ]
#  [ 0.   0.   0.5  0. ]
#  [ 0.   0.   0.   0. ]]
def genSzMat():
   tmp = numpy.zeros((4,4),dtype=dmrg_dtype)
   tmp[1,1] = -0.5
   tmp[2,2] = 0.5
   return tmp

# Op = S+
# [[ 0.  0.  0.  0.]
#  [ 0.  0.  0.  0.]
#  [ 0.  1.  0.  0.]
#  [ 0.  0.  0.  0.]]
def genSpMat():
   tmp = numpy.zeros((4,4),dtype=dmrg_dtype)
   tmp[2,1] = 1.0
   return tmp

# Op = S-
# [[ 0.  0.  0.  0.]
#  [ 0.  0.  1.  0.]
#  [ 0.  0.  0.  0.]
#  [ 0.  0.  0.  0.]]
def genSmMat():
   tmp = numpy.zeros((4,4),dtype=dmrg_dtype)
   tmp[1,2] = 1.
   return tmp

# Op = S^2
# [[ 0.    0.    0.    0.  ]
#  [ 0.    0.75  0.    0.  ]
#  [ 0.    0.    0.75  0.  ]
#  [ 0.    0.    0.    0.  ]]
def genS2Mat():
   tmp = numpy.zeros((4,4),dtype=dmrg_dtype)
   tmp[1,1] = 0.75
   tmp[2,2] = 0.75
   return tmp

####################
# Rank-4 MPO Sites #
####################

# S^2
def genS2GlobalSpatial(nsite,isite):
   ik = genIpMat()
   sz = genSzMat()
   sp = genSpMat()
   sm = genSmMat()
   s2 = genS2Mat()
   if isite == 0: 
      wop = numpy.zeros((1,5,4,4),dtype=dmrg_dtype)
      wop[0,0] = ik
      wop[0,1] = sm
      wop[0,2] = sp
      wop[0,3] = 2.0*sz
      wop[0,4] = s2
   elif isite == nsite-1:
      wop = numpy.zeros((5,1,4,4),dtype=dmrg_dtype)
      wop[0,0] = s2
      wop[1,0] = sp
      wop[2,0] = sm
      wop[3,0] = sz
      wop[4,0] = ik
   else:
      wop = numpy.zeros((5,5,4,4))
      wop[0,0] = ik
      wop[1,1] = ik
      wop[2,2] = ik
      wop[3,3] = ik
      wop[4,4] = ik
      wop[0,1] = sm
      wop[0,2] = sp
      wop[0,3] = 2.0*sz
      wop[0,4] = s2
      wop[1,4] = sp
      wop[2,4] = sm
      wop[3,4] = sz
      wop[4,4] = ik
   return wop

# Global operators at a given site = \sum_{ig}O[ig] 
def genGlobalSpatial(nsite,isite,key):
   # Operator cases
   if key == 'N':
      mat = genNpMat()
   elif key == 'Sz':
      mat = genSzMat()
   elif key == 'Sp':
      mat = genSpMat()
   elif key == 'Sm':
      mat = genSmMat()
   elif key == 'Omega':
      mat = genOmegaMat()
   elif key == 'NaNb':
      mat = genNaNbMat()
   else:
      print 'error: no such key in genGlobalSpatial! key =',key
      exit(1)
   # Site cases
   if isite == 0: 
      wop = numpy.zeros((1,2,4,4),dtype=dmrg_dtype)
      wop[0,0] = genIpMat()
      wop[0,1] = mat
   elif isite == nsite-1:
      wop = numpy.zeros((2,1,4,4),dtype=dmrg_dtype)
      wop[0,0] = mat
      wop[1,0] = genIpMat()
   else:
      wop = numpy.zeros((2,2,4,4))
      wop[0,0] = genIpMat()
      wop[1,1] = genIpMat()
      wop[0,1] = mat
   return wop

###########################
# MPO for local operators #
###########################

# Local operators at a given site - O[ig]
def genLocalSpatial(nsite,isite,ig,key):
   # Local cases
   if isite in ig:
      fac = 1.
   else:
      fac = 0.
   # Operator cases
   if key == 'N':
      mat = genNpMat()
   elif key == 'Sz':
      mat = genSzMat()
   elif key == 'Sp':
      mat = genSpMat()
   elif key == 'Sm':
      mat = genSmMat()
   elif key == 'Omega':
      mat = genOmegaMat()
   elif key == 'NaNb':
      mat = genNaNbMat()
   # Site cases
   if isite == 0: 
      wop = numpy.zeros((1,2,4,4),dtype=dmrg_dtype)
      wop[0,0] = genIpMat()
      wop[0,1] = fac*mat
   elif isite == nsite-1:
      wop = numpy.zeros((2,1,4,4),dtype=dmrg_dtype)
      wop[0,0] = fac*mat
      wop[1,0] = genIpMat()
   else:
      wop = numpy.zeros((2,2,4,4))
      wop[0,0] = genIpMat()
      wop[1,1] = genIpMat()
      wop[0,1] = fac*mat
   return wop

# Composite MPO with bond dimension 2^2=4 
# It could be used for various correlation functions like
# {S+[ig]S-[jg], S-[ig]S+[jg], Sz[ig]Sz[jg], NiNj, ...}.
def genLocal2Spatial(nsite,isite,ig,jg,ikey,jkey,fac):
   si = genLocalSpatial(nsite,isite,ig,ikey)
   sj = genLocalSpatial(nsite,isite,jg,jkey)
   return mpo_dmrg_opers.prodTwoOpers(si,sj)*fac

if __name__ == '__main__':
   paa = genNpXYMat(0,0)
   pab = genNpXYMat(0,1)
   pba = genNpXYMat(1,0)
   pbb = genNpXYMat(1,1)
   print 'paa\n',paa
   print 'pab\n',pab
   print 'pba\n',pba
   print 'pbb\n',pbb
   import mpo_dmrg_const
   kab = numpy.kron(mpo_dmrg_const.cret,mpo_dmrg_const.ann)
   kba = -numpy.kron(mpo_dmrg_const.annt,mpo_dmrg_const.cre)
   print 'kab\n',kab
   print 'pab-kab =',numpy.linalg.norm(pab-kab)
   print 'kba\n',kba
   print 'pba-kba =',numpy.linalg.norm(pba-kba)
   print paa.dot(paa)+pab.dot(pba)
   print pbb.dot(pbb)+pba.dot(pab)

   # S^2 local for one orbital?
   sz = genSzMat()
   sp = genSpMat()
   sm = genSmMat()
   print 'sz\n',sz
   print 'sp\n',sp
   print 'sm\n',sm
   print 's2\n',sz.dot(sz)+sz+sm.dot(sp)
   print 's2\n',sz.dot(sz)-sz+sp.dot(sm)
