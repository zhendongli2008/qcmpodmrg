import numpy
import copy
from zmpo_dmrg.source.mpsmpo import mpo_class

#
# Approximation to exp(H*x)
# We associate beta factor to D and C
#
def polyH(hmpo,xfac=1.0):
   print '\n[polyH]'
   dtype = hmpo.sites[0].dtype
   N = hmpo.nsite
   expHx = [0]*N
   if hmpo.qnums == None: 
      ifqnum = False
      qnums = None 
   else:
      ifqnum = True 
      qnums = [0]*N
   #
   # site-0
   #
   tmp = hmpo.sites[0]
   diml,dimr,nc1,nc2=tmp.shape 
   assert diml == 1
   assert dimr > 2 # assuming C always exists
   i1 = tmp[0,0].copy()
   c1 = tmp[0,1:dimr-1].copy()
   d1 = tmp[0,dimr-1].copy()
   c1 = xfac*c1
   d1 = xfac*d1
   tmp2 = numpy.zeros((diml,dimr-1,nc1,nc2),dtype=dtype)
   tmp2[0,0] = i1+d1
   tmp2[0,1:] = c1.copy()
   expHx[0] = tmp2.copy()
   if ifqnum: qnums[0] = copy.deepcopy(hmpo.qnums[0][:-1]) 
   #
   # site-(N-1)
   #
   tmp = hmpo.sites[N-1].copy()
   diml,dimr,nc1,nc2=tmp.shape
   assert dimr == 1
   assert diml > 2
   d1 = tmp[0,0].copy()
   b1 = tmp[1:diml-1,0].copy()
   i1 = tmp[diml-1,0].copy()
   d1 = xfac*d1
   tmp2 = numpy.zeros((diml-1,dimr,nc1,nc2),dtype=dtype)
   tmp2[0,0] = i1+d1
   tmp2[1:,0] = b1.copy()
   expHx[N-1] = tmp2.copy()
   if ifqnum: qnums[N-1] = copy.deepcopy(hmpo.qnums[N-1]) 
   #
   # site-(1,N-2)
   #
   for isite in range(1,N-1):
      tmp = hmpo.sites[isite]
      diml,dimr,nc1,nc2=tmp.shape
      i1 = tmp[0,0].copy()
      c1 = tmp[0,1:dimr-1].copy()
      d1 = tmp[0,dimr-1].copy()
      a1 = tmp[1:diml-1,1:dimr-1].copy()
      b1 = tmp[1:diml-1,dimr-1].copy()
      i2 = tmp[diml-1,dimr-1].copy()
      c1 = xfac*c1
      d1 = xfac*d1
      assert numpy.linalg.norm(i1-i2)<1.e-10
      tmp2 = numpy.zeros((diml-1,dimr-1,nc1,nc2),dtype=dtype)
      tmp2[0,0] = i1+d1
      tmp2[0,1:] = c1.copy()
      tmp2[1:,0] = b1.copy()
      tmp2[1:,1:] = a1.copy()  
      expHx[isite]= tmp2.copy()
      if ifqnum: qnums[isite] = copy.deepcopy(hmpo.qnums[isite][:-1])
   empo = mpo_class.class_mpo(N,expHx,qnums)
   return empo

#
# Exact representation of 1+x*H
# We associate beta factor to D and C
#
def linearH(hmpo,xfac=1.0):
   print '\n[linearH]'
   dtype = hmpo.sites[0].dtype
   N = hmpo.nsite
   expHx = [0]*N
   if hmpo.qnums == None: 
      ifqnum = False
      qnums = None 
   else:
      ifqnum = True 
      qnums = copy.deepcopy(hmpo.qnums)
   #
   # site-0: [I, tC, tD+I]
   #
   tmp = hmpo.sites[0]
   diml,dimr,nc1,nc2=tmp.shape 
   assert diml == 1
   assert dimr > 2 # assuming C always exists
   i1 = tmp[0,0].copy()
   c1 = tmp[0,1:dimr-1].copy()
   d1 = tmp[0,dimr-1].copy()
   c1 = xfac*c1
   d1 = xfac*d1
   tmp2 = numpy.zeros_like(tmp,dtype=dtype)
   tmp2[0,0] = i1
   tmp2[0,1:dimr-1] = c1.copy()
   tmp2[0,dimr-1] = d1+i1
   expHx[0] = tmp2.copy()
   #
   # site-(N-1): [tD, B, I]^t
   #
   tmp = hmpo.sites[N-1].copy()
   diml,dimr,nc1,nc2=tmp.shape
   assert dimr == 1
   assert diml > 2
   d1 = tmp[0,0].copy()
   b1 = tmp[1:diml-1,0].copy()
   i1 = tmp[diml-1,0].copy()
   d1 = xfac*d1
   tmp2 = numpy.zeros_like(tmp,dtype=dtype)
   tmp2[0,0] = d1
   tmp2[1:diml-1,0] = b1
   tmp2[diml-1,0] = i1
   expHx[N-1] = tmp2.copy()
   #
   # site-(1,N-2):
   #
   #	[ I  tC   tD ]
   #    [ 0   A    B ]
   #    [ 0   0    I ]
   #
   for isite in range(1,N-1):
      tmp = hmpo.sites[isite]
      diml,dimr,nc1,nc2=tmp.shape
      i1 = tmp[0,0].copy()
      c1 = tmp[0,1:dimr-1].copy()
      d1 = tmp[0,dimr-1].copy()
      a1 = tmp[1:diml-1,1:dimr-1].copy()
      b1 = tmp[1:diml-1,dimr-1].copy()
      i2 = tmp[diml-1,dimr-1].copy()
      c1 = xfac*c1
      d1 = xfac*d1
      assert numpy.linalg.norm(i1-i2)<1.e-10
      tmp2 = numpy.zeros_like(tmp,dtype=dtype)
      tmp2[0,0] = i1
      tmp2[0,1:dimr-1] = c1
      tmp2[0,dimr-1] = d1
      tmp2[1:diml-1,1:dimr-1] = a1
      tmp2[1:diml-1,dimr-1] = b1
      tmp2[diml-1,dimr-1] = i2
      expHx[isite]= tmp2.copy()
   empo = mpo_class.class_mpo(N,expHx,qnums)
   return empo
