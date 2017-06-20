import h5py
import copy
import math
import numpy
import scipy.linalg
from tools import misc
from tools import jwtrans
from tools import itools
from tools import detlib
from tools import tensorRep
from tools import tensorDecomp
from tools import mpslib
import mps_class
import mpo_qphys

#
# A simple MPO class: support basic algebras of MPO.
#
class class_mpo:
   #
   # MPO: {site[l,r,nu,nd]}
   #
   def __init__(self,k,sites=None,qnums=None):
      self.nsite=k
      # Sites
      if sites is None:
         self.sites=[0]*k
      else:
	 self.sites = copy.deepcopy(sites)
      # Quantum numbers	 
      self.qnums = None
      if qnums is not None:
         self.qnums = copy.deepcopy(qnums)

   # Diagonal term 
   def diagonal(self):
      tmpo = class_mpo(self.nsite)
      for i in range(tmpo.nsite):
	 # l,r,u,d -> u,d,l,r     
	 tmp1 = self.sites[i].transpose(2,3,0,1).copy()
	 tmp2 = numpy.zeros(tmp1.shape)
	 nd = tmp1.shape[0]
	 tmp2[range(nd),range(nd)] = tmp1[range(nd),range(nd)]
         tmpo.sites[i] = tmp2.transpose(2,3,0,1).copy() 
      return tmpo

   def copy(self):
      tmpo = class_mpo(self.nsite)
      for i in range(tmpo.nsite):
         tmpo.sites[i] = self.sites[i].copy()
      return tmpo

   def fromRank1(self,t1):
      assert self.nsite == len(t1)
      for i in range(self.nsite):
         shape = t1[i].shape
	 self.sites[i] = numpy.zeros((1,1,shape[0],shape[1]))
	 self.sites[i][0,0] = t1[i]
      return 0

   def bdim(self):
      N=self.nsite
      dk=[]
      for i in range(1,N):
         s=self.sites[i].shape
         dk.append(s[0]) 
      return dk

   def prt(self,ifqnums=None):
      print "\nMPOinfo:"
      print " nsite = ",self.nsite	   
      for i in range(self.nsite):
         print " Site : ",i," Shape : ",self.sites[i].shape,\
	       " Val = ",numpy.linalg.norm(self.sites[i])
      if self.qnums is not None and ifqnums:
	 self.prtQnums()
      print "End of MPOinfo\n"
      return 0

   def prtQnums(self):
      print " quantum numbers:"
      for i in range(self.nsite):
	 print " Bond : ",i," bdim[i] =",len(self.qnums[i])
	 print " Qnums[i] =",self.qnums[i]
      return 0

   def toMat(self):
      N = self.nsite
      for i in range(N):
	 if i == 0:
	    t1 = self.sites[0]
         else:
            t2 = self.sites[i]
	    # t1[l,r,nu,nd]*t2[l,r,nu',nd']
	    tmp = numpy.einsum('abij,bckl->acikjl',t1,t2)
	    s = tmp.shape
	    t1 = tmp.reshape((s[0],s[1],s[2]*s[3],s[4]*s[5]))
      t1 = t1[0,0].copy()
      return t1

   # Transform to MPS representation by merging u,d indices
   def toMPS(self):
      N = self.nsite
      t1 = [0]*N
      # s(0)[l,r,u,d]
      s = self.sites[0].shape
      assert s[0] == 1
      t1[0] = self.sites[0].reshape(s[1],s[2]*s[3]).transpose(1,0).copy()
      # s(N-1)[l,r,u,d]
      s = self.sites[N-1].shape
      assert s[1] == 1
      t1[N-1] = self.sites[N-1].reshape(s[0],s[2]*s[3]).copy()
      # s(1,N-2)[l,r,u,d]
      for i in range(1,N-1):
         s = self.sites[i].shape
	 t1[i] = self.sites[i].reshape(s[0],s[1],s[2]*s[3]).transpose(0,2,1).copy()
      return t1	

   #
   # [Split] by reshaping from MPS => MPO
   #
   def compress(self,thresh=1.e-10,iprt=2):
      #return 0
      t1 = self.toMPS()
      t2 = copy.deepcopy(t1)
      norm = mpslib.mps_norm(t1)
      mpslib.mps_compress(t1,thresh=thresh,iprt=iprt)
      #
      #Is it possible to normalize it first?
      #mpslib.mps_normalize(t1)
      #
      rr = mpslib.mps_diff(t1,t2,iprt=0)
      if rr > 1.e-5:
	 print 'error: compression error in MPOcompression! rr=',rr
	 print 'HSnorm = ',self.HSnorm()
      # from t1 (list) -> MPO
      N = self.nsite
      # t1(0)[ud,r']->s(0)[l=1,r',u,d]
      t = t1[0].shape
      s = self.sites[0].shape
      assert s[0] == 1
      tmp = t1[0].reshape(s[2],s[3],t[1]).transpose(2,0,1)
      self.sites[0] = tmp.reshape(1,t[1],s[2],s[3]).copy()
      # t1(N-1)[l',ud]->s(N-1)[l',r=1,u,d]
      t = t1[N-1].shape
      s = self.sites[N-1].shape
      assert s[1] == 1
      self.sites[N-1] = t1[N-1].reshape(t[0],1,s[2],s[3]).copy()
      # t1(i)[l',ud,r']->s(i)[l',r',u,d]
      for i in range(1,N-1):
	 t = t1[i].shape
         s = self.sites[i].shape
	 self.sites[i] = t1[i].reshape(t[0],s[2],s[3],t[2]).transpose(0,3,1,2).copy()
      return 0

   #--------------------------------
   # Algebra: 
   #   1. MPO1+MPO2
   #   2. MPO1*MPO2
   #   3. MPO1*fac
   #   4. MPO*MPS
   #--------------------------------
   def T(self):
      N = self.nsite
      mpo = class_mpo(N)
      for i in range(N):
         mpo.sites[i] = self.sites[i].transpose(0,1,3,2).copy()
      return mpo

   def trace(self):
      N = self.nsite
      for i in range(N):
	 tmp = self.sites[i]
	 mat = numpy.einsum('lrmm->lr',tmp)
	 if i == 0:
	    tmat = mat
	 else:
	    tmat = numpy.einsum('lr,rt->lt',tmat,mat)
      assert tmat.shape == (1,1)
      return tmat[0,0]

   def HSnorm(self):
      mps1 = self.toMPS()
      norm = mpslib.mps_norm(mps1)
      # equivalent to
      #mpo = self.T().prod(self)
      #print 'tr',mpo.trace()
      return norm

   # 1. sum of two MPOs: site[1:l1,1:r1,nu,nd]
   #		         site[l1:l2,r1:r2,nu,nd]	
   #
   def add(self,other):
      assert self.nsite == other.nsite
      N    = self.nsite
      mpo  = class_mpo(N)
      for i in range(N):
	 # Direct sum structure
	 (l1,r1,u1,d1) = self.sites[i].shape 
	 (l2,r2,u2,d2) = other.sites[i].shape 
	 assert (u1 == u2 and d1 == d2)
	 l12 = l1+l2
	 r12 = r1+r2
	 tmp = numpy.zeros((l12,r12,u1,d1))
	 tmp[:l1,:r1] = self.sites[i]
	 tmp[l1:,r1:] = other.sites[i]
	 mpo.sites[i] = tmp.copy()
      #
      # This is a great simplification.  
      # Partial sum: a[j] = sum_i A[i,j] 
      #
      # Reform the boundary term-1
      tmp = numpy.einsum('ijkl->jkl',mpo.sites[0])
      s = tmp.shape
      mpo.sites[0] = tmp.reshape((1,s[0],s[1],s[2])).copy()
      # Reform the boundary term-N
      tmp = numpy.einsum('ijkl->ikl',mpo.sites[N-1])
      s = tmp.shape
      mpo.sites[N-1] = tmp.reshape((s[0],1,s[1],s[2])).copy()
      return mpo

   #
   # 2. MPO1*MPO2: a[l1,r1,u1,d1]*b[l2,r2,u2,d2]=>ab[l1*l2,r1*r2,u1,d2]
   #
   def prod(self,other):
      assert self.nsite == other.nsite
      N    = self.nsite
      mpo  = class_mpo(N)
      for i in range(N):
	 tmp1 = self.sites[i].copy()
	 tmp2 = other.sites[i].copy()
	 #tmp = numpy.einsum('abxy,ijyz->aibjxz',self.sites[i],other.sites[i])
	 # ijyz->yijz
	 tmp2 = tmp2.transpose(2,0,1,3).copy()
	 # abxy,yijz -> abxijz -> aibjxz 
	 tmp = numpy.tensordot(tmp1,tmp2,axes=([3],[0]))
         tmp = tmp.transpose(0,3,1,4,2,5)
	 s = tmp.shape
  	 mpo.sites[i] = tmp.reshape((s[0]*s[1],s[2]*s[3],s[4],s[5])).copy()
      return mpo

   #
   # 3. MPO1*fac ? - numerical stable version ?
   #
   def mul(self,fac):
      #self.sites[0] = fac*self.sites[0]
      # Randomly distribute the factor
      #k = numpy.random.randint(self.nsite)
      #self.sites[k] = fac*self.sites[k]
      # What about equally distribution???
      sfac = math.pow(abs(fac),1/float(self.nsite))
      sign = 1.0
      if fac<0.: sign = -1.0
      for k in range(self.nsite):
         self.sites[k] = sfac*self.sites[k]
      self.sites[0] = sign*self.sites[0]
      return 0

   #
   # 4. MPO*MPS
   #
   #@profile
   def dot(self,mps):
      assert self.nsite == len(mps)
      N    = self.nsite
      mpslib.mps_mps2rank3(0,mps)
      tmps = []
      for i in range(N):
 	 # MPO[l,r,u,d]*mps[l',d,r']=MPS(ll',u,rr')
	 (l1,r1,u1,d1) = self.sites[i].shape 
	 (l2,d2,r2) = mps[i].shape 
	 l12 = l1*l2
	 r12 = r1*r2
	 #tmp = numpy.einsum('lrud,adb->laurb',self.sites[i],mps[i])
	 tmp = numpy.tensordot(self.sites[i],mps[i],axes=([3],[1])) # lruab
	 tmp = tmp.transpose(0,3,2,1,4) # lruab->laurb
	 tmps.append(tmp.reshape((l12,u1,r12)))
      mpslib.mps_mps2rank3(1,tmps)
      mpslib.mps_mps2rank3(1,mps)
      return tmps

   def dotMPS(self,mps,debug=False):
      if debug: print '\n[mpo_class.dotMPS]'	   
      assert self.nsite == mps.nsite
      mps2 = mps.torank2()
      tmps = self.dot(mps2)
      #
      # Quantum number operations if Qnums exist
      #
      qnums = None
      if self.qnums is not None and mps.qnums is not None:
	 if debug: print 'Merge quantum numbers:'
	 q1 = self.qnums
	 q2 = mps.qnums
	 b1 = self.bdim()
	 b2 = mps.bdim()
	 qnums = [0]*self.nsite
	 for ibond in range(self.nsite):
	    q1i = self.qnums[ibond]
	    q2i = mps.qnums[ibond]
	    q12 = mpo_qphys.dpt(q1i,q2i)
	    if debug: print 'q12=',q12
	    # Some compressions are required to remove negative 
	    # and large q12 that exceed the allowed qnums.
	    qnums[ibond] = copy.deepcopy(q12)
      # Product new MPS
      mps2 = mps_class.class_mps(self.nsite,sites=tmps,qphys=mps.qphys,qnums=qnums)
      return mps2

   #--------------------------------
   # IO
   #--------------------------------
   def dump(self,fname='mpo.h5'):
      print '\n[mpo_class.dump]: fname = ',fname
      f = h5py.File(fname, "w")
      f.create_dataset("nsite",data=self.nsite)
      f.create_dataset("hmpo_bdim",data=self.bdim())
      for isite in range(self.nsite):
         f.create_dataset("hmpo_"+str(isite),data=self.sites[isite])
      f.close()
      return 0

   def merge(self,partition):
      print '\n[mps_class.merge]'
      print ' Partition =',partition
      nsite = len(partition)
      if nsite == 0:
 	 print 'error: empty partition!'
	 exit()
      # Start construction
      tmpo = [0]*nsite
      qnums = [0]*nsite
      for idx,item in enumerate(partition):
         for jdx,isite in enumerate(item):
            if jdx == 0:
               cop = self.sites[isite].copy()
            else:
               tmp = self.sites[isite].copy()
               #tmp = numpy.einsum('abij,bckl->acikjl',cop,tmp)
               tmp = numpy.tensordot(cop,tmp,axes=([1],[0]))
	       # abij,bckl->aijckl->acikjl
	       tmp = tmp.transpose(0,3,1,4,2,5).copy()
               s = tmp.shape
               cop = tmp.reshape((s[0],s[1],s[2]*s[3],s[4]*s[5])).copy()
         tmpo[idx] = cop.copy()
	 # Taken the quantum numbers of the last site in each group: o---
	 if self.qnums is not None: 
	    qnums[idx] = self.qnums[item[-1]]
      if self.qnums is None: 
         tmpo = class_mpo(nsite,sites=tmpo)
      else:
         tmpo = class_mpo(nsite,sites=tmpo,qnums=qnums)
      return tmpo


##########################
# Auxilliary Subroutines #
##########################
def mpo_r1(k,p,otype):
   r = numpy.arange(k)
   mpo = [jwtrans.idn]*k
   # sign factor
   for r in range(p):
      mpo[r] = jwtrans.sgn
   # matrix representation   
   if otype == 1:
      mpo[p] = jwtrans.cre	   
   elif otype == 0:
      mpo[p] = jwtrans.ann 
   return mpo

def mpo_r1mul(t1,t2):
   k1 = len(t1)
   k2 = len(t2)
   assert k1 == k2
   mpo = [0]*k1
   for k in range(k1):
      mpo[k] = numpy.dot(t1[k],t2[k])
   return mpo

def mpo_r1mat(t1):
   'matrix representation of rank-1 mpo'
   k1 = len(t1)
   for k in range(k1):
      if k == 0:
	 mat = t1[k] 
      else:
	 mat = numpy.kron(mat,t1[k])
   return mat

# Diff <P-Q|P-Q>
def mpo_diff(mpo1,mpo2,iprt=0):
   if iprt != 0:
      print '[mpo_diff]'
      print 'mpo1.bdim=',mpo1.bdim()
      print 'mpo2.bdim=',mpo2.bdim()
   mps1 = mpo1.toMPS()
   mps2 = mpo2.toMPS()
   rr = mpslib.mps_diff(mps1,mps2,iprt)
   return rr

# LOCAL term
def u1mpo(k,i,j,arg,iop=0,debug=False):
   u1mpo = class_mpo(k)
   # ---i---j---
   for m in range(i):
      tmp = numpy.identity(2)
      tmp = tmp.reshape(1,1,2,2).copy()
      u1mpo.sites[m] = tmp.copy()
   for m in range(j+1,k):
      tmp = numpy.identity(2)
      tmp = tmp.reshape(1,1,2,2).copy()
      u1mpo.sites[m] = tmp.copy()
   # i-j: id[a,b,u,d] = d[a,b]d[u,d]
   tmp = numpy.zeros((4,4,2,2))
   for m in range(4):
      tmp[m,m] = numpy.identity(2)
   for m in range(i+1,j):
      u1mpo.sites[m] = tmp.copy()
   # 1. umat[(u,d),a]	   
   if iop == 0:
      tmp = u1mat(arg)
      if debug: print '\nucore[i,j]= (%d,%d)\n'%(i,j),tmp
   else:
      # A kind of random unitary	   
      tmp = numpy.random.uniform(-1,1,(4,4))
      q,r = scipy.linalg.qr(tmp)
      tmp = q
   tmp = tmp.reshape(1,2,2,4).copy()
   tmp = tmp.transpose(0,3,1,2).copy()
   u1mpo.sites[i] = tmp.copy()
   # 2. imat[a,(u,d)]
   tmp = numpy.identity(4).reshape(4,2,2,1).copy()
   tmp = tmp.transpose(0,3,1,2).copy()
   u1mpo.sites[j] = tmp.copy()
   if debug:
      print u1mpo.sites
      print k
      print u1mpo.sites[0].shape
      print u1mpo.sites[1].shape
      print u1mpo.toMat()
   return u1mpo

#
# Auxiliary subroutines
#
def u1mat(arg):
#-------------------------------
# Tensor version 
# {|0a0b> |0a1b> |1a0b> |1a1b>}
#   [ 1   0   0   0 ]
#   [ 0   c   s   0 ] 
#   [ 0  -s   c   0 ]
#   [ 0   0   0   1 ]
#-------------------------------
   u = numpy.zeros((4,4))
   u[0,0] = 1.0
   u[3,3] = 1.0
   c = math.cos(arg)
   s = math.sin(arg)
   u[1,1] = c
   u[1,2] = s#-s
   u[2,1] = -s#s
   u[2,2] = c
   u = u.reshape((2,2,2,2))
   u = u.transpose(0,2,1,3)
   u = u.reshape((4,4))
   return u

def ukmat(k,i,j,arg):
#-------------------------------
# Matrix version
#   [ c -s ] 
#   [ s  c ]
#-------------------------------
   u = numpy.identity(k)
   c = math.cos(arg)
   s = math.sin(arg)
   u[i,i] = c
   u[i,j] = -s
   u[j,i] = s
   u[j,j] = c
   return u

#
# Generate MPO representation of Hamiltonian
#
def genHmpo(h1e,h2e,thresh=1.e-10,iop=0):
   op1 = genHmpo1(h1e,thresh)
   if iop == 0:
      op2 = genHmpo2(h2e,thresh)
      op1 = op1.add(op2)
   op1.compress()
   return op1

def genHmpo1(h1e,thresh=1.e-10):
   # h[i,j]ai^+ aj
   nb = h1e.shape[0]
   idx = 0
   op1 = None
   for i in range(nb):
      for j in range(nb):
	 if abs(h1e[i,j])<thresh: continue
	 #print '(i,j)=',i,j,h1e[i,j]
         ci = mpo_r1(nb,i,1)
         aj = mpo_r1(nb,j,0)
         Aij = mpo_r1mul(ci,aj)
         op = class_mpo(nb)
         op.fromRank1(Aij)
         op.mul(h1e[i,j])
         idx += 1
         if idx == 1:
            op1 = op.copy()
         else:
            op1 = op1.add(op)		
      if op1 is not None:
	 op1.compress()
         #print 'bdim',j,op1.bdim()
   if idx == 0:
      print 'warning: no mpo generated in genHmpo1 with thresh=',thresh
      op1 = nullmpo(nb)
   return op1 

def genHmpo2(h2e,thresh=1.e-10):
   # v[i,j,k,l]ai^+ aj^+ ak al (i<j,k<l)
   nb = h2e.shape[0]
   idx = 0
   op1 = None
   for j in range(nb):
      for i in range(j+1):
         for l in range(nb):
   	    for k in range(l+1):
	       if abs(h2e[i,j,k,l])<thresh: continue
   	       #print 'idx=',idx,'(i,j,k,l)=',i,j,k,l,h2e[i,j,k,l]
               ci = mpo_r1(nb,i,1)
               cj = mpo_r1(nb,j,1)
               ak = mpo_r1(nb,k,0)
               al = mpo_r1(nb,l,0)
               Aij = mpo_r1mul(ci,cj)
   	       Akl = mpo_r1mul(ak,al)
   	       Aijkl = mpo_r1mul(Aij,Akl)
   	       op = class_mpo(nb)
   	       op.fromRank1(Aijkl)
   	       op.mul(h2e[i,j,k,l])
               idx += 1
   	       if idx == 1:
   	          op1 = op.copy()
   	       else:
    	          op1 = op1.add(op)		
      # j
      if op1 is not None:
         op1.compress()
         #print 'bdim',j,op1.bdim()
   if idx == 0:
      print 'error: no mpo generated in genHmpo2 with thresh=',thresh
      op1 = nullmpo(nb)
   return op1

#
# Some numerical test
#
def testRandomUrot(nb,i,j,arg,op1,thresh=1.e-10):
   hmpo = op1
   bdim0 = hmpo.bdim()
   u1 = u1mpo(nb,i,j,arg,debug=True)
   print 'u1mpo',u1.bdim()
   print u1.HSnorm()
   u1t  = u1.T()
   u1th = u1t.prod(hmpo)
   #u1th.compress()
   bdimt = u1th.bdim()
   uthu = u1th.prod(u1)
   bdim1 = uthu.bdim()
   uthu.compress(thresh)
   bdim2 = uthu.bdim()
   # Debug
   #mat2=
   #[[  0.           0.           0.           0.        ]
   # [  0.         -10.21524939  -3.69542687   0.        ]
   # [  0.          -3.69542687   5.06794118   0.        ]
   # [  0.           0.           0.          -5.14730821]]
   u1mat = u1.toMat()
   opmat = op1.toMat()
   trans = reduce(numpy.dot,(u1mat.T,opmat,u1mat))
   print 'trans\n',trans
   return uthu,bdim0,bdimt,bdim1,bdim2

def printUrot(i,j,arg,res): 
   bdim0,bdimt,bdim1,bdim2 = res[1:]
   print
   print 'i,j',i,j,'arg=',arg,'bdims=',bdim0[i],bdim0[i]*16
   print 'bdim0(   h  ) =',bdim0
   print 'bdimt(ut*h  ) =',bdimt
   print 'bdim1(ut*h*u) =',bdim1
   print 'bdim2(cmprss) =',bdim2
   return 0

#
# BCH expansion exp(K)*H*exp(-K)
#
def BCHtransform(hmpo,kmpo):
   thmpo = copy.deepcopy(hmpo)
   norm = thmpo.HSnorm()
   print thmpo.prt()
   print 'i,bdim1=',0,thmpo.bdim()
   print 'i,norm1=',0,norm
   # [k,H] = k*H - H*k
   seed = copy.deepcopy(hmpo)
   for i in range(1,100):
      kh = kmpo.prod(seed)
      hk = seed.prod(kmpo)
      hk.mul(-1.0)
      seed = kh.add(hk)
      seed.mul(1/float(i))
      seed.compress()
      norm = seed.HSnorm()
      thmpo = thmpo.add(seed)
      thmpo.compress()
      print 'i,bdim1=',i,thmpo.bdim(),norm,thmpo.HSnorm()
   return thmpo

#
# Kij = (kij ai^+ aj) - (kij* aj^+ ai)
# U = Exp(-K) 
#
def k1mpo(k,i,j,arg,thresh=1.e-10):
   kij = numpy.zeros((k,k))
   kij[i,j] = arg
   kij[j,i] = -arg
   kmpo = genHmpo1(kij,thresh)
   return kmpo

def s2projmpo(k,sval,sz,npoints=100):
   # Integrand 
   def rfun(phi):
      expm = numpy.zeros((4,4))
      expm[0,0] = 1.0
      c = math.cos(0.5*phi)
      s = math.sin(0.5*phi)
      expm[1,1] = c 
      expm[1,2] = -s 
      expm[2,1] = s 
      expm[2,2] = c 
      expm[3,3] = 1.0
      return expm
   # Weights
   from tools import smalld
   def wfun(phi,sval,sz):
      wt = (2.*sval+1.)/2.0*math.sin(phi)*smalld.value(sval,sz,sz,phi)
      return wt
   # To use Simpson rules, we use even no. of grids
   # https://en.wikipedia.org/wiki/Simpson%27s_rule
   assert npoints%2 == 0
   npts = npoints+1
   wts = numpy.zeros(npts)
   ###Trapezoidal rule
   #wts[:] = 2.0
   #wts[0] = 1.0
   #wts[-1] = 1.0
   #wts = wts*numpy.pi/(2.0*npoints)
   ###Simpson rule
   wts[0::2] = 2.0
   wts[1::2] = 4.0
   wts[0]  = 1.0
   wts[-1] = 1.0
   wts = wts*numpy.pi/(3.0*npoints)
   # Sites
   xdata = numpy.linspace(0,numpy.pi,num=npts)
   wts = wts*numpy.array(map(lambda x:wfun(x,sval,sz),xdata))
   expm1 = numpy.array(map(lambda x:rfun(x),xdata))
   expm0 = numpy.array([wts[i]*expm1[i] for i in range(npts)])
   # MPO
   op = class_mpo(k)
   op.sites[0] = numpy.zeros((1,npts,4,4))
   op.sites[0][0] = expm0
   for isite in range(1,k-1):
      op.sites[isite] = numpy.zeros((npts,npts,4,4))
      op.sites[isite][range(npts),range(npts)] = expm1
   op.sites[k-1] = numpy.zeros((npts,1,4,4))
   op.sites[k-1][range(npts),0] = expm1
   return op

def occmpo(k):
# <n1',n2',...,nk'|N|n1,n2,...,nk> factorized into
# a sum product of terms like {<n1'|n|n1>*I2*...*Ik}.
   kmpo = class_mpo(k)
   nocc = numpy.zeros((2,2))
   nocc[1,1] = 1.0
   iden = numpy.identity(2)
   # First term
   tmp = numpy.zeros((1,2,2,2))
   tmp[0,0] = nocc
   tmp[0,1] = iden
   kmpo.sites[0] = tmp.copy()
   # Last term
   tmp = numpy.zeros((2,1,2,2))
   tmp[0,0] = iden
   tmp[1,0] = nocc
   kmpo.sites[k-1] = tmp.copy()
   # Middle
   for i in range(1,k-1):
      tmp = numpy.zeros((2,2,2,2))
      tmp[0,0] = iden
      tmp[1,1] = iden
      tmp[1,0] = nocc
      kmpo.sites[i] = tmp.copy()
   return kmpo

def nullmpo(k):
# <n1',n2',...,nk'|I|n1,n2,...,nk> factorized into
# a product of <n1'|n1>*<n2'|n2>*...*<nk'|nk>.
   kmpo = class_mpo(k)
   for i in range(k):
      tmp = numpy.zeros((2,2))
      tmp = tmp.reshape(1,1,2,2).copy()
      kmpo.sites[i] = tmp.copy()
   return kmpo

def idenmpo(k):
# <n1',n2',...,nk'|I|n1,n2,...,nk> factorized into
# a product of <n1'|n1>*<n2'|n2>*...*<nk'|nk>.
   kmpo = class_mpo(k)
   for i in range(k):
      tmp = numpy.identity(2)
      tmp = tmp.reshape(1,1,2,2).copy()
      kmpo.sites[i] = tmp.copy()
   return kmpo

# U = Exp(-K)
def genUmpo(k,i,j,arg,targ=1.e-10,tcomp=1.e-10,debug=False):
   umpo = idenmpo(k)
   if abs(arg)>targ:
      iconv = False
      kmpo = k1mpo(k,i,j,arg,thresh=targ)
      smpo = copy.deepcopy(umpo)
      for i in range(1,100):
         # (-k)^n/n!
         smpo = smpo.prod(kmpo)
         smpo.mul(-1.0/float(i))
         smpo.compress(thresh=tcomp)
         norm = smpo.HSnorm()
	 if norm < 1.e-12: iconv=True; break
         umpo = umpo.add(smpo)
         umpo.compress(thresh=tcomp)
	 if debug: print 'i,bdim1=',i,umpo.bdim(),norm,umpo.HSnorm()
	 #print smpo.toMat()
	 #print umpo.toMat()
      if iconv == False:
         print 'Umpo not converged!'
         exit()
      else:
         print 'Umpo generated with maxsteps=',i
      #umat = umpo.toMat()
      #print umat
      #print umat.T.dot(umat)
      #exit()
   return umpo

#
# Exact MPO expression U1mpoG = exp(-Kij)
#
def genU1mpo(k,i,j,arg,thresh_arg=1.e-10):
   assert i != j	
   umpo = idenmpo(k)
   if abs(arg)>thresh_arg:
      targ = -arg
      c = math.cos(targ)
      s = math.sin(targ)
      kii = c-1.0
      kjj = c-1.0
      kij = s
      kji = -s
      kijij = 2.0*(1.0-c)
      ci = mpo_r1(k,i,1)
      cj = mpo_r1(k,j,1)
      ai = mpo_r1(k,i,0)
      aj = mpo_r1(k,j,0)
      Aii = mpo_r1mul(ci,ai)
      Aij = mpo_r1mul(ci,aj)
      Aji = mpo_r1mul(cj,ai)
      Ajj = mpo_r1mul(cj,aj)
      op1 = class_mpo(k)
      op1.fromRank1(Aii)
      op1.mul(kii)
      op2 = class_mpo(k)
      op2.fromRank1(Aij)
      op2.mul(kij)
      op3 = class_mpo(k)
      op3.fromRank1(Aji)
      op3.mul(kji)
      op4 = class_mpo(k)
      op4.fromRank1(Ajj)
      op4.mul(kjj)
      nij = mpo_r1mul(Aii,Ajj)
      op5 = class_mpo(k)
      op5.fromRank1(nij)
      op5.mul(kijij)
      umpo = umpo.add(op1)
      umpo = umpo.add(op2)
      umpo = umpo.add(op3)
      umpo = umpo.add(op4)
      umpo = umpo.add(op5)
      bdim0 = umpo.bdim()
      umpo.compress()
   return umpo

def testOneBodyU():
   print '\n[testOneBodyU]'
   k = 20 # LARGE k = 30 -> THE NUMERICAL ERROR GROWS WITH k !!!
   i = 4
   j = 15
   # Swap gate for i <-> j.
   arg = 10 #numpy.pi/2
   umpo1 = genUmpo(k,i,j,arg,debug=True)
   umpo2 = genU1mpo(k,i,j,arg)
   rr = mpo_diff(umpo1,umpo2)
   umpo1.compress(iprt=1,thresh=1.e-8)
   print 'umpo1',umpo1.bdim()
   print 'umpo2',umpo2.bdim()
   print 'diff=',rr
   # The result should be:
   #umpo1 [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1]
   #umpo2 [1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1]
   #diff= 0.0
   return 0
   
#def transformMPS(mps):
#   n = len(mps)
#   tmps = copy.deepcopy(mps)
#   mpslib.mps_mps2rank3(0,tmps)
#   for i in range(n-1):
#      print tmps[i].shape
#      ta = tmps[i].copy()
#      tb = tmps[i+1].copy()
#      tab = numpy.einsum('lar,rbs->labs',ta,tb)
#      tab = tab.transpose(1,2,0,3)
#      s = tab.shape
#      tab = tab.reshape((s[0]*s[1],s[2]*s[3])).copy()
#      u,s,v = scipy.linalg.svd(tab)
#      print tab.shape
#      print u
#      print s
#      
#      'follow the compression program'
#      exit()
#
#   return tmps

def testAtA(nb=8):
   print '\n[testAtA]'
   # Integrals
   numpy.random.seed(2)
   h1e = 10*numpy.random.uniform(-1,1,nb**2).reshape(nb,nb)
   h1e = h1e+h1e.T
   h1e2= 10*numpy.random.uniform(-1,1,nb**2).reshape(nb,nb)
   h1e2= h1e2+h1e2.T
   h2e = 10*numpy.random.uniform(-1,1,nb**4).reshape(nb,nb,nb,nb)
   
   op1 = genHmpo1(h1e)
   bdim0 = op1.bdim()
   op2 = op1.T().prod(op1)
   bdim1 = op2.bdim()
   op2.compress()
   bdim2 = op2.bdim()
   print 'bdim0=',bdim0
   print 'bdim1=',bdim1
   print 'bdim2=',bdim2

   op1b = genHmpo1(h1e2)
   bdim0 = op1b.bdim()
   op2b = op1b.T().prod(op1b)
   bdim1 = op2b.bdim()
   op2b.compress()
   bdim2 = op2b.bdim()
   print
   print 'bdim0b=',bdim0
   print 'bdim1b=',bdim1
   print 'bdim2b=',bdim2

   op3 = op2.add(op2b).add(op2.T().add(op2b))
   bdim3 = op3.bdim()
   op3.compress()
   bdim3b= op3.bdim()
   print
   print 'bdim3 =',bdim3
   print 'bdim3b=',bdim3b

   op4 = genHmpo2(h2e)
   bdim4 = op4.bdim()
   print
   print 'bdim4=',bdim4
   #
   #No significant saving on bond dimensions
   #Better for parallelization? H-MPO O(K^5)? 
   #Reduced storage cost for very large K? 
   #
   #bdim0= [4, 6, 8, 10, 12, 10, 8, 6, 4]
   #bdim1= [16, 36, 64, 100, 144, 100, 64, 36, 16]
   #bdim2= [4, 16, 29, 49, 67, 46, 29, 16, 4]
   #
   #bdim0b= [4, 6, 8, 10, 12, 10, 8, 6, 4]
   #bdim1b= [16, 36, 64, 100, 144, 100, 64, 36, 16]
   #bdim2b= [4, 16, 29, 48, 67, 46, 29, 16, 4]
   #
   #bdim3 = [16, 64, 116, 194, 268, 184, 116, 64, 16]
   #bdim3b= [4, 16, 44, 73, 80, 65, 39, 16, 4]
   #
   #bdim4= [4, 16, 39, 64, 80, 50, 37, 16, 4]
   #
   return 0


def detToMPS(vmat1):
   print '\n[detToMPS]'
   nb,nelec = vmat1.shape
   civec2=numpy.zeros(misc.binomial(nb,nelec))
   for strAB in itools.combinations(range(nb),nelec):
      addr=tensorRep.str2addr_o1(nb,nelec,tensorRep.string2bit(strAB))
      civec2[addr]=numpy.linalg.det(vmat1[list(strAB),:])
   # Fock-space representation
   citensor=tensorRep.toONtensor(nb,nelec,civec2)
   # ToMPS format
   mps=tensorDecomp.toMPS(citensor,[2]*nb,1.e-14,plot=False)
   #
   # It is important to realize that in toONtensor, the ordering of 
   # orbitals is actually [k,...,2,1] while in our case, we need
   # [1,2,..,k]. So a reverse procedure is need to reverse the MPS.
   #
   mps = mpslib.mps_reverse(mps)
   bdim0=mpslib.mps_bdim(mps)
   mpslib.mps_compress(mps)
   bdim1=mpslib.mps_bdim(mps)
   return mps

#
# A brute-force version of 1RDM
#
def makeRDM1(mps):
   nb = len(mps)
   rdm1 = numpy.zeros((nb,nb))
   for i in range(nb):
      for j in range(nb):
	 print 'rdm[i,j] (i,j)=',i,j  
         ci = mpo_r1(nb,i,1)
         aj = mpo_r1(nb,j,0)
         Aij = mpo_r1mul(ci,aj)
         op = class_mpo(nb)
         op.fromRank1(Aij)
	 rdm1[i,j] = mpslib.mps_dot(mps,op.dot(mps))
   print 'Hermicity=',numpy.linalg.norm(rdm1-rdm1.T)
   return rdm1


def testDet(nb=12,nelec=6):
   import jacobi
   print '\n[testDet]'
   # Init   
   numpy.random.seed(2)
   h=numpy.random.uniform(-1,1,nb*nb)
   h=h.reshape(nb,nb)
   h=0.5*(h+h.T)
   e1,v1,givens = jacobi.jacobi(h,ifswap=False)
   vmat1 = v1[:,:nelec].copy()
   mps = detToMPS(vmat1)
   bdim0 = mpslib.mps_bdim(mps)
   rdm1 = makeRDM1(mps)
   diff = numpy.linalg.norm(rdm1-vmat1.dot(vmat1.T))
   print 'RDMdiff=',diff
   if diff > 1.e-10: exit()
   #-----------------------------------------------------#
   # Try to convert MPS(in AO basis) to MPS(in MO basis) #
   # by applying the unitary transformation gates.       #
   #-----------------------------------------------------#
   ngivens = len(givens)
   mps1 = copy.deepcopy(mps)
   for idx,rot in enumerate(givens):
      k,l,arg = rot
      if abs(arg)<1.e-10: continue
      print '>>> idx=',idx,'/',ngivens,' arg=',arg,math.cos(arg),math.sin(arg)
      umpo = genU1mpo(nb,k,l,-arg)
      mps1 = copy.deepcopy(umpo.dot(mps1))
      print 'mps.bdim = ',mpslib.mps_bdim(mps1)
      mpslib.mps_compress(mps1)
   # Print  
   bdim0 = mpslib.mps_bdim(mps)
   bdim1 = mpslib.mps_bdim(mps1)
   print 'bdim0',bdim0
   print 'bdim1',bdim1
   # Comparison
   dmps = mpslib.detmps(nb,nelec)
   rr = mpslib.mps_diff(dmps,mps1,iprt=1)
   print 'diff=',rr
   nmpo = occmpo(nb)
   nelec1 = mpslib.mps_dot(dmps,nmpo.dot(dmps))
   nelec2 = mpslib.mps_dot(mps1,nmpo.dot(mps1))
   print 'Check nelec:'
   print 'nelec =',nelec
   print 'nelec1=',nelec1
   print 'nelec2=',nelec2
   #
   # bdim0 [2, 4, 8, 16, 32, 64, 128, 256, 128, 64, 32, 16, 8, 4, 2]
   # bdim1 [2, 3, 3, 3, 3, 3, 4, 4, 4, 3, 2, 2, 2, 1, 1]
   # [mps_diff] 
   # rr=0.00000e+00 
   # pp=1.00000 
   # pq=1.00000 
   # qp=1.00000 
   # qq=1.00000 
   # diff= 0.0
   # Check nelec:
   # nelec = 8
   # nelec1= 8.0
   # nelec2= 8.00000000001
   # 
   return 0


#
# Test for local/non-local unitary rotations - NOT FINISHED YET!
# 
def testUdis(nb=10):
   print '\n[testUdis]'
   nb = 10
   numpy.random.seed(2)
   # Integrals
   h1e = 10*numpy.random.uniform(-1,1,nb**2).reshape(nb,nb)
   h1e = h1e+h1e.T
   h1e2= 10*numpy.random.uniform(-1,1,nb**2).reshape(nb,nb)
   h1e2= h1e2+h1e2.T
   t1e = 10*numpy.random.uniform(-1,1,nb**3).reshape(nb,nb,nb)
   h2e = 10*numpy.random.uniform(-1,1,nb**4).reshape(nb,nb,nb,nb)
   print '--------------------'
   print ' MPO representation '
   print '--------------------'
   #
   # hij*ai^+*aj
   #
   # n = 12
   # [4, 6, 8, 10, 12, 14, 12, 10, 8, 6, 4]
   # n = 14
   # [4, 6, 8, 10, 12, 14, 16, 14, 12, 10, 8, 6, 4]
   # n = 16
   # [4, 6, 8, 10, 12, 14, 16, 18, 16, 14, 12, 10, 8, 6, 4]
   # n = 18
   # [4, 6, 8, 10, 12, 14, 16, 18, 20, 18, 16, 14, 12, 10, 8, 6, 4]
   # n = 20
   # [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4]
   #
   # h[i,j]ai^+ aj
   print 'Test for simple one-body non-local rotations'
   op1 = genHmpo1(h1e)
   bdim0 = op1.bdim()
   print '\nbdim0=',bdim0
   #
   # Local: MPO transform
   #
   i1 = 0
   j1 = 1
   arg1 = numpy.random.uniform(-1,1)
   res1 = testRandomUrot(nb,i1,j1,arg1,op1)
   uthu = res1[0]
   print 'bdim0',op1.bdim()
   print 'uthu ',uthu.bdim()
   #
   # Transform integrals
   #
   umat = ukmat(nb,i1,j1,arg1)
   th1e = reduce(numpy.dot,(umat.T,h1e,umat))
   top1 = genHmpo1(th1e)
   print
   print 'bdim0',op1.bdim()
   print 'uthu ',uthu.bdim()
   print 'top1 ',top1.bdim()
   mat1=uthu.toMat()
   mat2=top1.toMat()
   e1,v1 =scipy.linalg.eigh(mat1)
   e2,v2 =scipy.linalg.eigh(mat2)
   print 'e1',e1
   print 'e2',e2
   print 'h1e =\n',h1e
   print 'umat=\n',umat
   print 'th1e=\n',th1e
   print 'mat1=\n',mat1
   print 'mat2=\n',mat2
   print 'mdif=',numpy.linalg.norm(mat1-mat2)
   diff = mpo_diff(uthu,top1)
   print 'diff=',diff
   print op1.HSnorm()
   print uthu.HSnorm()
   print top1.HSnorm()
   #
   # Nonlocal: MPO transform - WRONG !
   #
   i2 = 1
   j2 = 4
   arg2 = numpy.random.uniform(-1,1)
   res2 = testRandomUrot(nb,i2,j2,arg2,op1)
   uthu = res2[0]
   print 'bdim0',op1.bdim()
   print 'uthu ',uthu.bdim()
   #
   # Transform integrals
   #
   umat = ukmat(nb,i2,j2,arg2)
   th1e = reduce(numpy.dot,(umat.T,h1e,umat))
   top1 = genHmpo1(th1e)
   print
   print 'bdim0',op1.bdim()
   print 'uthu ',uthu.bdim()
   print 'top1 ',top1.bdim()
   mat1=uthu.toMat()
   mat2=top1.toMat()
   print 'mdif=',numpy.linalg.norm(mat1-mat2)
   diff = mpo_diff(uthu,top1)
   print 'diff=',diff
   #
   # Nonlocal: New scheme
   # KappaMPO=[1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
   #
   kmpo = k1mpo(nb,i2,j2,arg2)
   top2 = BCHtransform(op1,kmpo)
   tmat = top2.toMat()
   print mat2
   print tmat
   print 'th1e=',th1e
   print 'th1e=',h1e
   print 'mdif2=',numpy.linalg.norm(mat2-tmat)
   print top1.HSnorm()
   
   umpo = genUmpo(nb,i2,j2,arg2)
   print umpo.bdim()
   exit()

   umat = umpo.toMat()
  
   kmat = kmpo.toMat()
   print 'kmpo_toMat=',kmat
   print scipy.linalg.expm(-kmat)
   print 'umat=',umat

   diff = mpo_diff(top2,top1)
   print 'diff2=',diff

   tmp0 = umpo.T().prod(op1)
   top3 = tmp0.prod(umpo)
   diff = mpo_diff(top3,top1)
   print 'diff3=',diff

   print top1.bdim()
   print top2.bdim()
   print top3.bdim()
   top3.compress()
   exit()

   printUrot(i1,j1,arg1,res1)
   printUrot(i2,j2,arg2,res2)

   op1t = op1.T()
   op2 = op1.prod(op1t)
   bdim0 = op2.bdim()
   op2.compress()
   print
   print bdim0
   print op2.bdim()
   exit()
   return 0

def testBdim2(h1e,itype,jtype,thresh=1.e-10):
   # h[i,j]ai aj
   nb = h1e.shape[0]
   idx = 0
   op1 = None
   for i in range(nb):
      for j in range(nb):
	 if abs(h1e[i,j])<thresh: continue
         ci = mpo_r1(nb,i,itype)
         aj = mpo_r1(nb,j,jtype)
         Aij = mpo_r1mul(ci,aj)
         op = class_mpo(nb)
         op.fromRank1(Aij)
         op.mul(h1e[i,j])
         idx += 1
         if idx == 1:
            op1 = op.copy()
         else:
            op1 = op1.add(op)		
      if op1 is not None:
	 op1.compress()
   if idx == 0:
      print 'error: no mpo generated in testBdim2 with thresh=',thresh
      exit()
   return op1 

def testBdim3(t1e,itype,jtype,ktype,thresh=1.e-10):
   # v[i,j,k]ai^+ aj^+ ak^+
   nb = h1e.shape[0]
   idx = 0
   op1 = None
   for i in range(nb):
      for j in range(nb):
         for k in range(nb):
	    if abs(t1e[i,j,k])<thresh: continue
            print '(i,j,k)=',i,j,k,t1e[i,j,k]
            ci = mpo_r1(nb,i,itype)
            cj = mpo_r1(nb,j,jtype)
            ck = mpo_r1(nb,k,ktype)
            Aij = mpo_r1mul(ci,cj)
            Aijk = mpo_r1mul(Aij,ck)
            op = class_mpo(nb)
            op.fromRank1(Aijk)
            op.mul(t1e[i,j,k])
            idx += 1
            if idx == 1:
               op1 = op.copy()
            else:
               op1 = op1.add(op)		
      if op1 is not None:
	 op1.compress(thresh=1.e-8)
	 print 'bdim=',op1.bdim()
   if idx == 0:
      print 'error: no mpo generated in testBdim3 with thresh=',thresh
      exit()
   return op1

def testBdims(nb):
   nb = 16
   h1e = 10*numpy.random.uniform(-1,1,nb**2).reshape(nb,nb)
   t1e = 10*numpy.random.uniform(-1,1,nb**3).reshape(nb,nb,nb)
   op2 = testBdim2(h1e,1,0)
   op3 = testBdim3(t1e,1,0,0)
   print op2.bdim()
   print op3.bdim()
   #
   # nb = 16
   # [4, 6, 9, 11, 12, 14, 16, 18, 16, 14, 12, 10, 8, 6, 4]
   # [4, 11, 44, 176, 316, 366, 396, 392, 410, 416, 411, 256, 64, 16, 4]
   #
   return 0 

#====================================
# Direct construction of MPO for h1e
#====================================
def localMPO(hp,sz,op):
   #
   # sum h[p]*op with ferminoic symmetry
   # = h1 a1 * I2 * ... * I{k-1} * Ik
   # + h2 sz * a2 * ... * I{k-1} * Ik
   # + ...
   # + hk sa * sz * ... * sz     * ap
   #
   # = [a1,s1]^T* [ i2 0 ] * [i3 0 ] * [i4]
   #		  [ a2 s2] * [a3 s3] * [a4]
   #
   k = hp.shape[0]
   idn = numpy.identity(2)
   tmp = op.copy()
   tmp = tmp.reshape(1,1,2,2).copy()
   sites = [0]*k
   if k == 0:
      return []
   elif k == 1:
      # h1 a1 
      sites[0] = hp[0]*tmp
   else:
      # h1 a1*I2 + h2 sz*a2
      # t[0]
      tmpo = numpy.zeros((1,2,2,2))
      tmpo[0,0] = hp[0]*op
      tmpo[0,1] = sz
      sites[0] = tmpo.copy()
      # t[k-1]
      tmpo = numpy.zeros((2,1,2,2))
      tmpo[0,0] = idn
      tmpo[1,0] = hp[k-1]*op
      sites[k-1] = tmpo.copy()
      # middle
      for i in range(1,k-1):
         tmpo = numpy.zeros((2,2,2,2))
	 tmpo[0,0] = idn
	 tmpo[1,0] = hp[i]*op
	 tmpo[1,1] = sz
	 sites[i] = tmpo.copy()
   return sites 


def genHmpo1d(h1e,thresh=1.e-10,ifcomp=False):
   # h[i,j]ai^+ aj
   nb = h1e.shape[0]
   cre = jwtrans.cre.reshape(1,1,2,2).copy()
   ann = jwtrans.ann.reshape(1,1,2,2).copy()
   sgn = jwtrans.sgn.reshape(1,1,2,2).copy()
   nii = jwtrans.nii.reshape(1,1,2,2).copy()
   idn = jwtrans.idn.reshape(1,1,2,2).copy()
   ## Last site
   pre0 = [idn]*(nb-1)
   op1 = pre0 + [h1e[nb-1,nb-1]*nii]
   mpo = class_mpo(nb,op1)
   #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
   #>>> Although this save O(K) terms on the initial MPOs.
   #>>> It seems to be more stable to put diag terms into
   #>>> the diagonal of MPO such that compression is ok.
   # Sum of local terms
   #hp = numpy.diag(h1e).copy()
   #mpo = class_mpo(nb,localMPO(hp,jwtrans.idn,jwtrans.nii))
   #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
   for i in range(nb-1):
      # phase factor due to jwtrans	   
      pre0 = [idn]*i
      ## term-1: hpp np
      op1 = pre0 + [h1e[i,i]*nii] + [idn]*(nb-i-1)
      # term-2: ap^+ (hpq aq)
      vec = h1e[i,i+1:]
      tmp = jwtrans.cre.dot(jwtrans.sgn)
      tmp = tmp.reshape(1,1,2,2).copy()
      op2 = pre0 + [tmp] + localMPO(vec,jwtrans.sgn,jwtrans.ann)
      # term-2: -ap (hqp aq^+)
      vec = h1e[i+1:,i]
      tmp = -jwtrans.ann.dot(jwtrans.sgn)
      tmp = tmp.reshape(1,1,2,2).copy()
      op3 = pre0 + [tmp] + localMPO(vec,jwtrans.sgn,jwtrans.cre)
      t2 = class_mpo(nb,op2)
      t3 = class_mpo(nb,op3)
      t1 = class_mpo(nb,op1)
      t1 = t1.add(t2)
      t1 = t1.add(t3)
      #t1 = t2.add(t3)
      mpo = mpo.add(t1)
   
   # Print
   print 'BEFORE:',mpo.bdim()
   # Only compress once
   if ifcomp: mpo.compress()
   print 'AFTER:',mpo.bdim()
   return mpo 


def testGenHmpo1():
   nb = 20 # numerically unstable for nb = 30 in compressions!
   h1e = 10*numpy.random.uniform(-1,1,nb**2).reshape(nb,nb)
   #h1e = 10*numpy.random.uniform(-1,1,nb)
   #h1e = numpy.diag(h1e)
   mpo1 = genHmpo1d(h1e)
   mpo2 = genHmpo1(h1e)
   mpo_diff(mpo1,mpo2,iprt=1)
   #exit()
   #nb = 100
   #h1e = 10*numpy.random.uniform(-1,1,nb**2).reshape(nb,nb)
   #mpo1 = genHmpo1d(h1e)
   #
   #[58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94]
   #[4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4]
   #mpo1 [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4]
   #mpo2 [4, 6, 8, 11, 13, 15, 16, 20, 21, 22, 22, 20, 18, 17, 16, 13, 10, 7, 4]
   #
   # Almost optimal: K=100
   #
   #[298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 490, 492, 494]
   #[4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 73, 78, 83, 88, 93, 97, 90, 95, 100, 116, 118, 129, 136, 142, 147, 152, 158, 164, 359, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 256, 64, 16, 4]
   return 0

# TEST
if __name__ == '__main__':

   #testAtA()
   testGenHmpo1()
   #testUdis()
   #testDet()

