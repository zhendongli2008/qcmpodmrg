import numpy 
import scipy.linalg
import math
import itools

def det_init(N,K,iop=1):
#
# iop=0 identity
#    =1 random
#    =2 cyclic huckel
#
   if iop == 0:
      v = numpy.identity(K)
   elif iop == 1:
      h=numpy.random.uniform(-1,1,K*K)
      h=h.reshape(K,K)
      h=0.5*(h+h.T)
      e,v=scipy.linalg.eigh(h)
   elif iop == 2:
      h=numpy.zeros((K,K))
      t=-1.0 # hopping is negative
      for i in range(K):
         h[i,(i+1)%K]=t
	 h[i,(i-1)%K]=t
      e,v=scipy.linalg.eigh(h)
      print "Cyclic Hamiltonian:"
      print h
      print e
      print v
      #exit(1)
   elif iop == 3:
      h=numpy.zeros((K,K))
      t=-2.0 # hopping is negative
      for i in range(K):
         h[i,(i+1)%K]=t
	 h[i,(i-1)%K]=t
	 h[i,i]=-i/10.0
      e,v=scipy.linalg.eigh(h)
      print "Cyclic Hamiltonian:"
      print h
      print e
      print v
      #exit(1)
   elif iop == 4:
      h=numpy.random.uniform(-1,1,K*N)
      h=h.reshape(K,N)
      v=h 
   return v[:,:N]

def det_to_mps(v):
   [K,N]=v.shape
   #print K,N
   D=1
   # C order
   mps=[];
   mpsi=numpy.zeros((K,D))
   mpsi[:,0]=v[:,0]
   mps.append(mpsi)
   for i in range(1,N-1):
      mpsi=numpy.zeros((D,K,D))
      mpsi[0,:,0]=v[:,i]
      mps.append(mpsi)
   mpsi=numpy.zeros((D,K))
   mpsi[0,:]=v[:,N-1]
   mps.append(mpsi)
   return mps

# With this, we can do Monte-Carlo !!!
def det_fci(v):
   print '\n[det_fci]'
   s=v.shape
   K=s[0]
   N=s[1]
   for i in itools.combinations(range(K),N):
      matrix=v[i,:]
      coeff=numpy.linalg.det(matrix)/math.sqrt(math.factorial(N))
      print i,coeff

def det_check(civec,nsorb,nelec,dets,iprt):
   if iprt>0: print '[det_check]'
   det_core=dets[0]
   det_site=dets[1]
   rank=len(det_core)
   ic=0
   civec2=numpy.zeros(civec.size)
   for i in itools.combinations(range(nsorb),nelec):
      for r in range(rank):
	 matrix=det_site[r][:,i]
         civec2[ic]+=numpy.linalg.det(matrix)*det_core[r]
      ic=ic+1
   diff=numpy.linalg.norm(civec-civec2)
   if iprt>0: print "DIFF=",diff
   return diff

def det_cofactor(matrix,i,j):
   nr,nc=matrix.shape
   minor=matrix[numpy.array(range(i)+range(i+1,nr))[:,numpy.newaxis],
		numpy.array(range(j)+range(j+1,nc))]
   return (-1)**(i+j)*numpy.linalg.det(minor)
