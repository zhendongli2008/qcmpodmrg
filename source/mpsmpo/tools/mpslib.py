# 
# def detmps(k,n):
# def mps_mps2rank3(iop,mps):
# def mps_check(tensor,shape,mps):
# def mps_print(mps):
# def mps_bdim(mps):
# def mps_dim(mps):
# def mps_size(mps):
# def mps_pdim(mps):
# def mps_exam(mps,thresh=1.e-14):
# def mps_prod(mps):
# def mps_compress(mps,iop=0,thresh=1.e-10,iprt=2):
# def mps_svd_cut(mat,thresh,D):
# def mps_svd_lupdate(iop,isite,mps,thresh,D,iprt=2):
# def mps_svd_rupdate(iop,isite,mps,thresh,D):
# def mps_leftSVD(mps,thresh,D,iprt=0):
# def mps_reverse(mps):
# def mps_rightSVD(mps,thresh,D,iprt=1):
# def mps_leftcanon(mps):
# def mps_rightcanon(mps):
# def mps_dot(mps1,mps2):
# def mps_norm(mps):
# def mps_scale(mps,alpha):
# def mps_normalize(mps):
# def mps_diff(mps1,mps2,iprt=0):
# def mps_random0(nsite,phys,D):
# def mps_random(mps,D,iop=0):
# def mps_ext(mps,Dincr):
# def mps_oneSite(mps,thresh):
# def mps_hstack(mps):
# 

#
# Tensor-Chain (MPS) decomposition:
#
#    T[p1,p2,p3] => A1[p1,a1]A2[a1,p2,a2]A3[a3,p3]
#
import numpy 
import scipy.linalg 
import math
import copy

def mps_conj(mps):
   nsite = len(mps)
   cmps = [0]*nsite
   for i in range(nsite):
      cmps[i] = numpy.conj(mps[i])
   return cmps

# |1> => |alpha> and |0> => |beta >.
def detmps(k,n):
   dmps = []
   for i in range(n):
      tmp = numpy.zeros((1,2,1))
      tmp[0,1,0] = 1.0
      dmps.append(tmp)
   for i in range(n,k):
      tmp = numpy.zeros((1,2,1))
      tmp[0,0,0] = 1.0
      dmps.append(tmp)
   mps_mps2rank3(1,dmps)
   return dmps

def mps_mps2rank3(iop,mps):
   # mps format -> mps[a0,p1,a1]
   N=len(mps)
   if iop == 0:
      # first
      shape=mps[0].shape
      mps[0]=mps[0].reshape(1,shape[0],shape[1])
      # last
      shape=mps[N-1].shape
      mps[N-1]=mps[N-1].reshape(shape[0],shape[1],1)
   elif iop == 1:
      # first
      shape=mps[0].shape
      mps[0]=mps[0].reshape(shape[1],shape[2])
      # last
      shape=mps[N-1].shape
      mps[N-1]=mps[N-1].reshape(shape[0],shape[1])
   return 0

def mps_check(tensor,shape,mps):
   # recover
   tensor1=mps_prod(mps)
   tensor2=tensor.copy().reshape(shape)
   diff=numpy.linalg.norm(tensor1-tensor2)
   print "DIFF=",diff
   return diff
   
def mps_print(mps):
   print "MPS="	
   N=len(mps)
   # test MPS
   for i in range(N):
      print "Site : ",i," Shape : ",mps[i].shape
      print mps[i]
 
def mps_bdim(mps):
   N=len(mps)
   dk=[]
   for i in range(1,N):
      s=mps[i].shape
      dk.append(s[0])
   return dk

def mps_dim(mps):
   N=len(mps)
   dk=[]
   for i in range(N):
      s=mps[i].shape
      dk.append(s)
   return dk

def mps_size(mps):
   s=mps_dim(mps)
   print 'mps_dim =',s
   s=map(numpy.prod, s)
   print 'mps_size =',s
   s=sum(s)
   print 'mps_size_tot =',s
   return s

def mps_pdim(mps):
   N=len(mps)
   pk=[]
   pk.append(mps[0].shape[0])
   for i in range(1,N-1):
      s=mps[i].shape
      pk.append(s[1])
   pk.append(mps[N-1].shape[1])
   return pk

# Compress the MPS into a big CI-coefficient tensor: A[1,1,p1p2...pN]
def mps_exam(mps,thresh=1.e-14):
   print "\n[mps_exam]"
   final=mps_prod(mps)
   final[abs(final)<thresh]=0.0
   print "NONZERO ENTRIES: "
   idx=numpy.argwhere(abs(final)>1.e-6)
   # this should be N! in case of N=K.
   print "No. of nonzero terms = ",len(idx) 
   print final[idx]

def mps_prod(mps):
   print "\n[mps_prod]: form full T[n1,n2,...,nk]"
   pindx=mps_pdim(mps)
   N=len(mps)
   tmps=mps[0].copy()
   # pack all site tensors
   for i in range(1,N-1):
      tmps2=numpy.einsum("Ij,jKl->IKl",tmps,mps[i])
      s=tmps2.shape
      tmps=tmps2.reshape((s[0]*s[1],s[2])).copy()
   tmps2=numpy.einsum("Ij,jK->IK",tmps,mps[N-1])
   s=tmps2.shape
   tensor=tmps2.reshape((s[0]*s[1])).reshape(pindx)
   return tensor

# In-place SVD comression
def mps_icompress(mps,thresh=1.e-10,iprt=2):
   if iprt != 2: print "\n[mps_icompress]"
   D=-1
   bdim0 = mps_bdim(mps)
   if iprt != 2: print "Bdim0=",bdim0
   # Left SVD
   mps_leftSVD(mps,thresh,D,iprt)
   bdim1 = mps_bdim(mps)
   if iprt != 2: print "Bdim1=",bdim1
   # Right SVD
   mps_rightSVD(mps,thresh,D,iprt)
   bdim2 = mps_bdim(mps)
   if iprt != 2: print "Bdim2=",bdim2
   return 0

# In-place comression
def mps_compress(mps,iop=0,thresh=1.e-10,iprt=2):
   if iprt != 2: print "\n[mps_compress]"
   #
   # save the initial one
   #
   mps0 = copy.deepcopy(mps)
#   norm = mps_norm(mps)
#   mps_scale(mps,1.0/norm)
   D=-1
   bdim0 = mps_bdim(mps)
   if iprt != 2: print "Bdim0=",bdim0
   #
   # Left SVD
   #
   mps_leftSVD(mps,thresh,D,iprt)
   bdim1 = mps_bdim(mps)
   if iprt != 2: print "Bdim1=",bdim1
   #
   # Right SVD
   #
   mps_rightSVD(mps,thresh,D,iprt)
   bdim2 = mps_bdim(mps)
   if iprt != 2: print "Bdim2=",bdim2
   #
   # Variational fit
   #
   if iop != 0: 
      mps_oneSite(mps,thresh)
      print "Bdim3=",mps_bdim(mps)
   rr = mps_diff(mps,mps0)
   if rr > 1.e-6:
      print 'WARNING: compression is not good enough! rr=',rr
      mps_diff(mps0,mps,iprt=1)
      exit()
   return 0


def mps_svd_cut(mat,thresh,D):
   if len(mat.shape) != 2:
      print "NOT A MATRIX in MPS_SVD_CUT !",mat.shape
      exit(1) 
   d1, d2 = mat.shape
   #------------------------------------
   try:
      u, sig, v = scipy.linalg.svd(mat, full_matrices=False)
   except ValueError:   
      u, sig, v = numpy.linalg.svd(mat, full_matrices=False)
   #------------------------------------
   # decide the final dimension of output according to thresh & D:
   r=len(sig)
   for i in range(r):
      if(sig[i]<thresh*1.01):
         r=i
         break
   # bond dimension at least = 1
   if r==0: r=1
   # final bond dimension:
   if D>0 :
      bdim=min(r,D) # D - use more flexible bond dimension
      rkep=min(r,D)
   else:
      bdim=r
      rkep=r
   s2=numpy.zeros((bdim))
   u2=numpy.zeros((d1,bdim),dtype = mat.dtype)
   v2=numpy.zeros((bdim,d2),dtype = mat.dtype)
   s2[:rkep]=sig[:rkep]
   u2[:,:rkep]=u[:,:rkep]
   v2[:rkep,:]=v[:rkep,:]
   return u2,s2,v2

# Update A[i]A[i+1]=>U[i](sVtA[i+1])
def mps_svd_lupdate(iop,isite,mps,thresh,D,iprt=2):
   N=len(mps)
   if isite == 0:
      mat=mps[isite]
   else:
      s=mps[isite].shape
      d1=s[0]*s[1]
      d2=s[2]
      mat=mps[isite].reshape((d1,d2))
   # SVD 
   u2,s2,v2=mps_svd_cut(mat,thresh,D)
   if iprt != 2:
      print 
      print ' s4/sigs =',math.pow(numpy.sum(s2**4),0.25),' len=',len(s2),'\n',s2
   # Update
   bdim=len(s2)
   if isite == 0:
      mps[isite]=u2.copy()
   else:	   
      mps[isite]=u2.reshape((s[0],s[1],bdim)).copy()
   # also update A[i+1]
   if iop:
      v2=numpy.diag(s2).dot(v2)
      if isite+1 == N-1:
         #mps[isite+1]=numpy.einsum("ij,jK->iK",v2,mps[isite+1])
         mps[isite+1]=numpy.tensordot(v2,mps[isite+1],axes=([1],[0]))
      else:
         #mps[isite+1]=numpy.einsum("ij,jKl->iKl",v2,mps[isite+1])
         mps[isite+1]=numpy.tensordot(v2,mps[isite+1],axes=([1],[0]))
   return 0

# Update A[i-1]A[i]=>(A[i-1]U[i]s)Vt
def mps_svd_rupdate(iop,isite,mps,thresh,D):
   N=len(mps)
   if isite == N-1:
      mat=mps[isite]
   else:	   
      s=mps[isite].shape
      d1=s[0]
      d2=s[1]*s[2]
      mat=mps[isite].reshape((d1,d2))
   # SVD 
   u2,s2,v2=mps_svd_cut(mat,thresh,D)
   # Update
   bdim=len(s2)
   if isite == N-1:
      mps[isite]=v2.copy()
   else:
      mps[isite]=v2.reshape((bdim,s[1],s[2])).copy()
   # also update A[i+1]
   if iop:
      u2=u2.dot(numpy.diag(s2))
      if isite-1 == 0:
         #mps[isite-1]=numpy.einsum("Jk,kl->Jl",mps[isite-1],u2)
         mps[isite-1]=numpy.tensordot(mps[isite-1],u2,axes=([1],[0]))
      else: 
         #mps[isite-1]=numpy.einsum("iJk,kl->iJl",mps[isite-1],u2)
         mps[isite-1]=numpy.tensordot(mps[isite-1],u2,axes=([1],[0]))
   return 0

def mps_leftSVD(mps,thresh,D,iprt=0):
   if iprt == 0:	
      print "\n[mps_leftSVD]"	
   elif iprt == 1:
      print "\n[mps_rightSVD]"
   N=len(mps)
   for i in range(N-1):
      b_i=str(mps_bdim(mps))
      mps_svd_lupdate(True,i,mps,thresh,D,iprt)
      b_o=str(mps_bdim(mps))
      if iprt != 2: print " SVD i b_i=",i,b_i,' -> ',b_o

def mps_reverse(mps):
   N=len(mps)
   tmps=[];
   tmps.append(mps[N-1].transpose(1,0).copy())
   for i in range(N-2,0,-1):
      tmps.append(mps[i].transpose(2,1,0).copy())
   tmps.append(mps[0].transpose(1,0).copy())
   return tmps

def mps_rightSVD(mps,thresh,D,iprt=1):
   tmps=mps_reverse(mps)
   mps_leftSVD(tmps,thresh,D,iprt)
   tmps=mps_reverse(tmps)
   #THIS DOES NOT WORK, use in-place replacement
   #mps=tmps 
   mps[:] = tmps

def mps_leftcanon(mps):
   thresh=0.0
   D=-1
   mps_leftSVD(mps,thresh,D)

def mps_rightcanon(mps):
   thresh=0.0
   D=-1
   mps_rightSVD(mps,thresh,D)

# 
# SOME BLAS like SUBROUTINES
#
def mps_dot(mps1,mps2):
#
# <MPS1|MPS2>
#
# p0 p1 p2    pN-1
# |__|__|_..._|    |MPS1>
#
# |__|__|_..._|    |MPS2>
#
   s1=len(mps1)
   s2=len(mps2)
   if s1 != s2:
      print "Inconsistent size in mps_dot:",s1,s2
      exit(1)
   # start contraction
   #tmp1=numpy.einsum('Iu,Id->ud',mps1[0],mps2[0])
   tmp1=numpy.tensordot(mps1[0],mps2[0],axes=([0],[0]))
   N=s1
   for i in range(1,N-1):
      #tmp2=numpy.einsum('ud,dJk->uJk',tmp1,mps2[i])
      tmp2=numpy.tensordot(tmp1,mps2[i],axes=([1],[0]))
      #tmp1=numpy.einsum('uJm,uJk->mk',mps1[i],tmp2)
      tmp1=numpy.tensordot(mps1[i],tmp2,axes=([0,1],[0,1]))
   # final site
   #tmp2=numpy.einsum('ud,dJ->uJ',tmp1,mps2[N-1])
   tmp2=numpy.tensordot(tmp1,mps2[N-1],axes=([1],[0]))
   #val =numpy.einsum('uJ,uJ',mps1[N-1],tmp2)
   val = numpy.tensordot(mps1[N-1],tmp2,axes=([0,1],[0,1]))
   return val

def mps_norm(mps):
   # Add complex conjugation
   ova = mps_dot(mps_conj(mps),mps)
   assert abs(ova.imag) < 1.e-10
   norm=math.sqrt(ova.real)
   return norm

def mps_scale(mps,alpha):
   if alpha<0.0:
      print "ERROR: scale factor < 0.0 ",alpha
      exit(1)
   N=len(mps)
   fac=math.pow(alpha,1.0/float(N))
   mps[:]=map(lambda x:x*fac,mps)

def mps_normalize(mps):
   norm=mps_norm(mps)
   mps_scale(mps,1.0/norm)

# Diff <P-Q|P-Q>
def mps_diff(mps1,mps2,iprt=0):
   pp=mps_dot(mps1,mps1)
   pq=mps_dot(mps1,mps2)
   qp=mps_dot(mps2,mps1)
   qq=mps_dot(mps2,mps2)
   rr=pp+qq-pq-qp
   if iprt >=1:
      print '[mps_diff] \nrr=%.5e \npp=%.5f \npq=%.5f \nqp=%.5f \nqq=%.5f '%(rr,pp,pq,qp,qq)
   if abs(pq-qp)>1.e-8*(pp+qq):
      print '[mps_diff] \nrr=%.5f \npp=%.5f \npq=%.5f \nqp=%.5f \nqq=%.5f '%(rr,pp,pq,qp,qq)
      print "ERROR: <p|q> != <q|p> in MPX_DIFF. ",pq-qp
      exit(1)
   if rr<0.0 and abs(rr)>1.e-8 and iprt>0:
      print "WARNING: <P-Q|P-Q> is not positive. rr=",rr
   return rr

# random mps with bond dimension D
def mps_random0(nsite,phys,D):
   mps=[];
   # first
   tmp=numpy.random.uniform(-1,1,phys*D).reshape((phys,D))
   mps.append(tmp)
   for i in range(1,nsite-1):
      tmp=numpy.random.uniform(-1,1,phys*D*D).reshape((D,phys,D))
      mps.append(tmp)
   # last
   tmp=numpy.random.uniform(-1,1,phys*D).reshape((D,phys))
   mps.append(tmp)
   return mps

def mps_random(mps,D,iop=0):
   N=len(mps)
   nmps=[];
   if iop==0:
      # first site
      s=mps[0].shape
      nmps.append(numpy.random.uniform(-1,1,s[0]*D).reshape(s[0],D))
      for i in range(1,N-1):
         s=mps[i].shape
         nmps.append(numpy.random.uniform(-1,1,D*s[1]*D).reshape(D,s[1],D))
      # last site
      s=mps[N-1].shape
      nmps.append(numpy.random.uniform(-1,1,D*s[1]).reshape(D,s[1]))
   elif iop==1:
      # first site
      s=mps[0].shape
      K=s[0]
      basis=numpy.identity(K)
      indx=numpy.random.randint(K,size=D)
      tmp=basis[indx]
      tmp=tmp.transpose(1,0)
      nmps.append(tmp)
      for i in range(1,N-1):
         s=mps[i].shape
	 K=s[1]
         basis=numpy.identity(K)
         indx=numpy.random.randint(K,size=D*D)
         tmp=basis[indx]
	 tmp=tmp.reshape((K,D,D))
	 tmp=tmp.transpose(1,0,2)
         nmps.append(tmp)
      # last site
      s=mps[N-1].shape
      K=s[1]
      basis=numpy.identity(K)
      indx=numpy.random.randint(K,size=D)
      tmp=basis[indx]
      nmps.append(tmp)
   print "DIMENSION=",mps_dim(nmps)
   return nmps

def mps_ext(mps,Dincr):
   N=len(mps)
   # first one	
   s=mps[0].shape
   tmp=numpy.zeros((s[0],s[1]+Dincr))
   tmp[:,:s[1]]=mps[0]
   mps[0]=tmp.copy()
   # middle
   for i in range(1,N-1):
      s=mps[i].shape
      tmp=numpy.zeros((s[0]+Dincr,s[1],s[2]+Dincr))
      tmp[:s[0],:s[1],:s[2]]=mps[i]
      mps[i]=tmp.copy()
   # last one
   s=mps[N-1].shape
   tmp=numpy.zeros((s[0]+Dincr,s[1]))
   tmp[:s[0],:]=mps[N-1]
   mps[N-1]=tmp.copy()
   
#
# Variationally minimize the error ERROR = <P-Q|Q-P>
# and find the minimal D to achive the error epsilon:
# The cost function is optimized by sweep, which 
# solving A*x[i]=b iteratively keeping mixed canonical form. 
#
# The key is that |P> is generally obtained as 'projection'
# of Q on the underlying manifold. Thus, its norm is less
# than 1.
#
# NOTE: The starting point is the RIGHT-canonical form. 
#
def mps_oneSite(mps,thresh):
   print "\n***** [mps_oneSite] *****"

   debug=False
   Dmax=max(mps_bdim(mps))
   #if Dmax == 1: return
   
   # for stability first normalze Q=MPS
   #print mps
   #print mps_dim(mps)
   #exit(1)
   norm = mps_norm(mps)
   mps_scale(mps,1.0/norm)
   mps_rightcanon(mps)
   N=len(mps)

   # initial guess by random right renormalized MPS
   D=1
   amps=mps_random(mps,D)
   mps_normalize(amps)
   mps_rightcanon(amps)
   
   #SVDguess
   #amps=copy.deepcopy(mps)
   #mps_rightSVD(amps,0.1,D)
   #print amps
   #print "AMPS=",mps_dim(amps)

   # begin optimization
   micro_thresh=1.e-8
   micro_maxiter=10
   macro_iter=1
   macro_maxiter=100
   macro_error=1.e0

   print "\nSite of MPS=",N
   print "Dmax of MPS=",Dmax
   print "NORM of MPS=",norm
   print "Initial Error:"
   mps_diff(amps,mps,1)

   #
   # a one-site sweep algorithm with canononicalization in each step
   #
   while D<Dmax and macro_iter<=macro_maxiter:
      print "\nMacro_iter = ",macro_iter," D = ",D
      macro_iter=macro_iter+1

      micro_iter=1
      micro_error=1.e0
      sweep_error_old=1.e0
      sweep_error_new=2.e0

      # 0. Initialize the Right-renormalize basis
      #    _______ ... _____    MPS approx
      #    | | | |     | | |
      #    | | | |     | | |	
      #    ======= ... =====    MPS exact
      #    0[1 2 3      N-2N-1]   
      # No. N-2		 1 0   (N-i-1) rbas
      # for the solution of site
      #    0 1 2 3 ...  N-2 
      rbas=[]
      tmp=numpy.einsum('iJ,kJ->ik',amps[N-1],mps[N-1])
      rbas.append(tmp)
      for i in range(N-2,0,-1):
          tmp2=numpy.einsum('iKu,ud->iKd',amps[i],tmp)
	  tmp =numpy.einsum('iKd,jKd->ij',tmp2,mps[i])
          rbas.append(tmp)
      #print rbas

      # 1. sweep
      while micro_error>micro_thresh and micro_iter<=micro_maxiter:
	 if debug: print "\n   micro_iter = ",micro_iter
	 micro_iter=micro_iter+1

	 #--- Start from right canonical form & Sweep to right ---
	 # 1. Right sweep and construct lbas 
	 #    [the last site is not touched]
         #    | | | |     | | |	
         #    ======= ... =====    MPS exact
         # No.[0 1 2 3      N-2]N-1 lbas
	 lbas=[]
	 amps[0]=numpy.einsum("Kl,ul->Ku",mps[0],rbas[N-2])
	 mps_svd_lupdate(True,0,amps,-0.1,D)
	 tmp =numpy.einsum("Ki,Kj->ij",amps[0],mps[0])
	 lbas.append(tmp)
	 for i in range(1,N-1):
	    string="   r-sweep[i] ="+str(i)
	    # Update amps[i]=((L*mps[i])*R)
	    tmp    =numpy.einsum('ij,jKl->iKl',lbas[i-1],mps[i])
	    amps[i]=numpy.einsum('iKl,ul->iKu',tmp,rbas[N-i-2])
	    # Recast A[i] into L-canonical form without cut off
	    # >>> QR can be used here actually <<<
	    mps_svd_lupdate(True,i,amps,-0.1,D)
	    #if mps_dot(amps,mps)<0.0:
	    #   amps[i]=-amps[i]
	    # Construct lbas via amps[i]*(lbas[i-1],mps[i])
	    lbas.append( numpy.einsum('iKu,iKd->ud',amps[i],tmp) )
	    sweep_error_new=mps_diff(amps,mps,0)
	    if debug: print string+" error = "+str(sweep_error_new)

	 # 2. left sweep and construct rbas again
         #     | | | |     | | |
	 #     ======= ... =====    MPS 
         #     0[1 2 3      N-2N-1]   
         # No.[0 1 2 3      N-2]N-1 lbas
         # No.  N-2	      1 0   (N-i-1) rbas
	 rbas=[]
	 amps[N-1]=numpy.einsum('ui,iK->uK',lbas[N-2],mps[N-1])
	 mps_svd_rupdate(True,N-1,amps,-0.1,D)
	 tmp=numpy.einsum('iJ,kJ->ik',amps[N-1],mps[N-1])
         rbas.append(tmp)
	 for i in range(N-2,0,-1):
	    string="   l-sweep[i] ="+str(i)
	    # Update amps[i]=(L*(mps[i]*R)) 
    	    tmp    =numpy.einsum('iKd,ud->iKu',mps[i],rbas[N-i-2])
	    amps[i]=numpy.einsum('ui,iKd->uKd',lbas[i-1],tmp)
	    # Recast A[i] into R-canonical form without cut off
	    # SVD - actually site i-1 is descarded in this case
	    #       while in eigenvalue problem it can be used as initial guess
	    mps_svd_rupdate(True,i,amps,-0.1,D)
	    #if mps_dot(amps,mps)<0.0:
	    #   amps[i]=-amps[i]
	    # Construct rbas via amps[i]*(rbas[i-1],mps[i])
	    rbas.append( numpy.einsum('jKu,iKu->ji',amps[i],tmp) )
	    sweep_error_new=mps_diff(amps,mps,0)
	    if debug: print string+" error = "+str(sweep_error_new)

         # define micro_error as error change after a whole sweep
         micro_error = abs(sweep_error_new - sweep_error_old)
	 sweep_error_old = sweep_error_new

      # define macro_error
      diff = mps_diff(amps,mps)
      macro_error = norm*diff
      print "\nMacro Error :"
      print " norm = %.10f  diff = %.10f  error = %.10f "%(norm,diff,macro_error)
      if macro_error<=thresh:
         break
      else:
	 D=D+1
	 print "\nExtend bond dimension:",D
	 if D < Dmax: mps_ext(amps,1)

   # check convergence
   print "\nFinal summary:"
   print " Dmax/D=",Dmax,D
   if macro_error>thresh:
      print " NOT COMPRESSIBLE in VARcompress (restore) !"
      mps_scale(mps,norm)
   else:
      mps[:]=amps
      mps_scale(mps,norm)
      print " mps_dim=",mps_dim(mps)
      print " norm=",norm," mps_norm=",mps_norm(mps)
   print "End of oneSitecompress\n"


def mps_hstack(mps):
   N=len(mps)
   v=[]
   v.append(mps[0])
   for i in range(1,N-1):
      tmp=mps[i].transpose(1,0,2)
      s=tmp.shape
      tmp=tmp.reshape((s[0],s[1]*s[2]))
      v.append(tmp)
   tmp=mps[N-1].transpose(1,0)
   v.append(tmp)
   v=numpy.hstack(v)
   return v
