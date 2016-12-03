#
# Solve Ax=-b
#
import time
import numpy
import scipy.linalg
from mpi4py import MPI

#
# Mask for Matrix vector product subroutine
#
class mask:
   def __init__(self,info,mvp):
      self.info = info
      self.mvp = mvp 
   def matvec(self,vec):
      return self.mvp(vec,*self.info)

class linsqSolver:
    def __init__(self):
        self.maxcycle =300
        self.crit_e   =1.e-10 # Tighter convergence.
        self.crit_vec =1.e-8
        self.crit_indp=1.e-12
        self.crit_demo=1.e-10
        # Basic setting
        self.iprt  = 1
        self.ndim  = 0
        self.neig  = 0
        self.diag  = None
        self.HVec  = None
	self.const = 0.0
        self.comm  = None
	self.dtype = numpy.float_
	self.mtype = MPI.DOUBLE
	self.nmvp  = 0
	self.lshift= 1.e-4
	self.noise = True
	self.nz    = 1.e-5
	# Projeciton
	self.ifex  = False
	self.qbas  = None
	# Full space of vectors
	self.ifall = True
	self.nfac  = 10
	# RHS of Linear squeation (neig,ndim)
	self.bvec  = None

    # Q*V=(1-C*C^d)*V, vbas.shape=(neig,vbas)
    def projection(self,vbas):
       vbas2 = vbas - reduce(numpy.dot,(vbas,self.qbas.T.conj(),self.qbas))
       return vbas2

    # Debug: full diagonalization
    def fullDiag(self):
	print 'diag=',self.diag
	print 'sumdiag=',numpy.sum(self.diag)
	v0 = numpy.identity(self.ndim,dtype=self.dtype)
        hmat = self.HVecs(v0)
        print 'hmat\n',hmat
	eig,vr = scipy.linalg.eigh(hmat)
	print 'eig=',eig
	exit()
    
    # Matrix vector product H[iproc]*v
    def HVecs(self,vbas):
        n = vbas.shape[0]
	self.nmvp += n
	# Parallel implementation
	self.comm.Barrier()
	wbas = numpy.zeros((n,self.ndim),dtype=self.dtype)
        for i in range(n):
           wbas_iproc = self.HVec(vbas[i])
 	   self.comm.Reduce([wbas_iproc,self.mtype],
		  	    [wbas[i],self.mtype],
		   	    op=MPI.SUM,root=0)
	return wbas
 
    # Generate V0 from B
    def genV0(self):
	return -self.bvec

    # Main solver: 
    def solve_iter(self,v0=None,iop=4):
	if self.neig > self.ndim:
	   print ' error in dvdson: neig>ndim, neig/ndim=',self.neig,self.ndim
	   exit(1)
	# Check the shape of RHS
	if self.comm.rank == 0: 
	   assert self.bvec.shape == (self.neig,self.ndim)
	   bnorm = numpy.linalg.norm(self.bvec)
	   print '[plinsq]: Ax=-b with (noise,size,ndim,crit_e,crit_v,||b||) = (%i,%d,%d,%.1e,%.1e,%.2e)'%\
				        (self.noise,self.comm.size,self.ndim,self.crit_e,self.crit_vec,bnorm)
	# Clear counter
	self.nmvp = 0
	t0 = time.time()
	#
	# ONLY rank-0 generates the basis, input v0 is an np.array (neig,ndim)
	#
	if self.comm.rank == 0:
	   if v0 == None:
	      vbas = self.genV0()
           else:
	      vbas = v0.copy()
	      if numpy.linalg.norm(vbas) < 1.e-14: 
	         print ' Regenerate initial guess ...'
		 vbas = self.genV0()
	   # Add random noise to interact with the whole space
	   if self.noise: vbas = vbas + self.nz*numpy.random.uniform(-1,1,size=(self.neig,self.ndim))
	   if self.ifex: vbas = self.projection(vbas)
	   # Check again
	   normV0 = numpy.linalg.norm(vbas[0])
	   print ' norm of V0 for plinsq =',normV0
        else:
	   normV0 = 0.
        # 2016.08.31: Check norm of initial guess
	normV0 = self.comm.bcast(normV0)
	if normV0 < 1.e-12:
	   # Return (eigs,rbas) directly
	   eigs = numpy.zeros(self.neig)
	   rbas = numpy.zeros_like(self.bvec)
	   return eigs,rbas,self.nmvp
        else:
	   if self.comm.rank == 0:
              genOrthoBas(vbas,self.crit_indp)
	   else:
	      vbas = numpy.empty((self.neig,self.ndim),dtype=self.dtype)
	self.comm.Bcast([vbas,self.mtype])
	wbas = self.HVecs(vbas)
	if self.comm.rank == 0: 
	   print ' iter  dim  nmvp   ieig         eig[-x^t*b]	 ediff      rnorm     emin     time/s     '
	   print ' --------------------------------------------------------------------------------------'
        #
        # Begin to solve
        #
        ineg  = 0
	ifconv= False
        neig  = self.neig
        eigs  = numpy.zeros(neig)+1.e3
	eigval= numpy.zeros((neig,self.maxcycle+1))+1.e3
	# initial dimension
	ndim  = neig
        for niter in range(1,self.maxcycle+1):

           #=======================================
	   # ONLY rank-0 compute the small problem
           #=======================================
	   if self.comm.rank == 0:
	      # Check othonormality of basis 
	      iden = numpy.dot(vbas,vbas.T.conj())
	      diff = numpy.linalg.norm(iden-numpy.identity(ndim))
	      if diff > 1.e-10:
	         print ' diff_VBAS=',diff
	         print iden
	         exit(1)
	      # An important note: Vh*H*V \= Vt*Hc*Vc
	      # WRONG  : tmpH = numpy.dot(vbas,wbas.T.conj())
	      # CORRECT:
	      tmpH = numpy.dot(vbas.conj(),wbas.T)
	      diff = numpy.linalg.norm(tmpH-tmpH.T.conj())
	      if diff > 1.e-10:
	         print ' diff_skewH=',diff
		 print ' tmpH =\n',tmpH
	         exit(1)
	      # Explicit symmetrizaiton
	      tmpH = 0.5*(tmpH+tmpH.T.conj())
	      # Linear equation
	      tmpB = numpy.dot(vbas.conj(),self.bvec.T)
	      # EigenSolve
	      vr = -scipy.linalg.solve(tmpH,tmpB)
	      # CHECK ORTHOGONALITY for vr[ndim,neig]
	      vr = vr[:,:neig].T.copy()
	      # Let L[x]=xd*A*x+(xd*b+bd*x) => A*x=-b; 
	      # then L[x*] = x*^t*b = -b*Ainv*b < 0
	      teig = numpy.einsum('ip,pi->i',vr.conj(),tmpB).real

              ##################################################
	      # Check eigenvalues: To detect 'intruder problem'
              ##################################################
	      e,v = scipy.linalg.eigh(tmpH)
	      if e[0]<0.0 and ineg==0: 
		 print ' Warning: negative eigenvalue: e[0]=',e[0]
		 ineg = 1
              ##################################################

  	      # Eigenvalue convergence
  	      nconv1 = 0
	      econv  = [False]*neig
  	      for i in range(neig):
  	         tmp = teig[i]-eigs[i]
  	         if abs(tmp) <= self.crit_e: 
		    econv[i] = True
		    nconv1+=1
  	      eigs = teig.copy()
 	      eigval[:,niter] = teig.copy()
	      
	      # Full Residuals: Res[i]=Ax+b
  	      rbas = numpy.dot(vr,wbas) + self.bvec
	      # For numerical reason
	      if self.ifex: rbas = self.projection(rbas)
	      nconv2 = 0
  	      rindx  = []
              rconv  = [0]*neig
	      for i in range(neig):
	         tmp = numpy.linalg.norm(rbas[i,:])
		 # Criteria for convergence
		 if tmp <= self.crit_vec:
  	            nconv2 +=1
  	            rconv[i]=(True,tmp)
                 else:
  	            rconv[i]=(False,tmp)  
		    rindx.append(i)     
 
  	      # Printing
	      t1 = time.time()
	      if self.iprt >= 0:
	         for i in range(neig):
	            print '%4d %4d %5d %4d + %s %20.12f %10.2e %9.1e %9.1e %9.1e'%(niter,ndim,self.nmvp,\
	                  i,str(rconv[i][0] or econv[i])[0],self.const+eigval[i,niter],\
		          eigval[i,niter]-eigval[i,niter-1],rconv[i][1],e[0],t1-t0)
	      t0 = time.time()

           #=======================================
           # Check convergence on rank-0 and then
	   # broadcast the result to each proc.
	   #=======================================
	   if self.comm.rank == 0:
   	      # Convergence by either criteria (NO - just residual)
	      ifconv = (nconv2 == neig) or (niter > 1 and nconv1 == neig)
	   else:
	      ifconv = None
	   ifconv = self.comm.bcast(ifconv,root=0)

	   # If converged, exit in all processors
  	   if ifconv or niter == self.maxcycle: break
 	   
           #=======================================
           # If not converged, use processor-0 to 
	   # generate the new basis and bcast them.
           #=======================================
      	   if self.comm.rank == 0:

              ##########################################
	      # Reduce the basis: to be done in future
              ##########################################
	      if (not self.ifall) or (self.ifall and ndim>self.nfac*self.neig):
	         # Only use the first one
	         vr = vr[0]/numpy.linalg.norm(vr[0])
	         vr = vr.reshape(1,ndim)
	         vbas = numpy.dot(vr,vbas)
	         wbas = numpy.dot(vr,wbas)
	      
              # New directions from residuals
	      for i in range(neig):
  	         if rconv[i][0] == True: continue
     	         # Various PRECONDITIONER:
	         if iop == 0:
	            # gradient
	            pass
	         elif iop == 4:
	            # ABS+LEVEL-SHIFT ~ Davidson+Gradient
		    tmp = abs(self.diag) + self.lshift
	            rbas[i,:] = rbas[i,:]/tmp

	      # Projection
	      if self.ifex: rbas = self.projection(rbas)
	      # Re-orthogonalization and get Nindp
	      nindp,vbas2 = dvdson_ortho(vbas,rbas[rindx,:],self.crit_indp)
	      if self.iprt > 0: print ' final nindp = ',nindp

 	   else:
	      nindp = None
	   
	   # Bcast for constructing vbas2 added to the previous ones 
   	   nindp = self.comm.bcast(nindp)
           if nindp != 0:
	      if self.comm.rank != 0:
		 vbas2 = numpy.empty((nindp,self.ndim),dtype=self.dtype)
	      self.comm.Bcast([vbas2,self.mtype])
              wbas2 = self.HVecs(vbas2)
	      if self.comm.rank == 0:
	         vbas  = numpy.vstack((vbas,vbas2))
                 wbas  = numpy.vstack((wbas,wbas2))
	         ndim  = vbas.shape[0]
           else:
	      print 'Convergence failure: unable to generate new direction: Nindp=0 !'
              exit(1)
	   
        if not ifconv:
           print 'Convergence failure: out of maxcycle ! maxcycle =',self.maxcycle
           exit(1)

        #=======================================
        # Only return the eigens from rank-0
	#=======================================
	if self.comm.rank == 0:
	   rbas = numpy.dot(vr,vbas)
	else:
	   rbas = None

	# Only processor-0 holds the correct [eigs,rbas]
	return eigs,rbas,self.nmvp

#
# From vbas to generate orthonormal basis
#
def genOrthoBas(vbas,crit_indp):
   vbas[0] = vbas[0]/numpy.linalg.norm(vbas[0])
   nbas = vbas.shape[0]
   if nbas != 1: 
      nindp,vbas2 = dvdson_ortho(vbas[0:1],vbas[1:],crit_indp)
      if nindp + 1 == nbas:
	 vbas[1:] = vbas2
      else:
	 print 'error: insufficient orthonormal basis: nbas/nindp',nbas,nindp+1
         exit()
   return 0

#
# Orthonormalization basis from rbas against previous vbas
#
def dvdson_ortho(vbas,rbas,crit_indp):
    debug = False
    if debug: print '[dvdson_ortho]'
    ndim  = vbas.shape[0] 
    nres  = rbas.shape[0]
    nindp = 0
    vbas2 = numpy.zeros(rbas.shape,dtype=rbas.dtype)
    # Clean projection (1-V*Vh)*R => ((V*Vh)*R)^t = Rt*Vht*Vt
    maxtimes = 5
    for k in range(maxtimes):
       rbas = rbas - reduce(numpy.dot,(rbas,vbas.T.conj(),vbas))
    # Final new basis from rbas   
    for i in range(nres): 
       rvec = rbas[i,:].copy()	    
       rii  = numpy.linalg.norm(rvec)
       if rii <= crit_indp: continue
       if debug: print '  i,rii=',i,rii
       # NORMALIZE
       rvec = rvec / rii
       rii  = numpy.linalg.norm(rvec)
       rvec = rvec / rii
       vbas2[nindp] = rvec
       nindp = nindp +1
       # Substract all previous things
       for k in range(maxtimes):
          rbas[i:,:] -= reduce(numpy.dot,(rbas[i:,:],vbas.T.conj(),vbas))
          rbas[i:,:] -= reduce(numpy.dot,(rbas[i:,:],vbas2[:nindp,:].T.conj(),vbas2[:nindp,:]))
    # Final basis from rbas
    vbas2 = vbas2[:nindp].copy()
    # iden
    if debug and nindp != 0:	   
       tmp = numpy.vstack((vbas,vbas2)) 	 	    
       iden = numpy.dot(tmp,tmp.T.conj())
       diff = numpy.linalg.norm(iden-numpy.identity(iden.shape[0],dtype=rbas.dtype))
       if diff > 1.e-10:
          print ' error in mgs_ortho: diff=',diff
          print iden
          exit(1)
       else:
          print ' final nindp from mgs_ortho =',nindp,' diffIden=',diff	   
    return nindp,vbas2


if __name__ == '__main__':

   # Real symmetric A
   def test_real():
      ndim = 100
      neig = 2
      numpy.random.seed(0)
      mat = numpy.random.uniform(-1,1,size=(ndim,ndim)) 
      mat = 0.5*(mat+mat.T) + numpy.diag(range(ndim))
      b = numpy.random.uniform(-1,1,size=(ndim,neig))
      x0 = scipy.linalg.solve(mat,b)
      e,v = scipy.linalg.eigh(mat)
      print 'eigs=',e

      def matvecp(v,mat):
         return mat.dot(v)
      info = [mat]
      masker = mask(info,matvecp)

      solver = linsqSolver()
      solver.iprt = 0
      solver.crit_vec = 1.e-7
      solver.maxcycle = 1000
      solver.ndim = ndim
      solver.diag = numpy.diag(mat)
      solver.neig = neig
      solver.HVec = masker.matvec
      solver.comm  = MPI.COMM_WORLD
      solver.dtype = numpy.float_
      solver.mtype = MPI.DOUBLE
      solver.bvec  = -b.T.copy()
      e1,x1,nmvp = solver.solve_iter(v0=None,iop=4)
      print 'vec1 =',x1.shape
      print 'vec0 =',x0.shape
      print 'vdiff=',numpy.linalg.norm(x1[0]-x0[:,0])
      return 0

   # Complex Hermitian A
   def test_complex():
      ndim = 100
      neig = 2
      numpy.random.seed(0)
      mat1 = numpy.random.uniform(-1,1,size=(ndim,ndim)) 
      mat2 = numpy.random.uniform(-1,1,size=(ndim,ndim))
      # The current method only applies to diagonal dominate case !
      hmat = 0.1j*(mat2-mat2.T) + 0.5*numpy.diag(range(ndim))
      print 'Hermicity=',numpy.linalg.norm(hmat-hmat.T.conj())
      b = numpy.random.uniform(-1,1,size=(ndim,neig)) \
        + 1.j*numpy.random.uniform(-1,1,size=(ndim,neig)) 
      x0 = scipy.linalg.solve(hmat,b)

      def matvecp(v,mat):
         return mat.dot(v)

      info = [hmat]
      masker = mask(info,matvecp)
      solver = linsqSolver()
      solver.iprt = 0
      solver.crit_vec = 1.e-7
      solver.maxcycle = 100
      solver.ndim = ndim
      solver.diag = numpy.diag(hmat.real)
      solver.neig = neig
      solver.HVec = masker.matvec
      solver.noise = True #False
      solver.comm  = MPI.COMM_WORLD
      solver.dtype = numpy.complex_
      solver.mtype = MPI.C_DOUBLE_COMPLEX
      solver.bvec  = -b.T.copy()
      e1,x1,nmvp = solver.solve_iter(v0=None,iop=4)
      print 'vec1 =',x1.shape
      print 'vec0 =',x0.shape
      print 'vdiff=',numpy.linalg.norm(x1[0]-x0[:,0])
      return 0

   # TEST
   #test_real()
   test_complex()
