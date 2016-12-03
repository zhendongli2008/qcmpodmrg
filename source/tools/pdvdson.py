#
# Solve Hc=ce
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

#
# LOBPCG solver with Davidson precondition
#
class eigenSolver:
    def __init__(self):
        self.maxcycle =20
        self.crit_e   =1.e-8
        self.crit_vec =1.e-8
        self.crit_indp=1.e-12
        self.crit_demo=1.e-10
        # Basic setting
        self.iprt  = 1
        self.ndim  = 0
        self.neig  = 0
        self.diag  = None
        self.HVec  = None
	self.noise = True
	self.const = 0.0
        self.comm  = None
	self.dtype = numpy.float_
	self.mtype = MPI.DOUBLE
	self.nmvp  = 0
	self.lshift= 1.e-4
	self.nz    = 1.e-5
	# Projeciton
	self.ifex  = False
	self.qbas  = None
	# Full space of vectors
	self.ifall = True
	self.nfac  = 10

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
    
    # Matrix vector product H[iproc]*V
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
 
    # Generate V0 from real diagonal values
    def genV0(self,neig):
        # Break the degeneracy artifically in generating v0
	diag = self.diag + 1.e-12*numpy.arange(1,1+self.ndim)/float(self.ndim)
	index = numpy.argsort(diag)[:neig]
        v0 = numpy.zeros((neig,self.ndim),dtype=self.dtype)
	v0[range(neig),index] = 1.0
        return v0

    # Main solver: LOBPCG + Davidson preconditioning
    def solve_iter(self,v0=None,iop=4,ifplot=False):
	if self.neig > self.ndim:
	   print ' error in dvdson: neig>ndim, neig/ndim=',self.neig,self.ndim
	   exit(1)
	# Clear counter
	self.nmvp = 0
	t0 = time.time()
	#
	# ONLY rank-0 generates the basis, input v0 is an np.array (neig,ndim)
	#
	if self.comm.rank == 0:
	   if v0 is None:
              vbas = self.genV0(self.neig)
           else:
	      vbas = v0.copy()
	   # Add random noise to interact with the whole space
	   if self.noise: vbas = vbas + self.nz*numpy.random.uniform(-1,1,size=(self.neig,self.ndim))
	   if self.ifex: vbas = self.projection(vbas) 
	   genOrthoBas(vbas,self.crit_indp)
	else:
	   vbas = numpy.empty((self.neig,self.ndim),dtype=self.dtype)
	self.comm.Bcast([vbas,self.mtype])
        wbas = self.HVecs(vbas)
        #
        # Begin to solve
        #
        ifconv= False
        neig  = self.neig
        eigs  = numpy.zeros(neig)+1.e3
	# Record history
	rnorm = numpy.zeros((neig,self.maxcycle))
	eigval= numpy.zeros((neig,self.maxcycle+1))+1.e3
	# initial dimension
	ndim  = neig
        for niter in range(1,self.maxcycle):
           
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
	      # EigenSolve
	      eig,vr = scipy.linalg.eigh(tmpH)
	      vr = vr[:,:neig].T.copy()
	      # CHECK ORTHOGONALITY for vr[neig,ndim]
	      over = numpy.dot(vr,vr.T.conj())
	      diff = numpy.linalg.norm(over-numpy.identity(neig))
	      if diff > 1.e-10: 
	         print ' diff_VEC=',diff
	         print over
	         exit(1)
	      # Save
	      teig = eig[:neig]
 	 	 
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
  
              # Full Residuals: Res[i]=Res'[i]-w[i]*X[i]
  	      rbas = numpy.dot(vr,vbas)
  	      rbas = numpy.dot(vr,wbas) - numpy.dot(numpy.diag(eigs),rbas)
	      if self.ifex: rbas = self.projection(rbas)
	      nconv2 = 0
  	      rindx  = []
              rconv  = [0]*neig
	      for i in range(neig):
	         tmp = numpy.linalg.norm(rbas[i,:])
	         rnorm[i,niter-1] = tmp
		 # Criteria for convergence
		 if tmp <= self.crit_vec:
  	            nconv2 +=1
  	            rconv[i]=(True,tmp)
                 else:
  	            rconv[i]=(False,tmp)  
		    if not econv[i]: rindx.append(i)     

  	      # Printing
	      t1 = time.time()
	      if self.iprt >= 0:
	         if niter == 1: 
		    print '[pdvdson]: Hc=ce with (noise,size,ndim,crit_e,crit_v) = (%i,%d,%d,%.1e,%.1e)'%\
				      (self.noise,self.comm.size,self.ndim,self.crit_e,self.crit_vec)
	            print ' iter  dim  nmvp   ieig         eigenvalue        ediff      rnorm     time/s  '
	            print ' ------------------------------------------------------------------------------'
		 if niter%2 == 1:
	            for i in range(neig):
	               print '%4d %4d %5d %4d + %s %20.12f %11.3e %10.3e %9.2e'%(niter,ndim,self.nmvp,\
	                     i,str(rconv[i][0] or econv[i])[0],self.const+eigval[i,niter],\
		             eigval[i,niter]-eigval[i,niter-1],rconv[i][1],t1-t0)
	         else:		
	            for i in range(neig):
	               print '%4d %4d %5d %4d - %s %20.12f %11.3e %10.3e %9.2e'%(niter,ndim,self.nmvp,\
	                     i,str(rconv[i][0] or econv[i])[0],self.const+eigval[i,niter],\
		             eigval[i,niter]-eigval[i,niter-1],rconv[i][1],t1-t0)
	      t0 = time.time()

           #=======================================
           # Check convergence on rank-0 and then
	   # broadcast the result to each proc.
	   #=======================================
	   if self.comm.rank == 0:
   	      # Convergence by either criteria (NO - just residual)
	      #ifconv = (nconv1 == neig) or (nconv2 == neig)
	      ifconv = len(rindx) == 0
	   else:
	      ifconv = None
	   ifconv = self.comm.bcast(ifconv,root=0)

	   # If converged, exit in all processors
  	   if ifconv or niter == self.maxcycle-1:
              #=======================================
              # Only return the eigens from rank-0
	      #=======================================
	      if self.comm.rank == 0:
	         eigs = eigs + self.const
		 rbas = numpy.dot(vr,vbas)
	      else:
		 eigs = None
		 rbas = None
   	      break		
 	   
           #=======================================
           # If not converged, use processor-0 to 
	   # generate the new basis and bcast them.
           #=======================================
      	   if self.comm.rank == 0:

	      # Reduce the basis to span{x[k],x[k]-x[k-1]}
	      if (not self.ifall) or (self.ifall and ndim>self.nfac*self.neig):
                 # Rotated basis to minimal subspace that
  	         # can give the exact [neig] eigenvalues
	         # Also, the difference vector = xold - xnew as correction 
	         pr = numpy.identity(ndim,dtype=self.dtype)[:neig,:] - vr
	         nindp,vr2 = dvdson_ortho(vr,pr[rindx,:],self.crit_indp)
	         if nindp !=0: vr = numpy.vstack((vr,vr2))
	         vbas = numpy.dot(vr,vbas)
	         wbas = numpy.dot(vr,wbas)
	      
              # New directions from residuals
	      for i in range(neig):
  	         if rconv[i][0] == True: continue
     	         # Various PRECONDITIONER:
	         if iop == 0:
	            # gradient
	            pass
	         elif iop == 1:
	            # Davidson
	            tmp = self.diag - eigs[i]
 	            tmp[abs(tmp)<self.crit_demo] = self.crit_demo
	            rbas[i,:] = rbas[i,:]/tmp
 	         elif iop == 2:
	            # Olsen's algorithm works for close diag ~ H : 0.00067468 [3]
	            tmp = self.diag - eigs[i]
 	            tmp[abs(tmp)<self.crit_demo] = self.crit_demo	
	            e1 = numpy.dot(vbas[i,:],rbas[i,:]/tmp)/numpy.dot(vbas[i,:],vbas[i,:]/tmp)
	            rbas[i,:] = -(rbas[i,:]-e1*vbas[i,:])/tmp
	         elif iop == 3:
	            # ABS     
	            tmp = abs(self.diag - eigs[i])
 	            tmp[abs(tmp)<self.crit_demo] = self.crit_demo	
	            rbas[i,:] = rbas[i,:]/tmp
	         elif iop == 4:
	            # ABS+LEVEL-SHIFT ~ Davidson+Gradient
		    tmp = abs(self.diag - eigs[i]) + self.lshift
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
           #exit(1)

	#
	# Plot iteration history if necessary
	#
	if ifplot:
           import matplotlib.pyplot as plt
	   plt.plot(range(self.ndim),self.diag)
           plt.show()
	   plt.savefig("diag.png")

	   for i in range(self.neig):
 	      plt.plot(range(1,niter+1),numpy.log10(rnorm[i,:niter]),label=str(i+1))
	   plt.legend()  
	   plt.savefig("res_conv.png")
           plt.show()
        
	   for i in range(self.neig):
 	      plt.plot(range(1,niter+1),eigval[i,1:niter+1],label=str(i+1))
	   plt.legend()  
	   plt.savefig("eig_conv.png")
           plt.show()

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
       nindp = nindp + 1
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
      mat = 0.5*(mat+mat.T)
      efull,v = scipy.linalg.eigh(mat)

      def matvecp(v,mat):
         return mat.dot(v)

      info = [mat]
      masker = mask(info,matvecp)
      solver = eigenSolver()
      solver.iprt = 0
      solver.crit_vec = 1.e-7
      solver.maxcycle = 200
      solver.ndim = ndim
      solver.diag = numpy.diag(mat)
      solver.neig = neig
      solver.HVec = masker.matvec
      solver.noise = True #False
      solver.comm  = MPI.COMM_WORLD
      solver.dtype = numpy.float_
      solver.mtype = MPI.DOUBLE
      eigs,civec,nmvp = solver.solve_iter(v0=None,iop=4)
      vec = v[:,:neig].T.copy()
      ova = numpy.abs(vec.dot(civec.T))
      print 'eigs  =',eigs
      print 'efull =',efull[:neig]
      print 'civec =',civec.shape
      print 'vec   =',vec.shape
      print 'vdiff =',numpy.linalg.norm(ova-numpy.identity(neig))
      return 0

   # Complex Hermitian A
   def test_complex():
      ndim = 100
      neig = 2
      numpy.random.seed(0)
      mat1 = numpy.random.uniform(-1,1,size=(ndim,ndim)) 
      mat2 = numpy.random.uniform(-1,1,size=(ndim,ndim))
      # eig =  -4.56176434e-02   9.84557515e-01   1.97239188e+00   2.98118809e+00
      #mat = numpy.diag(range(ndim))+1.e-1j*(mat2-mat2.T)
      mat = 0.5j*(mat2-mat2.T)
      print 'Hermicity=',numpy.linalg.norm(mat-mat.T.conj())
      efull,v = scipy.linalg.eigh(mat)

      def matvecp(v,mat):
         return mat.dot(v)

      info = [mat]
      masker = mask(info,matvecp)
      solver = eigenSolver()
      solver.iprt = 0
      solver.crit_vec = 1.e-7
      solver.maxcycle = 200
      solver.ndim = ndim
      solver.diag = numpy.diag(mat.real)
      solver.neig = neig
      solver.HVec = masker.matvec
      solver.noise = True #False
      solver.comm  = MPI.COMM_WORLD
      solver.dtype = numpy.complex_
      solver.mtype = MPI.C_DOUBLE_COMPLEX
      eigs,civec,nmvp = solver.solve_iter(v0=None,iop=4)
      vec = v[:,:neig].T.copy()
      ova = numpy.abs(vec.dot(civec.T.conj()))
      print 'eigs  =',eigs
      print 'efull =',efull[:neig]
      print 'civec =',civec.shape
      print 'vec   =',vec.shape
      print 'vdiff =',numpy.linalg.norm(ova-numpy.identity(neig))
      return 0

   # TEST
   test_real()
   test_complex()
