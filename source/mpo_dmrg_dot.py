#!/usr/bin/env python
#
# ci_solver & pt_solver
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def ci_solver(dmrg,isite,ncsite,flst,status,ifsym):
# def pt_solver(dmrg,isite,ncsite,flst,status,ifsym):
#
import numpy
from mpi4py import MPI
from tools import plinsq
from tools import pdvdson
from tools import pgdvdson
import mpo_dmrg_kernel
import mpo_dmrg_dotutil
from sysutil_include import dmrg_dtype,dmrg_mtype

#
# Solve (generalized) eigenvalue problem
#
def ci_solver(dmrg,isite,ncsite,actlst,flst,status,ifsym):
   rank = dmrg.comm.rank
   #
   # Initialization
   #
   dicDPT = mpo_dmrg_dotutil.initializeDPTspace(dmrg,isite,ncsite,flst,ifsym)
   ndim = dmrg.dims[3]
   info = [dmrg,isite,ncsite,flst,status,ifsym]
   ioff = 0
   eigs = []
   civecs = []
   nmvps = 0
   #
   # Loop over states
   #
   for qkey in dmrg.qsectors:
      # 
      # Symmetry of the space
      #
      key,neig,ndim0,prjmap = mpo_dmrg_dotutil.symmetrySpaceInfo(dmrg,qkey,ncsite,dicDPT)
      if rank == 0: 
	 print ' Qsym =',key,' neig =',neig
	 print ' Dimensions for ldim,cdim,rdim,ndim,ndim0 =',dmrg.dims,ndim0
      #
      # Diag
      #
      Diag = mpo_dmrg_dotutil.genHDiag(info,ndim0,prjmap)
      #
      # Solve HC=(S)CE 
      #
      if not dmrg.ifs2proj:
	 masker = pdvdson.mask([info,ndim0,prjmap],mpo_dmrg_kernel.HVec)
         solver = pdvdson.eigenSolver()
         solver.HVec = masker.matvec
      else:
         masker0 = pgdvdson.mask([info,ndim0,prjmap],mpo_dmrg_kernel.HVec)
         masker1 = pgdvdson.mask([info,ndim0,prjmap],mpo_dmrg_kernel.SVec)
         solver = pgdvdson.eigenSolver()
         solver.HVec = masker0.matvec
         solver.SVec = masker1.matvec  
      #	
      # Generate orthonormal basis only at rank-0 
      #
      if rank == 0 and dmrg.ifex:
         solver.ifex = True
	 pbas = mpo_dmrg_kernel.PBas(info,ndim0,prjmap)
	 u,sigs,vt = mpo_dmrg_kernel.PBasOrtho(pbas)
	 solver.qbas = vt.copy()
      solver.iprt = 0
      solver.crit_e = dmrg.crit_e
      solver.crit_vec = dmrg.crit_vec
      solver.ndim = ndim0
      solver.neig = neig
      solver.diag = Diag
      solver.noise = dmrg.ifdvdnz and (dmrg.noise > 1.e-10) 
      solver.const = dmrg.const
      solver.comm  = dmrg.comm
      solver.dtype = dmrg_dtype
      solver.mtype = dmrg_mtype
      # To be consistent with [genGuess], map Vl*Vr to Vlr
      v0guess = None
      if rank == 0 and dmrg.psi0 is not None:
         v0guess = dmrg.psi0[ioff:ioff+neig,prjmap].copy()
      # Davidson
      t_start = MPI.Wtime()
      eig,vec,nmvp = solver.solve_iter(v0=v0guess,iop=4)
      t_diff = MPI.Wtime() - t_start
      #
      # Save info on rank0
      #
      if rank == 0:
         civec = numpy.zeros((neig,ndim),dtype=dmrg_dtype)
         civec[:,prjmap] = vec
         civecs.append(civec)
	 eigs  += list(eig)
         nmvps += nmvp
         ioff  += neig  
         if dmrg.iprt >= 0: 
            print ' Eigenvalues = ',eig
	    print ' VectorNorms = ',map(lambda x:numpy.linalg.norm(x),vec)
  	    if dmrg_dtype == numpy.complex_:
	       print ' VectorReal2 = ',map(lambda x:x.dot(x),vec.real)
	       print ' VectorImag2 = ',map(lambda x:x.dot(x),vec.imag)
	    print ' Wtime for eigen problem of Qsym:%s = %.2f s'%(key,t_diff)
   #
   # Final
   #
   eigsArray,sigs,dwts,qred,site,srotR = mpo_dmrg_dotutil.finalizeDot(info,civecs,eigs,actlst)
   return nmvps,eigsArray,sigs,dwts,qred,site,srotR

#
# Solve linear equation for PT
#
def pt_solver(dmrg,isite,ncsite,actlst,flst,status,ifsym):
   rank = dmrg.comm.rank
   #
   # Initialization
   #
   dicDPT = mpo_dmrg_dotutil.initializeDPTspace(dmrg,isite,ncsite,flst,ifsym)
   ndim = dmrg.dims[3]
   info = [dmrg,isite,ncsite,flst,status,ifsym]
   ioff = 0
   eigs = []
   civecs = []
   nmvps = 0
   #
   # Loop over states
   #
   for qkey in dmrg.qsectors:
      # 
      # Symmetry of the space
      #
      key,neig,ndim0,prjmap = mpo_dmrg_dotutil.symmetrySpaceInfo(dmrg,qkey,ncsite,dicDPT)
      if rank == 0: 
	 print ' Qsym =',key,' neig =',neig
	 print ' Dimensions for ldim,cdim,rdim,ndim,ndim0 =',dmrg.dims,ndim0
      #
      # Diag
      #
      Diag = mpo_dmrg_dotutil.genHDiag(info,ndim0,prjmap)
      #
      # Solve Ax=-b by plinsq.linsqSolver()
      #
      masker0 = plinsq.mask([info,ndim0,prjmap],mpo_dmrg_kernel.HVec)
      solver = plinsq.linsqSolver()
      solver.HVec = masker0.matvec
      # Set up parameters
      solver.iprt = 0
      solver.crit_e = dmrg.crit_e
      solver.crit_vec = dmrg.crit_vec
      solver.ndim = ndim0
      solver.neig = neig
      solver.diag = abs(Diag-dmrg.et)+1.e-4 # for PT
      solver.noise = dmrg.ifdvdnz and (dmrg.noise > 1.e-10)
      solver.const = dmrg.const
      solver.comm  = dmrg.comm
      solver.dtype = dmrg_dtype
      solver.mtype = dmrg_mtype
      #
      # Construction of RHS = <Psi(1)|(H-E0)|chi[m]>c[m]
      #
      BVec = mpo_dmrg_dotutil.genRHSpt(info,ndim0,prjmap)
      #
      # Compression of H|Psi0> by minimizing L[Psi1]=<Psi1-Psi0|Psi1-Psi0>.
      #
      if dmrg.ifcompression:
	 # Locally, the fitting functional is exact. <dPsi/dB^*|Psi> = <dPsi/dB^*|H|Psi0> 
	 # such that in mixed canonical form, where the metric is identity, the result can 
	 # be simply got from Bk = <bk|Psi> = <bk|H|Psi0>.
	 assert neig == 1
	 vec = BVec.reshape((neig,ndim0))
	 # The norm her in exact case is equal to sqrt{<Psi|HH|Psi>} = sqrt{E0^2} = |E0|.
	 eig = [numpy.linalg.norm(vec)]
	 nmvp = 0
      #
      # Normal PT: L[Psi1]=<Psi1|H0-E0|Psi1>+2<Psi0|V|Psi0>.
      #
      else:
	 #
	 # Construct <Psi|Hd|Psi[i]> which could be used for several tasks:
	 # (1) imposing conjugate direction, (2) doing high-order PT with Hd.
	 #
         if dmrg.ifH0:
      	    HdVecI = mpo_dmrg_dotutil.genRHSpt(info,ndim0,prjmap,iHd=1)
   	    HdVec = numpy.dot(dmrg.coef,HdVecI)
         else:
            HdVec = BVec
         #
	 # Projectors: (1) {|i>}, (2) {|0>}+{A|i>}.
         #
         solver.ifex = True
	 if rank == 0:
            pbas = mpo_dmrg_kernel.PBas(info,ndim0,prjmap)
            # Projection operators: P = |0><0|
	    if not dmrg.ifH0ortho:
               # For numerical reason, it is essential to use orthonormal 
	       # projector, since sigs can be very small.
	       if dmrg.ifptn:
                  u,sigs,vt = mpo_dmrg_kernel.PBasOrtho(pbas[0:1])
               else:
                  u,sigs,vt = mpo_dmrg_kernel.PBasOrtho(pbas)
               solver.qbas = vt.copy()
	    # H0-orthogonality
     	    else:
	       pbas[1:] = HdVecI[1:] - dmrg.et*pbas[1:] # <l1|Hd-E0|psi[i]>
	       u,sigs,vt = mpo_dmrg_kernel.PBasOrtho(pbas)
	       solver.qbas = vt.copy()
	 #   
	 # Formation of RHS using BVec just for rank-0
         #
	 if rank == 0:
	    # General-order PT: H0 = PE0P+QHdQ
	    # then U = H-H0 such that QU = QH-QHdQ
	    # RHS = -Q*(U|n-1>+sum_k=1^n e[k]|n-k>)
	    # n=1: RHS = -Q*U|0> = -QH|0>
	    # n>1: RHS = -[Q(H-Hd)|n-1> + sum_k=1^n e[k]|n-k>]
	    if dmrg.ifptn and dmrg.nref > 1:
               BVec = BVec - HdVec
               for iorder in range(1,dmrg.nref):
                  BVec -= dmrg.enlst[iorder]*pbas[dmrg.nref-iorder]
            # Reshape for plinsq
	    BVec = BVec.reshape(1,ndim0).copy()
            if dmrg.ifs2proj: BVec = BVec*dmrg.n0
            solver.bvec = solver.projection(BVec)
         # To be consistent with [genGuess], map Vl*Vr to Vlr
         v0guess = None
         if rank == 0 and dmrg.psi0 is not None:
            v0guess = dmrg.psi0[ioff:ioff+neig,prjmap].copy()
         # Linear equation
         t_start = MPI.Wtime()
         eig,vec,nmvp = solver.solve_iter(v0=v0guess,iop=4)
         t_diff = MPI.Wtime() - t_start
      #
      # Save info on rank0
      #
      if rank == 0:
         civec = numpy.zeros((neig,ndim),dtype=dmrg_dtype)
         civec[:,prjmap] = vec
         civecs.append(civec)
	 eigs  += list(eig)
         nmvps += nmvp
         ioff  += neig  
         if dmrg.iprt >= 0: 
            print ' Eigenvalues = ',eig
	    print ' VectorNorms = ',map(lambda x:numpy.linalg.norm(x),vec)
  	    if dmrg_dtype == numpy.complex_:
	       print ' VectorReal2 = ',map(lambda x:x.dot(x),vec.real)
	       print ' VectorImag2 = ',map(lambda x:x.dot(x),vec.imag)
	    print ' Wtime for eigen problem of Qsym:%s = %.2f s'%(key,t_diff)
            # Check A-orthogonality for MPS-based CG algorithms
	    if dmrg.ifH0ortho and dmrg.nref>1:
	       print ' Check A-orthogonality for MPSPT2 =',numpy.dot(vec,pbas[1:].T)
   #
   # Final
   #
   eigsArray,sigs,dwts,qred,site,srotR = mpo_dmrg_dotutil.finalizeDot(info,civecs,eigs,actlst)
   return nmvps,eigsArray,sigs,dwts,qred,site,srotR
