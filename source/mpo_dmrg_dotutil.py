#!/usr/bin/env python
#
# Utils for solving local [dot] CI/PT problems
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def initializeDPTspace(dmrg,isite,ncsite,flst,ifsym):
# def genDimensions(dmrg,isite,ncsite,fL,fR):
# def floatKey(key):
# def setupQlst(dmrg,isite,ncsite,ifsym):
# def dptSymmetry(dmrg,isite,ncsite,ifsym,debug=False):
#
# def symmetrySpaceInfo(dmrg,qkey,ncsite,dicDPT=None):
# def setupQtmpCI(dmrg,ncsite,key):
#
# def genHDiag(info,ndim0,prjmap):
# def genRHSpt(info,ndim0,prjmap,iHd=0):
#
# def finalizeDot(info,civecs0,eigs0):
# def decimation(civecs,info,debug=False):
# def genGuess(dmrg,isite,ncsite,civecs,srotR,status):
#
# def setupQtmpPRDM(dmrg,ncsite,key,status):
# def vonNeumannEntropy(sigs,thresh=1.e-10):
#
import copy
import time
import numpy
from mpi4py import MPI
import mpo_dmrg_io
import mpo_dmrg_qphys
import mpo_dmrg_kernel
import mpo_dmrg_qparser
from qtensor import qtensor
from qtensor import qtensor_util
from sysutil_include import dmrg_dtype,dmrg_mtype

# For dot optimization
def initializeDPTspace(dmrg,isite,ncsite,flst,ifsym):
   fL,fR = flst[0][0:2]
   dmrg.dims = genDimensions(dmrg,isite,ncsite,fL,fR)
   # Reduction of direct product decomposition for product space
   dicDPT = None
   if dmrg.ifQt:
      setupQlst(dmrg,isite,ncsite,ifsym)
   else:
      dicDPT = dptSymmetry(dmrg,isite,ncsite,ifsym)
   return dicDPT

# Dimensions for direct product space, assuming operators are available.
def genDimensions(dmrg,isite,ncsite,fL,fR):
   if ncsite == 1:
      cdim = dmrg.dphys[isite]
      # [*===] or [===*]	 
   elif ncsite == 2:
      cdim = dmrg.dphys[isite]*dmrg.dphys[isite+1]
      # [**==] or [==**]	 
   if dmrg.ifQt:
      ldim = 0
      rdim = 0
      for name in fL:
	 if 'slc' in name:
	    ldim = fL[name].attrs['shape'][2]
	    break
      for name in fR:
	 if 'slc' in name:
	    rdim = fR[name].attrs['shape'][2]
	    break
      if ldim == 0 or rdim == 0: 
	 print 'error: ldim/rdim == 0!',ldim,rdim
	 exit(1)
   else:	  
      ldim = fL['opers0'].shape[2]
      rdim = fR['opers0'].shape[2]
   ndim = ldim*cdim*rdim
   return ldim,cdim,rdim,ndim

# Setup Qlst for latter usage
def setupQlst(dmrg,isite,ncsite,ifsym):
   assert ifsym == True
   qsyml = dmrg.qnuml[isite]
   if ncsite == 1:
      qsymc = dmrg.qphys[isite]
      qsymr = dmrg.qnumr[isite+1]
      # This information is essential for latter usages of Qt
      dmrg.qlst = [numpy.array(qsyml),numpy.array(qsymc),numpy.array(qsymr)]
   elif ncsite == 2:
      qsymc1 = dmrg.qphys[isite]
      qsymc2 = dmrg.qphys[isite+1]
      qsymc = mpo_dmrg_qphys.dpt(qsymc1,qsymc2)
      qsymr = dmrg.qnumr[isite+2]
      dmrg.qlst = [numpy.array(qsyml),numpy.array(qsymc1),numpy.array(qsymc2),numpy.array(qsymr)]
   return 0 

# Float format for keys
def floatKey(key):
   fkey = str(map(lambda x:float(x),eval(key)))
   return fkey

# Symmetry-adapted CI space for NQt case.
def dptSymmetry(dmrg,isite,ncsite,ifsym,debug=False):
   ldim,cdim,rdim,ndim = dmrg.dims
   # case: NOSYM 
   if not ifsym or (ifsym and dmrg.isym == 0):
      prjmap = numpy.array(range(ndim))
      dic0 = {}
      for key in dmrg.qsectors:
         dic0[key] = prjmap.copy()
   # case: Symmetry 
   else:
      # L->R: ==*--- or ==**--
      # R->L: --*=== or --**==
      qsyml = dmrg.qnuml[isite]
      if ncsite == 1:
         qsymc = dmrg.qphys[isite]
         qsymr = dmrg.qnumr[isite+1]
         dmrg.qlst = [numpy.array(qsyml),numpy.array(qsymc),numpy.array(qsymr)]
      elif ncsite == 2:
         qsymc1 = dmrg.qphys[isite]
	 qsymc2 = dmrg.qphys[isite+1]
	 qsymc = mpo_dmrg_qphys.dpt(qsymc1,qsymc2)
 	 qsymr = dmrg.qnumr[isite+2]
         dmrg.qlst = [numpy.array(qsyml),numpy.array(qsymc1),numpy.array(qsymc2),numpy.array(qsymr)]
      qsymt = reduce(mpo_dmrg_qphys.dpt,(qsyml,qsymc,qsymr))
      assert len(qsyml) == ldim
      assert len(qsymc) == cdim
      assert len(qsymr) == rdim
      assert len(qsymt) == ndim
      # Dictionary
      dic0 = {}
      for idx,item in enumerate(qsymt):
	 dic0.setdefault(str(item),[]).append(idx)
   dic = {}
   for key in dic0:
      dic[floatKey(key)] = numpy.array(dic0[key])
   if debug:
      print '[mpo_dmrg_util.dptSymmetry] (ldim,cdim,rdim,ndim) = ',(ldim,cdim,rdim,ndim)
      print ' dic =',[(key,len(dic[key])) for key in dic]
      lst = sorted(map(lambda x:eval(x),dic.keys()))
      for ikey in lst:
         key = str(ikey)
         print ' key/dim =',key,len(dic[key])
   return dic

# prjmap and ndim0
def symmetrySpaceInfo(dmrg,qkey,ncsite,dicDPT=None):
   key = floatKey(qkey)
   neig = dmrg.qsectors[qkey]
   # The only essential thing is prjmap, which will be used in civec->CIMAT.
   if dmrg.ifQt:
      # Setup Qtensor template information
      setupQtmpCI(dmrg,ncsite,key)
      prjmap = dmrg.qtmp.prjmap(dmrg.idlstCI)
      ndim0  = dmrg.qtmp.size_allowed
   else:

      try:
         prjmap = dicDPT[key]
      except KeyError:
         print 'error: No basis for key=',key
         print 'keys = ',dicDPT.keys()
         exit(1)
      ndim0  = len(prjmap)
     
#>    For testing purpose - 2016.10.21 - precursor for tamps     
#>      #------------------------
#>      ndim0 = dmrg.dims[-1]
#>      prjmap = range(ndim0)
#>      #print 'dicDPT=',[(key,len(dicDPT[key])) for key in dicDPT]
#>    
#>      print
#>      prjmap = []
#>      nk = eval(qkey)[0]
#>      for ky in dicDPT:
#>	 nd = eval(ky)[0]
#>	 if abs(nk-nd)<1.e-8:
#>	    prjmap += list(dicDPT[ky])
#>	    print ' key=',ky,len(dicDPT[ky])
#>      ndim0 = len(prjmap)
#>      print
#>      #------------------------

   if ndim0 == 0:
      print 'error: No enough basis in the symmetry sector !'
      print ' qkey =',qkey
      exit(1)
   return key,neig,ndim0,prjmap

# Setup Qtmp for CI
def setupQtmpCI(dmrg,ncsite,key):
   dmrg.qkey = numpy.array(eval(key))
   if ncsite == 1:
      qnums = [dmrg.qlst[0],dmrg.qlst[1],dmrg.qkey-dmrg.qlst[2]]
      dmrg.qtmp = qtensor.qtensor([False,False,True])
   elif ncsite == 2:
      qnums = [dmrg.qlst[0],dmrg.qlst[1],dmrg.qlst[2],dmrg.qkey-dmrg.qlst[3]]
      dmrg.qtmp = qtensor.qtensor([False,False,False,True])
   rank,qsyms,ndims,idlst = qtensor_util.fromQnums(qnums)
   dmrg.qtmp.fromQsyms(rank,qsyms,ndims)
   dmrg.idlstCI = copy.deepcopy(idlst)
   return 0

# Diag
def genHDiag(info,ndim0,prjmap):
   dmrg = info[0]
   rank = dmrg.comm.rank
   if not dmrg.ifprecond:
      Diag = numpy.ones(ndim0,dtype=numpy.float_)
   else:
      t_start = MPI.Wtime()
      # Local diag
      t0 = time.time()
      Diag_iproc = mpo_dmrg_kernel.HDiag(info,ndim0,prjmap)
      t1 = time.time()
      if dmrg.iprt > 0: print ' Time for diagonal = %.2f s'%(t1-t0),' rank=',rank
      # Sum of them
      Diag = numpy.zeros(ndim0,dtype=numpy.float_)
      dmrg.comm.Reduce([Diag_iproc,MPI.DOUBLE],
                       [Diag,MPI.DOUBLE],
                       op=MPI.SUM,root=0)
      # No need to broadcast, since only rank=0 execute preconditioning
      dmrg.comm.Barrier()
      t_diff = MPI.Wtime() - t_start
      if rank == 0 and dmrg.iprt >= 0: print ' Wtime for diagonal = %.2f s'%t_diff
      # Check 
      if dmrg.iprt > 1:
         print ' rank=',rank,' checkSum=',numpy.sum(Diag_iproc)
         if rank==0: print ' total checkSum=',numpy.sum(Diag)
   return Diag

# RHS of PT equation: <Psi(1)|(H-E0)|chi[m]>c[m]
def genRHSpt(info,ndim0,prjmap,iHd=0):
   dmrg = info[0]
   rank = dmrg.comm.rank
   t_start = MPI.Wtime()
   # Local diag
   t0 = time.time()
   BVec_iproc = mpo_dmrg_kernel.BVec(info,ndim0,prjmap,iHd)
   t1 = time.time()
   if dmrg.iprt > 0: print ' Time for BVec = %.2f s'%(t1-t0),' rank=',rank
   # Depending on iHd, return different values
   if iHd == 0:
      BVec = numpy.zeros(ndim0,dtype=dmrg_dtype)
   elif iHd == 1:
      BVec = numpy.zeros((dmrg.nref,ndim0),dtype=dmrg_dtype)
   # Sum of them
   dmrg.comm.Reduce([BVec_iproc,dmrg_mtype],
                    [BVec,dmrg_mtype],
                    op=MPI.SUM,root=0)
   # No need to broadcast, since only rank=0 execute preconditioning
   dmrg.comm.Barrier()
   t_diff = MPI.Wtime() - t_start
   if rank == 0 and dmrg.iprt >= 0: print ' Wtime for BVec = %.2f s'%t_diff,' iHd =',iHd
   return BVec

# Final
def finalizeDot(info,civecs0,eigs0,actlst):
   dmrg  = info[0]
   isite = info[1]
   ncsite= info[2]
   status= info[4]
   rank = dmrg.comm.rank
   # BroadCast final results for updating blocks
   if rank == 0:
      civecs = numpy.vstack(civecs0)
      eigs = numpy.array(eigs0).real # Assuming real eigenvalues
      cshape = civecs.shape 
      eshape = eigs.shape
   else:
      cshape = None
      eshape = None
   cshape = dmrg.comm.bcast(cshape)
   eshape = dmrg.comm.bcast(eshape)
   if rank !=0:
      civecs = numpy.zeros(cshape,dtype=dmrg_dtype)
      eigs   = numpy.zeros(eshape,dtype=numpy.float_)
   dmrg.comm.Bcast([civecs,dmrg_mtype])
   dmrg.comm.Bcast([eigs  ,MPI.DOUBLE])
   # Decimation
   sigs,dwts,qred,site,srotR = decimation(civecs,info)
   # Generate initial guess for the next optimization subproblem
   if rank==0 and dmrg.ifguess: genGuess(dmrg,isite,ncsite,actlst,civecs,srotR,status)
   return eigs,sigs,dwts,qred,site,srotR

# Decimation via diagonalization of RDM
def decimation(civecs,info,debug=False):
   ti = time.time()
   dmrg,isite,ncsite,flst,status,ifsym = info
   if dmrg.iprt > 0: print '[mpo_dmrg_ci.decimation]'
   rank = dmrg.comm.rank
   ldim,cdim,rdim,ndim = dmrg.dims
   neig = civecs.shape[0]
   if status == 'L':
      dim = dmrg.dphys[isite]
      cdimc = cdim/dim 
      lcdim = ldim*dim
      crdim = cdimc*rdim
      dimSuperBlock = lcdim
      civecs = civecs.reshape(neig,ldim,dim,crdim)
   elif status == 'R':
      jsite = isite+ncsite-1
      dim = dmrg.dphys[jsite]
      cdimc = cdim/dim 
      lcdim = ldim*cdimc
      crdim = dim*rdim
      dimSuperBlock = crdim
      civecs = civecs.reshape(neig,lcdim,dim,rdim)
   # RDM for superblock 
   rdm1 = numpy.zeros((dimSuperBlock,dimSuperBlock),dtype=dmrg_dtype)
   # Add pRDM
   if dmrg.inoise == 0 and dmrg.noise >= 1.e-10:
      t_start = MPI.Wtime()
      # Local pRDM
      t0 = time.time()
      rdm1_iproc = mpo_dmrg_kernel.pRDM(civecs,info)
      t1 = time.time()
      if dmrg.iprt > 0: print ' Time for pRDM = %.2f s'%(t1-t0),' rank=',rank
      # Sum of them
      dmrg.comm.Reduce([rdm1_iproc,dmrg_mtype],
                       [rdm1,dmrg_mtype],
                       op=MPI.SUM,root=0)
      # No need to broadcast
      dmrg.comm.Barrier()
      t_diff = MPI.Wtime() - t_start
      if rank == 0 and dmrg.iprt >= 0: 
         print ' Wtime for pRDM = %.2f s'%t_diff,'with noise =',dmrg.noise
      # Check 
      if dmrg.iprt > 1:
         print ' rank=',rank,' checkSum=',numpy.sum(rdm1_iproc)
         if rank==0: print ' total checkSum=',numpy.sum(rdm1)
   else:
      # Simple noise
      if rank == 0 and dmrg.inoise == 1 and dmrg.noise >= 1.e-10: 
	 civecs += numpy.random.uniform(-1,1,size=civecs.shape)*10.0*dmrg.noise
         print ' Simple random noise with size (for civecs) =',dmrg.noise*10.0 # due to square
      # Only rank-0 will add RDM
      rdm1 = mpo_dmrg_kernel.pRDM(civecs,info)
   #
   # Only rank-0 performs decimation.
   #
   if rank == 0:

      # 1. Always normalize the RDM0 
      trRDM = numpy.trace(rdm1)
      threshRDM = 1.e-12
      if trRDM < -threshRDM: 
	 print 'error: RDM should be positive definite! trRDM=',trRDM
	 exit(1)
      # if |rdm|<eps, use identity like rdm1. 
      elif trRDM > -threshRDM and trRDM < threshRDM:
	 rdm1_dim = rdm1.shape[0]
	 rdm1 = rdm1 + numpy.identity(rdm1_dim)/float(rdm1_dim)
      else:
	 rdm1 = rdm1/trRDM

      # 2. Setup cutoff 
      Dcut = dmrg.Dmax
      # To make the rank2-to-rank3 change works correctly,
      # cut at the boundary of one site sweeps: L->R:====* and R->L:*====.
      if ncsite == 1:
         if (status == 'L' and isite == dmrg.nsite-1) or \
            (status == 'R' and isite == 0): 
            # Simply cut to one state only, the same is done for 
            # two site case in [lastSite]. This does not affect 
            # iterations, since the boundary sites are restarted.
            Dcut = 1 #dmrg.neig

      # 3. Perform decimation
      if status == 'L':

	 if dmrg.Dcut is not None:
	    Dcut = dmrg.Dcut[isite+1]
	    print ' dmrg.Dcut =',dmrg.Dcut
         if dmrg.isym > 0:
            qsyml  = dmrg.qnuml[isite]
            qsymc  = dmrg.qphys[isite]
            qsymlc = mpo_dmrg_qphys.dpt(qsyml,qsymc)
            if debug:
               print ' qsyml  =',qsyml
               print ' qsymc  =',qsymc
               print ' qsymlc =',qsymlc
            classes = copy.deepcopy(qsymlc)
         else:
            classes = [0]*lcdim
	 # Calculate renormalized basis   
	 if dmrg.ifsci and isite != dmrg.nsite-1 and dmrg.ncsite == 2:
 	    dwts,qred,rotL,sigs = mpo_dmrg_qparser.rdm_sci(rdm1,classes,dmrg.trsci,Dcut=-1,debug=False)
  	 else:
 	    dwts,qred,rotL,sigs = mpo_dmrg_qparser.rdm_blkdiag(rdm1,classes,dmrg.thresh,Dcut,debug)
         nres = len(qred)
         site = rotL.reshape(ldim,dim,nres).copy()
         # W[a,n',r] = U[(l,n),a]*C[(l,n),n',r]
	 srotR = numpy.empty((neig,nres,cdimc,rdim),dtype=dmrg_dtype) # Ij,aIX->ajX
	 for ieig in range(neig):
	    cimat = civecs[ieig].reshape(lcdim,crdim)
	    # C=U*s*Vd, rhoL=C*Cd=U*s2*Ud, rotL=U => s*Vd=Ud*C  
	    srotR[ieig] = numpy.dot(rotL.T.conj(),cimat).reshape(nres,cdimc,rdim)

      elif status == 'R':

	 if dmrg.Dcut is not None:
            jsite = isite+ncsite-1
	    Dcut = dmrg.Dcut[jsite]
	    print ' dmrg.Dcut =',dmrg.Dcut
         if dmrg.isym > 0:
            jsite = isite+ncsite-1
            qsymr = dmrg.qnumr[jsite+1]
            qsymc = dmrg.qphys[jsite]
            qsymcr = mpo_dmrg_qphys.dpt(qsymc,qsymr)
            if debug:
               print ' qsymc  =',qsymc
               print ' qsymr  =',qsymr
               print ' qsymcr =',qsymcr
            classes = copy.deepcopy(qsymcr)
         else:
            classes = [0]*crdim
	 # Calculate renormalized basis   
	 if dmrg.ifsci and isite != 0 and dmrg.ncsite == 2:
 	    dwts,qred,rotL,sigs = mpo_dmrg_qparser.rdm_sci(rdm1,classes,dmrg.trsci,Dcut=-1,debug=False)
  	 else:
 	    dwts,qred,rotL,sigs = mpo_dmrg_qparser.rdm_blkdiag(rdm1,classes,dmrg.thresh,Dcut,debug)
	 nres = len(qred)
	 # rotL=V => [V*].T => [Vd](nres,dim*rdim)
	 site = rotL.conj().reshape(dim,rdim,nres).transpose(2,0,1).copy()
         # W[l,n,a] = C[l,n,(n',r)]*V[a,(n',r)]
	 srotR = numpy.empty((neig,ldim,cdimc,nres),dtype=dmrg_dtype) # aXI,Ij->aXj
	 for ieig in range(neig):
	    cimat = civecs[ieig].reshape(lcdim,crdim)
	    # C=U*s*Vd, rhoR=Cd*C=V*s2*Vd, rotL=V => U*s=C*V  
	    srotR[ieig] = numpy.dot(cimat,rotL).reshape(ldim,cdimc,nres)

      # Check	
      if debug:
         print ' CImat.shape[neig,L,R] = ',civecs.shape 
         print ' Truncated sigs :',sigs.shape,' range = (%12.8e,%12.8e)'%(numpy.amax(sigs),numpy.amin(sigs))
         print ' Site[i] =',isite,' shape =',site.shape
         print ' Sum of sigs =',numpy.sum(sigs)
         print ' Discarded weights =',dwts
      else:
	 print ' Dcut =%5d  RDM-based decimation: %5d =>%5d  dwts = %7.2e'%(Dcut,dimSuperBlock,nres,dwts)
	 tmpsigs = -numpy.sort(-sigs)
	 nsigprt = 4
	 sigs_frst = tmpsigs[:nsigprt]
	 sigs_last = tmpsigs[-1:-nsigprt-1:-1][-1::-1]
	 print ' first %d sigs2 = '%nsigprt,sigs_frst
	 print ' last  %d sigs2 = '%nsigprt,sigs_last
	 print ' von Neumann entropy =',vonNeumannEntropy(tmpsigs),\
	       ' theoretical maxS =',numpy.log(len(tmpsigs))	

      # Store the norm of wavefunction at the boundary for one-site case.
      if dmrg.ifpt and ncsite == 1:
         if (status == 'L' and isite == dmrg.nsite-1) or \
            (status == 'R' and isite == 0): 
	    assert civecs.shape[0] == 1
	    # |ref><ref|x>
	    ovlp = numpy.tensordot(site,civecs[0],axes=([0,1,2],[0,1,2]))
	    site = site*ovlp
	 
      # Prepare communication
      qred = numpy.array(qred)
      dwts = numpy.array(dwts) 
      qshape = qred.shape
      dshape = dwts.shape
      sshape = sigs.shape
      ushape = site.shape
      vshape = srotR.shape
   # Other ranks
   else:
      qshape = None
      dshape = None
      sshape = None
      ushape = None
      vshape = None
   # Bcast   
   qshape = dmrg.comm.bcast(qshape)
   dshape = dmrg.comm.bcast(dshape)
   sshape = dmrg.comm.bcast(sshape)
   ushape = dmrg.comm.bcast(ushape)
   vshape = dmrg.comm.bcast(vshape)
   if rank != 0:
      qred  = numpy.zeros(qshape,dtype=numpy.float_) 
      dwts  = numpy.zeros(dshape,dtype=numpy.float_) 
      sigs  = numpy.zeros(sshape,dtype=numpy.float_)
      site  = numpy.zeros(ushape,dtype=dmrg_dtype)
      srotR = numpy.zeros(vshape,dtype=dmrg_dtype)
   dmrg.comm.Bcast([qred ,MPI.DOUBLE])
   dmrg.comm.Bcast([dwts ,MPI.DOUBLE])
   dmrg.comm.Bcast([sigs ,MPI.DOUBLE])
   dmrg.comm.Bcast([site ,dmrg_mtype])
   dmrg.comm.Bcast([srotR,dmrg_mtype])
   tf = time.time()
   if dmrg.comm.rank == 0: print ' Wtime for decimation = %.2f s'%(tf-ti)
   return sigs,dwts,qred,site,srotR

# Generate initial guess: Only rank0 generates and uses guess later. 
def genGuess(dmrg,isite,ncsite,actlst,civecs,srotR,status):
   if ncsite == 2:
      # R: --**, -**=, **==
      if status == 'R': 
	 # Boundary case - simply copy for the left sweep 
	 if isite == actlst[-1]: 
	    dmrg.psi0 = civecs.copy()
         elif 'site'+str(isite-1) in dmrg.flmps:
	    for ieig in range(dmrg.neig):
	       # Move (L*Vs)*U	    
	       #tmp = numpy.einsum('abc,cde->abde',dmrg.lmps[isite-1],srotR[ieig])
	       tmp = mpo_dmrg_io.loadSite(dmrg.flmps,isite-1,dmrg.ifQt) 
	       if dmrg.ifQt: tmp = tmp.toDenseTensor(tmp.idlst)
	       tmp = numpy.tensordot(tmp,srotR[ieig],axes=([2],[0]))
	       s = numpy.prod(tmp.shape)
	       if ieig == 0: dmrg.psi0 = numpy.zeros((dmrg.neig,s),dtype=dmrg_dtype)
	       dmrg.psi0[ieig] = tmp.reshape(s).copy()
      # L: **--, =**-, ==**
      elif status == 'L':
	 if isite == actlst[-1]:
	    dmrg.psi0 = civecs.copy()
         elif 'site'+str(isite+2) in dmrg.frmps:
            for ieig in range(dmrg.neig):
	       # Move U*(sV*R)	    
	       #tmp = numpy.einsum('abc,cde->abde',srotR[ieig],dmrg.rmps[isite+2])
	       tmp = mpo_dmrg_io.loadSite(dmrg.frmps,isite+2,dmrg.ifQt) 
	       if dmrg.ifQt: tmp = tmp.toDenseTensor(tmp.idlst)
	       tmp = numpy.tensordot(srotR[ieig],tmp,axes=([2],[0]))
	       s = numpy.prod(tmp.shape)
	       if ieig == 0: dmrg.psi0 = numpy.zeros((dmrg.neig,s),dtype=dmrg_dtype)
	       dmrg.psi0[ieig] = tmp.reshape(s).copy()
   elif ncsite == 1:
      # R: ---*, --*=, -*==, *===
      if status == 'R':
	 # Boundary case
	 if isite == actlst[-1]: 
	    dmrg.psi0 = civecs.copy()
         elif 'site'+str(isite-1) in dmrg.flmps:
	    for ieig in range(dmrg.neig):
	       # Move (L*Vs)*U	    
	       #tmp = numpy.einsum('abc,cde->abde',dmrg.lmps[isite-1],srotR[ieig])
	       tmp = mpo_dmrg_io.loadSite(dmrg.flmps,isite-1,dmrg.ifQt) 
	       if dmrg.ifQt: tmp = tmp.toDenseTensor(tmp.idlst)
	       tmp = numpy.tensordot(tmp,srotR[ieig],axes=([2],[0]))
	       s = numpy.prod(tmp.shape)
	       if ieig == 0: dmrg.psi0 = numpy.zeros((dmrg.neig,s),dtype=dmrg_dtype)
	       dmrg.psi0[ieig] = tmp.reshape(s).copy()
      # L: *---, =*--, ==*-, ===*
      elif status == 'L':
	 if isite == actlst[-1]:
	    dmrg.psi0 = civecs.copy()
         elif 'site'+str(isite+1) in dmrg.frmps:
            for ieig in range(dmrg.neig):
	       # Move U*(sV*R)	    
	       #tmp = numpy.einsum('abc,cde->abde',srotR[ieig],dmrg.rmps[isite+1])
	       tmp = mpo_dmrg_io.loadSite(dmrg.frmps,isite+1,dmrg.ifQt) 
	       if dmrg.ifQt: tmp = tmp.toDenseTensor(tmp.idlst)
	       tmp = numpy.tensordot(srotR[ieig],tmp,axes=([2],[0]))
	       s = numpy.prod(tmp.shape)
	       if ieig == 0: dmrg.psi0 = numpy.zeros((dmrg.neig,s),dtype=dmrg_dtype)
	       dmrg.psi0[ieig] = tmp.reshape(s).copy()
   return 0

# Setup Qtmp for pRDM
def setupQtmpPRDM(dmrg,ncsite,key,status):
   dmrg.qkey = numpy.array(eval(key))
   if ncsite == 1:
      qsymr = dmrg.qkey-dmrg.qlst[2]
      qnums = [dmrg.qlst[0],dmrg.qlst[1],qsymr]
      dmrg.qtmp = qtensor.qtensor([False,False,True])
   elif ncsite == 2:
      if status == 'L':
         qsymr = mpo_dmrg_qphys.dpt(dmrg.qlst[2],dmrg.qlst[3])
         qsymr = dmrg.qkey - qsymr
         qnums = [dmrg.qlst[0],dmrg.qlst[1],qsymr]
      elif status == 'R':
         qsyml = mpo_dmrg_qphys.dpt(dmrg.qlst[0],dmrg.qlst[1])
         qsymr = dmrg.qkey-dmrg.qlst[3]
         qnums = [qsyml,dmrg.qlst[2],qsymr]
      dmrg.qtmp = qtensor.qtensor([False,False,True])
   rank,qsyms,ndims,idlst = qtensor_util.fromQnums(qnums)
   dmrg.qtmp.fromQsyms(rank,qsyms,ndims)
   dmrg.idlstCI = copy.deepcopy(idlst)
   # For pRDM:
   if status == 'L':
      qnums2 = [qnums[0],qnums[1],qnums[0],qnums[1]]
   elif status == 'R':
      qnums2 = [qnums[1],qnums[2],qnums[1],qnums[2]]
   rank,qsyms,ndims,idlst = qtensor_util.fromQnums(qnums2)
   dmrg.idlstPRDM = copy.deepcopy(idlst)
   return 0

# S = \sum_i -si^2*log(si^2)
def vonNeumannEntropy(sigs,thresh=1.e-12):
   ssum = 0.
   for sig2 in sigs:
     if sig2 < thresh: continue
     ssum += -sig2*numpy.log(sig2)
   return ssum
