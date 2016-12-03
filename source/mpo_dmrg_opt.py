#!/usr/bin/env python
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def sweep(dmrg,sitelst,ncsite,status,ifsym=False):
# def solver(dmrg,isite,ncsite,flst,status,ifsym):
# def sweep_checkOpersList(dmrg,nsite,sitelst,ncsite,status):
# def checkOpers(nsite,sitelst,ncsite,fnamel,fnamer,status,debug=False):
# def sweep_fileOpenOpers(dmrg,ncsite,isite):
# def sweep_fileOpenOpersN(dmrg,isite,ncsite,status):
# def sweep_fileCloseOpers(flsts):
# def sweep_updateSite(dmrg,isite,ncsite,sigs,qred,site,status):
# def sweep_updateBoundary(dmrg,ncsite,sitelst,qred,srotR,status):
# 
import shutil
import os
import h5py
import time
import numpy
import mpo_dmrg_io
import mpo_dmrg_prt
import mpo_dmrg_kernel
import mpo_dmrg_dot
import mpo_dmrg_dotutil
from qtensor import qtensor

# DMRG sweep optimization in one given direction
# We assume LMPS/RMPS and QNUML/QNUMR (ifsym) are available 
def sweep(dmrg,sitelst,ncsite,status,ifsym=False):
   ti = time.time()
   rank = dmrg.comm.rank
   if rank == 0: print '\n[mpo_dmrg_opt.sweep]'
   assert ncsite == 1 or ncsite == 2
   nsite = dmrg.nsite
   # Check the existence of files
   actlst = sweep_checkOpersList(dmrg,nsite,sitelst,ncsite,status)
   if rank == 0: 
      mpo_dmrg_prt.parameters(dmrg.Dmax,dmrg.crit_e,dmrg.noise,\
           	      	      ncsite,sitelst,actlst,status)
   # Start sweep
   elst = []
   dlst = []
   nmvp = 0
   dmrg.comm.Barrier()
   for isite in actlst:
      if rank == 0: mpo_dmrg_prt.block(nsite,isite,ncsite,status)
      # Open operator files
      flst = sweep_fileOpenOpers(dmrg,ncsite,isite)
      # Blocking and Solve local problem
      t0 = time.time()
      result = solver(dmrg,isite,ncsite,actlst,flst,status,ifsym)
      t1 = time.time()
      if rank ==0: print ' Time for CIsolver = %.2f s'%(t1-t0)
      # Save results
      nmvps,eigs,sigs,dwts,qred,site,srotR = result
      nmvp += nmvps
      elst.append(eigs)
      dlst.append(dwts)
      # Update block information
      siteN = sweep_updateSite(dmrg,isite,ncsite,sigs,qred,site,status)
      flstN = sweep_fileOpenOpersN(dmrg,isite,ncsite,status)
      # Renormalization of superblock
      t1 = time.time()
      mpo_dmrg_kernel.renorm(dmrg,isite,ncsite,flst,flstN,siteN,status)
      t2 = time.time()
      if rank ==0:
         print ' Time for renormal = %.2f s'%(t2-t1)
         print ' Time for allsteps = %.2f s'%(t2-t0)
      # Final
      sweep_fileCloseOpers(flst)
      sweep_fileCloseOpers(flstN)
   # Boundary cases
   sweep_updateBoundary(dmrg,ncsite,sitelst,qred,srotR,status)
   # Final
   tf = time.time()
   tfi = tf-ti
   indx,eav,dwt = mpo_dmrg_prt.singleSweep(dmrg,[elst,dlst],status,tfi)
   
   #> # Save previous MPS for restart
   #> if status == 'L':
   #>    dmrg.flmps.close()
   #>    shutil.copy(dmrg.path+'/lmps',dmrg.path+'/lmps0')
   #>    dmrg.flmps = h5py.File(dmrg.path+'/lmps','w')
   #>    dmrg.flmps['nsite'] = dmrg.nsite
   #>    mpo_dmrg_io.saveQnum(dmrg.flmps,0,dmrg.qnuml[0])
   #> elif status == 'R':
   #>    dmrg.frmps.close()
   #>    shutil.copy(dmrg.path+'/rmps',dmrg.path+'/rmps0')
   #>    dmrg.frmps = h5py.File(dmrg.path+'/rmps','w')
   #>    dmrg.frmps['nsite'] = dmrg.nsite
   #>    mpo_dmrg_io.saveQnum(dmrg.frmps,dmrg.nsite,dmrg.qnumr[-1])

   return nmvp,indx,eav,dwt,elst,dlst
 
# Solver
def solver(dmrg,isite,ncsite,actlst,flst,status,ifsym):
   rank = dmrg.comm.rank
   if rank == 0: 
      print '[mpo_dmrg_opt.solver] isweep =',dmrg.isweep
      print ' * isym/ifsym/s2proj =',\
	       (dmrg.isym,ifsym,dmrg.ifs2proj)
      print ' * ifQt/guess/precond =',\
	       (dmrg.ifQt,dmrg.ifguess,dmrg.ifprecond)
      print ' * ifex/ifpt/ifH0/nref =',\
	       (dmrg.ifex,dmrg.ifpt,dmrg.ifH0,dmrg.nref)
      print ' Check file lists:'
      print ' * flst0(lrop[0],lrpop[0]) =',flst[0]
      print ' * flst1(lrsop/lrpop[iex]) =',flst[1]
      print ' * flst2(lrhop[iex]) =',flst[2]
      print ' * flst3(lrdop[iex]) =',flst[3]
   if not dmrg.ifpt:
      result = mpo_dmrg_dot.ci_solver(dmrg,isite,ncsite,actlst,flst,status,ifsym)
   else:
      result = mpo_dmrg_dot.pt_solver(dmrg,isite,ncsite,actlst,flst,status,ifsym)
   return result

# Check the existence of operator files
def sweep_checkOpersList(dmrg,nsite,sitelst,ncsite,status):
   # Check file list
   fnamel = dmrg.path+'/lop'
   fnamer = dmrg.path+'/rop'
   actlst = checkOpers(nsite,sitelst,ncsite,fnamel,fnamer,status)
   if dmrg.comm.rank == 0 and dmrg.ifs2proj: 
      fnamel = dmrg.path+'/lpop'
      fnamer = dmrg.path+'/rpop'
      checkOpers(nsite,sitelst,ncsite,fnamel,fnamer,status)
   # Excited state cases
   if dmrg.ifex and dmrg.comm.rank == 0:
      for iref in range(dmrg.nref):
         if not dmrg.ifs2proj:
            fnamel = dmrg.path+'/ref'+str(iref)+'_lsop'
            fnamer = dmrg.path+'/ref'+str(iref)+'_rsop'
            checkOpers(nsite,sitelst,ncsite,fnamel,fnamer,status)
         else:
            fnamel = dmrg.path+'/ref'+str(iref)+'_lpop'
            fnamer = dmrg.path+'/ref'+str(iref)+'_rpop'
            checkOpers(nsite,sitelst,ncsite,fnamel,fnamer,status)
   # PT case	    
   if dmrg.ifpt:
      for iref in range(dmrg.nref):
         fnamel = dmrg.path+'/ref'+str(iref)+'_lhop'
         fnamer = dmrg.path+'/ref'+str(iref)+'_rhop'
         checkOpers(nsite,sitelst,ncsite,fnamel,fnamer,status)
      if dmrg.ifH0:	 
         for iref in range(dmrg.nref):
            fnamel = dmrg.path+'/ref'+str(iref)+'_ldop'
            fnamer = dmrg.path+'/ref'+str(iref)+'_rdop'
            checkOpers(nsite,sitelst,ncsite,fnamel,fnamer,status)
   return actlst

def checkOpers(nsite,sitelst,ncsite,fnamel,fnamer,status,debug=False):
   if len(sitelst)<ncsite:
      print 'error: len(sitelst) < ncsite!'
      exit(1)
   if status == 'L':
      # [**--]
      oplstL = [sitelst[0]-1]
      oplstR = range(sitelst[0]+ncsite,min(nsite,sitelst[-1]+ncsite)+1)
      # rank= 1  ncsite= 2  sitelst= [0, 1, 2, 3]  actlst= [0, 1, 2, 3]
      actlst = [i-ncsite for i in oplstR]
   elif status == 'R':
      # [--**]
      oplstL = range(max(-1,sitelst[0]-ncsite),sitelst[-1]-ncsite+1)
      oplstR = [sitelst[-1]+1]
      #
      # i+1 maitains the consistency for [j,j+1] to represent two-site configuration.
      # This is useful for deducing the dimension of CI space used in ci.Solver.
      #
      # rank= 1  ncsite= 2  sitelst= [0, 1, 2, 3]  actlst= [2, 1, 0]
      actlst = [i+1 for i in oplstL[-1::-1]]
   # Check operators
   for op in oplstL:
     path = fnamel+'_site_'+str(op)
     if not os.path.isfile(path):
        print 'error: operator file does not exist!',path
        exit(1)
   for op in oplstR:
     path = fnamer+'_site_'+str(op)
     if not os.path.isfile(path):
        print 'error: operator file does not exist!',path
        exit(1)
   if debug:
      print '[mpo_dmrg_opt.checkOpers]'
      print ' fnamel =',fnamel
      print ' fnamer =',fnamer
      print ' status =',status
      print ' sitelst=',sitelst
      print ' oplstL =',oplstL
      print ' oplstR =',oplstR
      print ' actlst =',actlst
   return actlst

# Open operator files
def sweep_fileOpenOpers(dmrg,ncsite,isite):
   # Collect
   flst0 = [0]*4
   fnameL = dmrg.path+'/lop'+'_site_'+str(isite-1)
   fnameR = dmrg.path+'/rop'+'_site_'+str(isite+ncsite)
   flst0[0] = h5py.File(fnameL,"r")
   flst0[1] = h5py.File(fnameR,"r")
   if dmrg.comm.rank == 0 and dmrg.ifs2proj: 
      fnameL = dmrg.path+'/lpop'+'_site_'+str(isite-1)
      fnameR = dmrg.path+'/rpop'+'_site_'+str(isite+ncsite)
      flst0[2] = h5py.File(fnameL,"r")
      flst0[3] = h5py.File(fnameR,"r")
   # Excited state case
   flst1 = [0]*(2*dmrg.nref)
   if dmrg.ifex and dmrg.comm.rank == 0:
      for iref in range(dmrg.nref):
         if not dmrg.ifs2proj: 
            fnameL = dmrg.path+'/ref'+str(iref)+'_lsop'+'_site_'+str(isite-1) 
            fnameR = dmrg.path+'/ref'+str(iref)+'_rsop'+'_site_'+str(isite+ncsite)
	    flst1[2*iref  ] = h5py.File(fnameL,"r")
	    flst1[2*iref+1] = h5py.File(fnameR,"r")
 	 else:
            fnameL = dmrg.path+'/ref'+str(iref)+'_lpop'+'_site_'+str(isite-1) 
            fnameR = dmrg.path+'/ref'+str(iref)+'_rpop'+'_site_'+str(isite+ncsite)
            flst1[2*iref  ] = h5py.File(fnameL,"r")
            flst1[2*iref+1] = h5py.File(fnameR,"r")
   # PT case
   flst2 = [0]*(2*dmrg.nref)
   flst3 = [0]*(2*dmrg.nref)
   if dmrg.ifpt:
      for iref in range(dmrg.nref):
         fnameL = dmrg.path+'/ref'+str(iref)+'_lhop'+'_site_'+str(isite-1)
         fnameR = dmrg.path+'/ref'+str(iref)+'_rhop'+'_site_'+str(isite+ncsite)
         flst2[2*iref  ] = h5py.File(fnameL,"r")
         flst2[2*iref+1] = h5py.File(fnameR,"r")
      if dmrg.ifH0:	 
         for iref in range(dmrg.nref):
            fnameL = dmrg.path+'/ref'+str(iref)+'_ldop'+'_site_'+str(isite-1)
            fnameR = dmrg.path+'/ref'+str(iref)+'_rdop'+'_site_'+str(isite+ncsite)
            flst3[2*iref  ] = h5py.File(fnameL,"r")
            flst3[2*iref+1] = h5py.File(fnameR,"r")
   flst = [flst0,flst1,flst2,flst3]
   return flst

# Open file for renormalization
def sweep_fileOpenOpersN(dmrg,isite,ncsite,status):
   # Collect
   flst0 = [0]*2
   if status == 'L':

      fnameN = dmrg.path+'/lop'+'_site_'+str(isite)
      flst0[0] = h5py.File(fnameN,"w")
      if dmrg.comm.rank == 0 and dmrg.ifs2proj:
	 fnameN = dmrg.path+'/lpop'+'_site_'+str(isite)
         flst0[1] = h5py.File(fnameN,"w")
      # Excited state case
      flst1 = [0]*dmrg.nref
      if dmrg.ifex and dmrg.comm.rank == 0:
         for iref in range(dmrg.nref):
	    if not dmrg.ifs2proj:
	       fnameN = dmrg.path+'/ref'+str(iref)+'_lsop'+'_site_'+str(isite) 
               flst1[iref] = h5py.File(fnameN,"w")
            else:
               fnameN = dmrg.path+'/ref'+str(iref)+'_lpop'+'_site_'+str(isite) 
               flst1[iref] = h5py.File(fnameN,"w")
      # PT case
      flst2 = [0]*dmrg.nref
      flst3 = [0]*dmrg.nref
      if dmrg.ifpt:
         for iref in range(dmrg.nref):
            fnameN = dmrg.path+'/ref'+str(iref)+'_lhop'+'_site_'+str(isite)
            flst2[iref] = h5py.File(fnameN,"w")
	 if dmrg.ifH0:
	    for iref in range(dmrg.nref):
               fnameN = dmrg.path+'/ref'+str(iref)+'_ldop'+'_site_'+str(isite)
               flst3[iref] = h5py.File(fnameN,"w")
 
   elif status == 'R':

      jsite = isite+ncsite-1
      fnameN = dmrg.path+'/rop'+'_site_'+str(jsite)
      flst0[0] = h5py.File(fnameN,"w")
      if dmrg.comm.rank == 0 and dmrg.ifs2proj:
         fnameN = dmrg.path+'/rpop'+'_site_'+str(jsite)
         flst0[1] = h5py.File(fnameN,"w")
      # Excited state case
      flst1 = [0]*dmrg.nref
      if dmrg.ifex and dmrg.comm.rank == 0:
         for iref in range(dmrg.nref):
	    if not dmrg.ifs2proj:
               fnameN = dmrg.path+'/ref'+str(iref)+'_rsop'+'_site_'+str(jsite) 
               flst1[iref] = h5py.File(fnameN,"w")
            else:
               fnameN = dmrg.path+'/ref'+str(iref)+'_rpop'+'_site_'+str(jsite) 
               flst1[iref] = h5py.File(fnameN,"w")
      # PT case
      flst2 = [0]*(dmrg.nref)
      flst3 = [0]*(dmrg.nref)
      if dmrg.ifpt:
         for iref in range(dmrg.nref):
            fnameN = dmrg.path+'/ref'+str(iref)+'_rhop'+'_site_'+str(jsite)
            flst2[iref] = h5py.File(fnameN,"w")
	 if dmrg.ifH0:
	    for iref in range(dmrg.nref):
               fnameN = dmrg.path+'/ref'+str(iref)+'_rdop'+'_site_'+str(jsite)
               flst3[iref] = h5py.File(fnameN,"w")

   flstN = [flst0,flst1,flst2,flst3]
   return flstN

def sweep_fileCloseOpers(flsts):
   for flst in flsts:
      for ifile in flst:
         if ifile != 0: ifile.close()
   return 0

# Update site
def sweep_updateSite(dmrg,isite,ncsite,sigs,qred,site,status):
   # Update symmetry and save sites
   if status == 'L':
      # Example:[==**-] isite=2, lmps[2], qnuml[3]
      dmrg.sigsl[isite+1] = sigs.copy()
      dmrg.qnuml[isite+1] = qred.copy()
      if dmrg.ifQt:
         qtmp = qtensor.qtensor([False,False,True])
         #				Ql[i-1]  +  Qn[i]  =  Ql[i]
	 qtmp.fromDenseTensor(site,[dmrg.qlst[0],dmrg.qlst[1],qred])
         siteN = qtmp
      else:
         siteN = site.copy()
      # DUMP
      mpo_dmrg_io.saveQnum(dmrg.flmps,isite+1,qred)
      mpo_dmrg_io.saveSite(dmrg.flmps,isite,siteN)
   elif status == 'R':
      # Example: [--**=] isite=2, jsite=3, rmps[3], qnumr[3]
      jsite = isite+ncsite-1
      dmrg.sigsr[jsite] = sigs.copy()
      dmrg.qnumr[jsite] = qred.copy()
      if dmrg.ifQt:
         qtmp = qtensor.qtensor([True,False,False])
         #		          Qr[i-1]  =  Qn[i]   +   Qr[i]
         qtmp.fromDenseTensor(site,[qred,dmrg.qlst[-2],dmrg.qlst[-1]])
         siteN = qtmp
      else:
	 siteN = site.copy()
      # DUMP
      mpo_dmrg_io.saveQnum(dmrg.frmps,jsite,qred)
      mpo_dmrg_io.saveSite(dmrg.frmps,jsite,siteN)
   return siteN

# Boundary Sites in twoSite sweeps using qred,srotR
def sweep_updateBoundary(dmrg,ncsite,sitelst,qred,srotR,status):
   if ncsite == 2:
      if status == 'L' and sitelst[-1] == dmrg.nsite-1: 
	 # >>> Only the gs state is saved for generating MPS.
	 smat = srotR[0]
         norm = numpy.linalg.norm(smat)
	 if (not dmrg.ifpt) or (dmrg.ifpt and dmrg.ifcompression): smat = smat/norm
	 key  = mpo_dmrg_dotutil.floatKey(dmrg.qsectors.keys()[0])
	 qsym = [numpy.array(eval(key))]
	 if dmrg.ifQt:
   	    qtmp = qtensor.qtensor([False,False,True])
	    qtmp.fromDenseTensor(smat,[qred,dmrg.qphys[dmrg.nsite-1],qsym])
	    smat = qtmp
	 # DUMP
	 mpo_dmrg_io.saveQnum(dmrg.flmps,dmrg.nsite,qsym)
	 mpo_dmrg_io.saveSite(dmrg.flmps,dmrg.nsite-1,smat)
      elif status == 'R' and sitelst[0] == 0:
	 # >>> Only the gs state is saved for generating MPS.
         smat = srotR[0]
         norm = numpy.linalg.norm(smat)
	 if (not dmrg.ifpt) or (dmrg.ifpt and dmrg.ifcompression): smat = smat/norm
	 key  = mpo_dmrg_dotutil.floatKey(dmrg.qsectors.keys()[0])
	 qsym = [numpy.array(eval(key))]
	 if dmrg.ifQt:
   	    qtmp = qtensor.qtensor([True,False,False])
	    qtmp.fromDenseTensor(smat,[qsym,dmrg.qphys[0],qred])
  	    smat = qtmp	
	 # DUMP
	 mpo_dmrg_io.saveQnum(dmrg.frmps,0,qsym)
	 mpo_dmrg_io.saveSite(dmrg.frmps,0,smat)
   return 0
