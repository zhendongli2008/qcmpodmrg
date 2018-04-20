#!/usr/bin/env python
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def copyMPS(fmps1,fmps0,ifQt):
# def dumpMPS(fmps,mpslst,icase=0):
# def loadMPS(fmps,icase=0):
# def saveQnum(fmps,isite,qred):
# def loadQnums(fmps):
# def saveSite(fmps,isite,site):
# def loadSite(fmps,isite,ifQt):
#
# def dumpMPO_H(dmrg,debug=False):
# def dumpMPO_R(dmrg,debug=False):
# def dumpMPO_H0(dmrg,debug=False):
# def dumpMPO_Hubbard(dmrg,debug=False):
# def dumpMPO_H1e(dmrg,int1e,debug=False):
#
# def finalMPS(dmrg):   
# def finalMPO(dmrg):
#
# def loadInts(dmrg,mol):
# 
import h5py
import time
import numpy
import mpo_dmrg_opers
import mpo_dmrg_ptopers
from misc import mpo_dmrg_model
from qtensor import qtensor
from qtensor import qtensor_opers
from sysutil_include import dmrg_dtype,dmrg_mtype 
import mpo_dmrg_opers1e
from qtensor import qtensor_opers1e

#
# Structure of MPS file:
# f/nsite
# f/qnum0,f/qnum1,...
# f/site0,f/site1,...
#

# copyMPS from fmps0 to fmps1
def copyMPS(fmps1,fmps0,ifQt):
   nsite = fmps0['nsite'].value
   print '\n[mpo_dmrg_io.copyMPS] ifQt=',ifQt,' nsite=',nsite
   if 'nsite' in fmps1: del fmps1['nsite']
   fmps1['nsite'] = nsite
   # Store qnums
   for isite in range(nsite+1):
      key = 'qnum'+str(isite)
      if key in fmps1: del fmps1[key]
      fmps1[key] = fmps0[key].value
   # Store sites   
   if not ifQt:
      for isite in range(nsite):
         key = 'site'+str(isite)
         if key in fmps1: del fmps1[key]
	 fmps1[key] = fmps0[key].value
   else:
      for isite in range(nsite):
	 key = 'site'+str(isite)
         if key in fmps1: del fmps1[key]
         site = qtensor.qtensor()
         site.load(fmps0,key)
         site.dump(fmps1,key)
   return 0

def dumpMPS(fmps,mpslst,icase=0):
   # No symmetry - mpslst as a list
   if icase == 0:
      nsite = len(mpslst)
      fmps['nsite'] = nsite
      for isite in range(nsite):
         fmps['site'+str(isite)] = mpslst[isite] 
   # With qnums and numpy.array
   elif icase == 1:
      nsite = len(mpslst[0])
      sites = mpslst[0]
      qnums = mpslst[1]
      fmps['nsite'] = nsite
      for isite in range(nsite):
         fmps['site'+str(isite)] = sites[isite]
      for isite in range(nsite+1):
	 fmps['qnum'+str(isite)] = qnums[isite]
   # With qnums and Qt	   
   elif icase == 2:
      nsite = len(mpslst[0])
      sites = mpslst[0]
      qnums = mpslst[1]
      fmps['nsite'] = nsite
      for isite in range(nsite):
	 key = 'site'+str(isite)
         sites[isite].dump(fmps,key)
      for isite in range(nsite+1):
	 fmps['qnum'+str(isite)] = qnums[isite]
   return 0

def loadMPS(fmps,icase=0):
   nsite = fmps['nsite'].value 
   # No symmetry - mpslst as a list
   if icase == 0:
      mpslst = [0]*nsite
      for isite in range(nsite):
         mpslst[isite] = fmps['site'+str(isite)].value
      result = [mpslst]
   # With qnums and numpy.array
   elif icase == 1:
      sites = [0]*nsite
      qnums = [0]*(nsite+1)
      for isite in range(nsite):
	 sites[isite] = loadSite(fmps,isite,False)
      for isite in range(nsite+1):
	 qnums[isite] = fmps['qnum'+str(isite)].value
      result = [sites,qnums]
   # With qnums and Qt	   
   elif icase == 2:
      sites = [0]*nsite
      qnums = [0]*(nsite+1)
      for isite in range(nsite):
	 sites[isite] = loadSite(fmps,isite,True)
      for isite in range(nsite+1):
	 qnums[isite] = fmps['qnum'+str(isite)].value
      result = [sites,qnums]
   return result

#
# Save and Load for Qnum / Site
#
def saveQnum(fmps,isite,qred):
   key = 'qnum'+str(isite)
   if key in fmps: del fmps[key]
   fmps[key] = qred
   return 0

def loadQnums(fmps):
   nsite = fmps['nsite'].value 
   qnums = [0]*(nsite+1)
   for isite in range(nsite+1):
      qnums[isite] = fmps['qnum'+str(isite)].value
   return qnums

def saveSite(fmps,isite,site):
   key = 'site'+str(isite)
   if key in fmps: del fmps[key]
   if isinstance(site,numpy.ndarray):
      fmps[key] = site
   else:
      site.dump(fmps,key)
   return 0

def loadSite(fmps,isite,ifQt):
   key = 'site'+str(isite)
   if ifQt:
      site = qtensor.qtensor()
      site.load(fmps,key)
   else:
      site = fmps[key].value
   return site

#
# dumpMPO
#

# DUMP Wx[isite] for H
def dumpMPO_H(dmrg,debug=False):
   rank = dmrg.comm.rank
   t0 = time.time()
   dmrg.fhop = h5py.File(dmrg.path+'/hop','w')
   dmrg.fhop['nsite'] = dmrg.nsite
   dmrg.fhop['nops'] = dmrg.nops
   dmrg.fhop['wts'] = dmrg.hpwts
   for isite in range(dmrg.nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = dmrg.fhop.create_group(gname)
      if not dmrg.ifQt:
	 # Loop over operators
         for iop in range(dmrg.nops):
            pindx = dmrg.opers[iop]
            cop = mpo_dmrg_opers.genHRfacSpatial(pindx,dmrg.sbas,isite,\
              		       		         dmrg.int1e,dmrg.int2e,\
              		       		         dmrg.qpts,dmrg.pdic)
            grp['op'+str(iop)] = cop
      else:
         # Significantly compressed by a factor of 1/10 !
         # HDF5 "dateApr_27_19_57_01_2016_rank0_pid38023_mpo_dmrg/hop" {
         #     GROUP "/" {
         #        GROUP "site0" {
         #           GROUP "op0" {
         #              DATASET "dims_allowed" {
	 # Loop over operators
         for iop in range(dmrg.nops):
            pindx = dmrg.opers[iop]
            cop = qtensor_opers.genHRfacSpatialQt(pindx,dmrg.sbas,isite,\
              		         	          dmrg.int1e,dmrg.int2e,\
              		         	          dmrg.qpts,dmrg.pdic,dmrg.maxslc)
            cop.dump(grp,'op'+str(iop))
      tf = time.time()
      if rank == 0: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   print ' path = ',dmrg.path+'/hop'
   print ' time for dumpMPO[hop] = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return 0

# R=exp(-i*phi*Sy) operators
def dumpMPO_R(dmrg,debug=False):
   rank = dmrg.comm.rank
   if rank == 0 and dmrg.ifs2proj:
      t0 = time.time()
      dmrg.fpop = h5py.File(dmrg.path+'/pop','w')
      dmrg.fpop['nsite'] = dmrg.nsite
      dmrg.fpop['nops'] = dmrg.npts
      dmrg.fpop['wts'] = dmrg.qwts
      # the R(theta) are the same for all sites!
      for iop in range(dmrg.npts):
         pts = dmrg.qpts[iop]
         if not dmrg.ifQt:
            cop = mpo_dmrg_opers.genExpISyPhi(pts)
            dmrg.fpop['op'+str(iop)] = cop
         else:
            cop = qtensor_opers.genExpISyPhiQt(pts)
            cop.dump(dmrg.fpop,'op'+str(iop))
      t1 = time.time()
      print ' path = ',dmrg.path+'/pop'
      print ' time for dumpMPO[pop] = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return 0

# MPO for H0 used in mps-based perturbation theory
# ifH0 = 1 - diagH without compression
#      = 2 - diagH (D=3K) [H_{EN}]
#      Spin-free Version:
#      = 3 - [ii|jj],[ij|ij],[ij|ji] (D=5K)
#      = 4 - spin-free (D=5K)
def dumpMPO_H0(dmrg,debug=False):
   if dmrg.ifs2proj and dmrg.ifH0 < 3:
      print 'error: PT for SP-MPS must be based on spin-free H0!'
      print '       In the current implementation ifH0 = 3 or 4.' 
      exit(1)
   rank = dmrg.comm.rank
   t0 = time.time()
   dmrg.fdop = h5py.File(dmrg.path+'/dop','w')
   dmrg.fdop['nsite'] = dmrg.nsite
   dmrg.fdop['nops'] = dmrg.nops
   dmrg.fdop['wts'] = dmrg.hpwts
   for isite in range(dmrg.nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = dmrg.fdop.create_group(gname)
      # Different cases for H0:
      if dmrg.ifH0 == 1:

         if not dmrg.ifQt:
            for iop in range(dmrg.nops):
               pindx = dmrg.opers[iop]
               cop = mpo_dmrg_opers.genHRfacSpatial(pindx,dmrg.sbas,isite,\
                 		       		    dmrg.int1e,dmrg.int2e,\
                 		       		    dmrg.qpts,dmrg.pdic)
	       dim = cop.shape[-1]
	       cop[...,~numpy.eye(dim,dtype=bool)] = 0.
               grp['op'+str(iop)] = cop
         else:
            for iop in range(dmrg.nops):
               pindx = dmrg.opers[iop]
               cop = qtensor_opers.genHRfacSpatialQt(pindx,dmrg.sbas,isite,\
                 		         	     dmrg.int1e,dmrg.int2e,\
                 		         	     dmrg.qpts,dmrg.pdic,dmrg.maxslc)
	       cop = cop.diagH()
               cop.dump(grp,'op'+str(iop))

      elif dmrg.ifH0 in [2,3,4]:          

	 icase = dmrg.ifH0-2
         if not dmrg.ifQt:
            for iop in range(dmrg.nops):
               pindx = dmrg.opers[iop]
               cop = mpo_dmrg_ptopers.genHenRfacSpatial(dmrg,pindx,isite,icase)
               grp['op'+str(iop)] = cop
         else:
            for iop in range(dmrg.nops):
               pindx = dmrg.opers[iop]
               cop = qtensor_opers.genHenRfacSpatialQt(dmrg,pindx,isite,icase)
               cop.dump(grp,'op'+str(iop))

      else:  
	 print '\nerror: No such option for dumpMPO_H0!'
	 exit(1)

      tf = time.time()
      if rank == 0: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   print ' time for dumpMPO[dop] = %.2f s'%(t1-t0),' rank =',rank
   return 0

def dumpMPO_Hubbard(dmrg,debug=False):
   rank = dmrg.comm.rank
   t0 = time.time()
   dmrg.fhop = h5py.File(dmrg.path+'/hop','w')
   dmrg.fhop['nsite'] = dmrg.nsite
   dmrg.fhop['nops'] = dmrg.nops
   dmrg.fhop['wts'] = dmrg.hpwts
   for isite in range(dmrg.nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = dmrg.fhop.create_group(gname)
      for iop in range(dmrg.nops):
	 cop = mpo_dmrg_model.genHubbardSpatial(dmrg,isite) 
         if not dmrg.ifQt:
            grp['op'+str(iop)] = cop
         else:
	    #cop.dump(grp,'op'+str(iop))
            print 'Not implemented yet!'
	    exit()
         if debug: print ' isite/iop =',isite,iop,' pindx =',pindx
      tf = time.time()
      if rank == 0: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   print ' time for dumpMPO[Hubbard] = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return 0

# DUMP Wx[isite] for H
def dumpMPO_H1e(dmrg,debug=False):
   rank = dmrg.comm.rank
   t0 = time.time()
   dmrg.fhop = h5py.File(dmrg.path+'/hop','w')
   dmrg.fhop['nsite'] = dmrg.nsite
   dmrg.fhop['nops'] = dmrg.nops
   dmrg.fhop['wts'] = dmrg.hpwts
   for isite in range(dmrg.nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = dmrg.fhop.create_group(gname)
      if not dmrg.ifQt:
	 # Loop over operators
         for iop in range(dmrg.nops):
            pindx = dmrg.opers[iop]
            cop = mpo_dmrg_opers1e.genHRfacSpatial(pindx,dmrg.sbas,isite,dmrg.h1e,\
              		       		           dmrg.qpts,dmrg.model_u)
            grp['op'+str(iop)] = cop
      else:
         for iop in range(dmrg.nops):
            pindx = dmrg.opers[iop]
            cop = qtensor_opers1e.genHRfacSpatialQt(pindx,dmrg.sbas,isite,dmrg.h1e,\
              		         	            dmrg.qpts,dmrg.maxslc,dmrg.model_u)
            cop.dump(grp,'op'+str(iop))
      tf = time.time()
      if rank == 0: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   print ' path = ',dmrg.path+'/hop'
   print ' time for dumpMPO[1e] = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return 0

#######
# Final
########

def finalMPS(dmrg):   
   print ' close flmps ...'
   try: 
      dmrg.flmps.close()
   except AttributeError or ValueError:
      pass
   print ' close frmps ...'
   try: 
      dmrg.frmps.close()
   except AttributeError or ValueError:
      pass
   return 0

def finalMPO(dmrg):
   rank = dmrg.comm.rank
   print ' close fhop ...'
   try:
      dmrg.fhop.close() 
   except AttributeError or ValueError:
      pass 
   if rank == 0 and dmrg.ifs2proj: 
      print ' close fpop ...'
      try:
         dmrg.fpop.close()
      except AttributeError or ValueError:
         pass 
   return 0

# Load H1e and ERIs
def loadInts(dmrg,mol):
   rank = dmrg.comm.rank
   # Load all integrals on each node
   if dmrg.ifAllInts:
      t0 = time.time()
      f = h5py.File(mol.fname, "r")
      dmrg.int1e = numpy.empty((dmrg.sbas,dmrg.sbas),dtype=dmrg_dtype)
      dmrg.int2e = numpy.empty((dmrg.sbas,dmrg.sbas,dmrg.sbas,dmrg.sbas),dtype=dmrg_dtype)
      if rank == 0:
         f = h5py.File(mol.fname, "r")
         dmrg.int1e= f['int1e'].value
         dmrg.int2e= f['int2e'].value
      dmrg.comm.Bcast([dmrg.int1e,dmrg_mtype])
      dmrg.comm.Bcast([dmrg.int2e,dmrg_mtype])
   # Distribute different integrals on different nodes
   else:
      # Rank-0 loads its all integrals first 
      if rank == 0:
         print '\n[mpo_dmrg_io.loadInts]'
         t0 = time.time()
         f = h5py.File(mol.fname, "r")
         dmrg.int1e= f['int1e'][dmrg.pidx]
         dmrg.int2e= f['int2e'][dmrg.pidx]
      # We do it sequentially for size>1
      for irank in range(1,dmrg.comm.size):
         # Send the pidx to rank-0
         if irank == rank:
            dmrg.comm.send(dmrg.pidx, dest=0, tag=0)
            dmrg.int1e = numpy.empty((dmrg.pdim,dmrg.sbas),dtype=dmrg_dtype)
            dmrg.int2e = numpy.empty((dmrg.pdim,dmrg.sbas,dmrg.sbas,dmrg.sbas),dtype=dmrg_dtype)
            dmrg.comm.Recv([dmrg.int1e,dmrg_mtype], source=0, tag=1)
            dmrg.comm.Recv([dmrg.int2e,dmrg_mtype], source=0, tag=2)
         # Receive the pidx from irank   
         if rank == 0:
            pidx = dmrg.comm.recv(None, source=irank, tag=0)
            int1e= f['int1e'][pidx]
            int2e= f['int2e'][pidx]
            dmrg.comm.Send([int1e,dmrg_mtype], dest=irank, tag=1)
            dmrg.comm.Send([int2e,dmrg_mtype], dest=irank, tag=2)
         dmrg.comm.Barrier()   
   # test
   debug = False
   if debug:
      print 'rank/pdim/int2e=',rank,dmrg.pdim,dmrg.int2e.shape
      tmp = h5py.File('tmp'+str(irank)+'.h5','w')
      tmp['int1e'] = dmrg.int1e
      tmp['int2e'] = dmrg.int2e
      tmp.close()
   if rank == 0: 
      f.close()	   
      t1 = time.time()
      print " loadInts sucessfully! time = %.2f s"%(t1-t0)
   return 0
