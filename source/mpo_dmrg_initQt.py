#!/usr/bin/env python
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def genBmatQt(dmrg,fname,isite,debug=False):
# def genBopsQt(dmrg,fname,nop,isite,ifslc=False,debug=False):
# def genHopsQt(dmrg,fbmps,fkmps,fname,status,debug=False):
# def genSopsQt(dmrg,fbmps,fkmps,fname,status,debug=False):
# def genPopsQt(dmrg,fbmps,fkmps,fname,status,debug=False):
# 
import os
import h5py
import time
import numpy
import mpo_dmrg_io
from qtensor import qtensor
from qtensor import qtensor_opers
from sysutil_include import dmrg_dtype,dmrg_mtype

# Boundary matrix (not sliced)
def genBmatQt(dmrg,fname,isite,debug=False):
   if debug: print '[mpo_dmrg_init.genBmatQt]'
   prefix = fname+'_site_'
   # left or right boundary  
   f1name = prefix+str(isite)
   if debug: print ' f1name=',f1name
   f1 = h5py.File(f1name,"w")
   # *** Construct Qt ***
   status = [False,True]
   if dmrg.isym==1: 
   # Even in the case of SP-MPS, for <A|B> we used (N,Sz).
   # or (dmrg.isym==2 and dmrg.ifs2proj): 
      qb = [[0.]]
      qk = [[0.]]
   else: 
      qb = [[0.,0.]]
      qk = [[0.,0.]]
   bdim = 1
   kdim = 1
   cop = numpy.ones(1,dtype=dmrg_dtype).reshape(bdim,kdim)
   tmps = qtensor.qtensor(status) 
   tmps.fromDenseTensor(cop,[qb,qk])
   tmps.dump(f1,'mat')
   f1.close()
   return 0

def genBopsQt(dmrg,fname,nop,isite,ifslc=False,debug=False):
   if debug: print '[mpo_dmrg_init.genBopsQt]'
   prefix = fname+'_site_'
   # left or right boundary  
   f1name = prefix+str(isite)
   if debug: print ' f1name=',f1name
   f1 = h5py.File(f1name,"w")
   # *** Construct Qt ***
   status = [True,False,True]
   if dmrg.isym==1 or (dmrg.isym==2 and dmrg.ifs2proj): 
      qo = [[0.]]
      qb = [[0.]]
      qk = [[0.]]
   else: 
      qo = [[0.,0.]]
      qb = [[0.,0.]]
      qk = [[0.,0.]]
   odim = 1
   bdim = 1
   kdim = 1
   cop = numpy.ones(1,dtype=dmrg_dtype).reshape(odim,bdim,kdim)
   tmps = qtensor.qtensor(status) 
   tmps.fromDenseTensor(cop,[qo,qb,qk])
   if not ifslc:
      for iop in range(nop):
         tmps.dump(f1,'opers'+str(iop))
   else:
      #DataGroups:
      # opers0: opers0_slc0,opers0_slc1,...
      # opers1: opers1_slc0,opers1_slc1,...
      tmps1 = qtensor.Qt(slcdim=1,maxslc=1)
      for iop in range(nop):
         tmps1.dic[0]  = tmps
	 tmps1.size[0] = tmps.size_allowed 
	 tmps1.dump(f1,'opers'+str(iop))
   f1.close()
   return 0

#@profile
def genHopsQt(dmrg,fbmps,fkmps,fname,status,debug=False):
   if dmrg.comm.rank == 0: 
      print '[mpo_dmrg_init.genHopsQt] status=',status
      print ' fname = ',fname
   t0 = time.time()
   nop = dmrg.nops
   # sites
   bnsite = fbmps['nsite'].value
   knsite = fkmps['nsite'].value
   assert bnsite == knsite
   nsite = bnsite
   prefix = fname+'_site_'
   if debug:
      print ' opers = ',dmrg.opers
      print ' fname  = ',fname
      print ' nop    = ',nop
      print ' nsite  = ',nsite
   dims = numpy.array([6]+[dmrg.maxslc]*2)
   ncase = numpy.prod(dims)
   # L->R sweeps 
   if status == 'L':

      genBopsQt(dmrg,fname,nop,-1,ifslc=True)
      lops = qtensor.Qt()
      cops = qtensor.Qt()
      for isite in range(0,nsite):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite-1),"r")
         f1name = prefix+str(isite)
	 f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 # Lower the symmetry 
	 if dmrg.ifs2proj:
	    bsite = bsite.reduceQsymsToN()
	    ksite = ksite.reduceQsymsToN()
	 if isite == nsite-1: exphop = numpy.zeros(nop,dtype=dmrg_dtype)
	 # cases for symmetry template depending on ap^+Vp
	 ista = dict([(i,-1) for i in range(ncase)])
	 top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop
	    # LOP
	    lops.loadInfo(f0,'opers'+str(iop))
	    # COP
	    pindx = dmrg.opers[iop]
	    cops.load(dmrg.fhop,'site'+str(isite)+'/op'+str(iop))
	    # Symmetry & Buffer
	    icase0 = qtensor_opers.case(pindx[0],isite)
	    # COMPUTE
	    qops = qtensor.Qt(1,cops.maxslc[1])
	    for jslc in range(cops.maxslc[1]):
	       for islc in range(cops.maxslc[0]):
		  ijdx = cops.ravel([islc,jslc])
		  if cops.size[ijdx] == 0 or lops.size[islc] == 0: continue
		  cop = cops.dic[ijdx]
		  lop = lops.loadSLC(f0,'opers'+str(iop),islc)
		  # Symmetry & Buffer
		  icase = numpy.ravel_multi_index([icase0,jslc,islc],dims)
		  if ista[icase] == -1:
	             ilen1 = top1[icase].tensordotSYM(bsite,lop,axes=([0],[1]))
	             ilen2 = top2[icase].tensordotSYM(cop,top1[icase],axes=([0,2],[2,0]))
		     ilen3 = top3[icase].tensordotSYM(top2[icase],ksite,axes=([1,3],[1,0]))
		     ista[icase] = ilen3
		  if ista[icase] <= 0: continue
	          top1[icase].tensordotCAL(bsite,lop,ifc1=True)
	          lop = None
		  top2[icase].tensordotCAL(cop,top1[icase])
	          top1[icase].value = None
	          top3[icase].tensordotCAL(top2[icase],ksite)
	          top2[icase].value = None
		  if qops.size[jslc] == 0:
	             qops.dic[jslc].copy(top3[icase])
	             qops.size[jslc] = qops.dic[jslc].size_allowed
		     # We dump the idlst information for operators & MPS
		     qops.dic[jslc].idlst = [cop.idlst[1],bsite.idlst[2],ksite.idlst[2]]
	          else:
	             qops.dic[jslc].value += top3[icase].value  
		  top3[icase].value = None
	       if qops.dic[jslc].size > 0: qops.dumpSLC(f1,'opers'+str(iop),jslc)
               # Get the expectation value <psi|Hx|psi> at the right boundary
	       # The nice thing is at the boundary, slicing = 1.
               if isite == nsite-1: exphop[iop] = qops.dic[jslc].value
	       qops.dic[jslc] = None
	    # DUMP information
	    qops.dumpInfo(f1,'opers'+str(iop))

	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	          ' nop =',nop,' t = %.2f s'%(tf-ti)

   elif status == 'R':

      genBopsQt(dmrg,fname,nop,nsite,ifslc=True)
      rops = qtensor.Qt()
      cops = qtensor.Qt()
      for isite in range(nsite-1,-1,-1):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite+1),"r")
         f1name = prefix+str(isite)
         f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 # Lower the symmetry 
	 if dmrg.ifs2proj:
	    bsite = bsite.reduceQsymsToN()
	    ksite = ksite.reduceQsymsToN()
	 if isite == 0: exphop = numpy.zeros(nop,dtype=dmrg_dtype)
	 # cases for symmetry template depending on ap^+Vp
	 ista = dict([(i,-1) for i in range(ncase)])
	 top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop
	    # ROP
	    rops.loadInfo(f0,'opers'+str(iop))
	    # COP
	    pindx = dmrg.opers[iop]
	    cops.load(dmrg.fhop,'site'+str(isite)+'/op'+str(iop))
	    # Symmetry & Buffer
	    icase0 = qtensor_opers.case(pindx[0],isite)
	    # COMPUTE
	    qops = qtensor.Qt(1,cops.maxslc[0])
	    for jslc in range(cops.maxslc[0]):
	       for islc in range(cops.maxslc[1]):
		  ijdx = cops.ravel([jslc,islc])
		  if cops.size[ijdx] == 0 or rops.size[islc] == 0: continue
		  cop = cops.dic[ijdx]
		  rop = rops.loadSLC(f0,'opers'+str(iop),islc)
		  # Symmetry & Buffer
		  icase = numpy.ravel_multi_index([icase0,jslc,islc],dims)
	          if ista[icase] == -1:
	  	     # Flip horizontal directions
	             cop.qsyms[0]  = -cop.qsyms[0]
	             cop.qsyms[1]  = -cop.qsyms[1]
	             cop.status[0] = ~cop.status[0] 
	             cop.status[1] = ~cop.status[1] 
	             ilen1 = top1[icase].tensordotSYM(bsite,rop,axes=([2],[1]))
	             ilen2 = top2[icase].tensordotSYM(cop,top1[icase],axes=([1,2],[2,1]))
	             ilen3 = top3[icase].tensordotSYM(top2[icase],ksite,axes=([1,3],[1,2]))
		     ista[icase] = ilen3
	          if ista[icase] <= 0: continue
		  top1[icase].tensordotCAL(bsite,rop,ifc1=True)
		  rop = None
	          top2[icase].tensordotCAL(cop,top1[icase])
	          top1[icase].value = None
	          top3[icase].tensordotCAL(top2[icase],ksite)
	          top2[icase].value = None
		  if qops.size[jslc] == 0:
	             qops.dic[jslc].copy(top3[icase])
	             qops.size[jslc] = qops.dic[jslc].size_allowed
		     # We dump the idlst information for operators & MPS
	             qops.dic[jslc].idlst = [cop.idlst[0],bsite.idlst[0],ksite.idlst[0]]
	          else:
	             qops.dic[jslc].value += top3[icase].value  
	          top3[icase].value = None
	       if qops.dic[jslc].size > 0: qops.dumpSLC(f1,'opers'+str(iop),jslc)
	       # Get the expectation value <psi|Hx|psi> at the boundary
	       if isite == 0: exphop[iop] = qops.dic[jslc].value
	       qops.dic[jslc] = None
	    # DUMP information
	    qops.dumpInfo(f1,'opers'+str(iop))

	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	          ' nop =',nop,' t = %.2f s'%(tf-ti)
  
   t1 = time.time()
   dmrg.comm.Barrier()
   print ' time for genHops = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return exphop

#
# <B|K> - We assume fbmps and fkmps are L-MPS. 
#
def genSopsQt(dmrg,fbmps,fkmps,fname,status,debug=False):
   if dmrg.comm.rank == 0: 
      print '[mpo_dmrg_init.genSopsQt] status=',status
      print ' fname = ',fname
   t0 = time.time()
   # sites
   bnsite = fbmps['nsite'].value
   knsite = fkmps['nsite'].value
   assert bnsite == knsite
   nsite = bnsite
   prefix = fname+'_site_'
   if debug:
      print ' fname  = ',fname
      print ' nsite  = ',nsite
   # L->R sweeps 
   if status == 'L':

      genBmatQt(dmrg,fname,-1)
      lop = qtensor.qtensor()
      for isite in range(0,nsite):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite-1),"r")
         f1name = prefix+str(isite)
	 f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 # LOP
	 lop.load(f0,'mat')
	 tmp = qtensor.tensordot(lop,bsite,axes=([0],[0]),ifc2=True)
	 tmp = qtensor.tensordot(tmp,ksite,axes=([0,1],[0,1]))
	 tmp.dump(f1,'mat') 
	 # Get the expectation value <psi|Hx|psi> at the boundary
	 if isite == nsite-1: expsop = tmp.value[0]
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	          ' t = %.2f s'%(tf-ti)

   elif status == 'R':

      genBmatQt(dmrg,fname,nsite)
      rop = qtensor.qtensor()
      for isite in range(nsite-1,-1,-1):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite+1),"r")
         f1name = prefix+str(isite)
         f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 # ROP
	 rop.load(f0,'mat')
	 tmp = qtensor.tensordot(bsite,rop,axes=([2],[0]),ifc1=True)
	 tmp = qtensor.tensordot(tmp,ksite,axes=([1,2],[1,2]))
	 tmp.dump(f1,'mat')
	 # Get the expectation value <psi|Hx|psi> at the boundary
	 if isite == 0: expsop = tmp.value[0]
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	          ' t = %.2f s'%(tf-ti)

   t1 = time.time()
   print ' time for genSops = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return expsop

# <R> where R operator is independent of site
def genPopsQt(dmrg,fbmps,fkmps,fname,status,debug=False):
   if dmrg.comm.rank == 0: 
      print '[mpo_dmrg_init.genPopsQt] status=',status
      print ' fname = ',fname
   t0 = time.time()
   nop = dmrg.npts
   odim = 1
   # sites
   bnsite = fbmps['nsite'].value
   knsite = fkmps['nsite'].value
   assert bnsite == knsite
   nsite = bnsite
   prefix = fname+'_site_'
   if debug:
      print ' fname  = ',fname
      print ' nop    = ',nop
      print ' nsite  = ',nsite
   # L->R sweeps 
   if status == 'L':

      genBopsQt(dmrg,fname,nop,-1,ifslc=False)
      lop = qtensor.qtensor()
      cop = qtensor.qtensor()
      for isite in range(0,nsite):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite-1),"r")
         f1name = prefix+str(isite)
	 f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 bsite = bsite.reduceQsymsToN() 
	 ksite = ksite.reduceQsymsToN()
	 if isite == nsite-1: exppop = numpy.zeros(nop,dtype=dmrg_dtype)
	 # cases for symmetry template depending on ap^+Vp
	 ncase = 1
	 ista = dict([(i,0) for i in range(ncase)])
	 top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 icase = 0
         for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop

	    # COP
	    cop.load(dmrg.fpop,'op'+str(iop))
	    # LOP
	    lop.load(f0,'opers'+str(iop))
	    if ista[icase] == 0:
	       top1[icase].tensordotSYM(bsite,lop,axes=([0],[1]))
	       top2[icase].tensordotSYM(cop,top1[icase],axes=([0,2],[2,0]))
	       top3[icase].tensordotSYM(top2[icase],ksite,axes=([1,3],[1,0]))
	       # We dump the idlst information for operators & MPS
	       top3[icase].idlst = [cop.idlst[1],bsite.idlst[2],ksite.idlst[2]]
	       ista[icase] = 1
	    top1[icase].tensordotCAL(bsite,lop,ifc1=True)
	    top2[icase].tensordotCAL(cop,top1[icase])
	    top1[icase].value = None
	    top3[icase].tensordotCAL(top2[icase],ksite)
	    top2[icase].value = None
	    top3[icase].dump(f1,'opers'+str(iop))
	    # Get the expectation value <psi|Hx|psi> at the boundary
	    if isite == nsite-1: exppop[iop] = top3[icase].value
	    top3[icase].value = None

	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	          ' nop =',nop,' t = %.2f s'%(tf-ti)

   elif status == 'R':

      genBopsQt(dmrg,fname,nop,nsite,ifslc=False)
      rop = qtensor.qtensor()
      cop = qtensor.qtensor()
      for isite in range(nsite-1,-1,-1):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite+1),"r")
         f1name = prefix+str(isite)
         f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 bsite = bsite.reduceQsymsToN() 
	 ksite = ksite.reduceQsymsToN()
	 if isite == 0: exppop = numpy.zeros(nop,dtype=dmrg_dtype)
	 # cases for symmetry template depending on ap^+Vp
	 ncase = 1
	 ista = dict([(i,0) for i in range(ncase)])
	 top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
	 icase = 0
         for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop
	    
	    # WOP
	    cop.load(dmrg.fpop,'op'+str(iop))
	    cop.qsyms[0]  = -cop.qsyms[0]
	    cop.qsyms[1]  = -cop.qsyms[1]
	    cop.status[0] = ~cop.status[0] 
	    cop.status[1] = ~cop.status[1] 
	    # ROP
	    rop.load(f0,'opers'+str(iop))
	    # Contract
	    if ista[icase] == 0:
	       top1[icase].tensordotSYM(bsite,rop,axes=([2],[1]))
	       top2[icase].tensordotSYM(cop,top1[icase],axes=([1,2],[2,1]))
	       top3[icase].tensordotSYM(top2[icase],ksite,axes=([1,3],[1,2]))
	       # We dump the idlst information for operators & MPS
	       top3[icase].idlst = [cop.idlst[0],bsite.idlst[0],ksite.idlst[0]]
	       ista[icase] = 1
	    top1[icase].tensordotCAL(bsite,rop,ifc1=True)
	    top2[icase].tensordotCAL(cop,top1[icase])
	    top1[icase].value = None
	    top3[icase].tensordotCAL(top2[icase],ksite)
	    top2[icase].value = None
	    top3[icase].dump(f1,'opers'+str(iop))
	    # Get the expectation value <psi|Hx|psi> at the boundary
	    if isite == 0: exphop[iop] = top3[icase].value
	    top3[icase].value = None

	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	          ' nop =',nop,' t = %.2f s'%(tf-ti)

   t1 = time.time()
   print ' time for genPops = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return exppop
