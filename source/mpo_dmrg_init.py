#!/usr/bin/env python
#
# Initialize renormalized operators for Boundary/H/S/Proj.
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
# 
# def genBmat(dmrg,fname,isite,debug=False):
# def genBmatNQt(dmrg,fname,isite,debug=False):
# def genBops(dmrg,fname,nop,isite,ifslc=False,debug=False):
# def genBopsNQt(fname,nop,isite,debug=False):
# def genHops(dmrg,fbmps,fkmps,fname,status,debug=False):
# def genHopsNQt(dmrg,fbmps,fkmps,fname,status,debug=False):
# def genSops(dmrg,fbmps,fkmps,fname,status,debug=False):
# def genSopsNQt(dmrg,fbmps,fkmps,fname,status,debug=False):
# def genPops(dmrg,fbmps,fkmps,fname,status,debug=False):
# def genPopsNQt(dmrg,fbmps,fkmps,fname,status,debug=False):
# 
import os
import time
import h5py
import numpy
import mpo_dmrg_io
import mpo_dmrg_initQt
from sysutil_include import dmrg_dtype,dmrg_mtype

# Dump <x1|O|x2> x=l,r into file    
def genBmat(dmrg,fname,isite,debug=False):
   if debug: print '[mpo_dmrg_init.genBmat] ifQt=',dmrg.ifQt
   if dmrg.ifQt:
      mpo_dmrg_initQt.genBmatQt(dmrg,fname,isite,debug)
   else:
      genBmatNQt(fname,isite,debug)
   return 0

def genBmatNQt(fname,isite,debug=False):
   if debug: print '[mpo_dmrg_init.genBmatNQt]'
   prefix = fname+'_site_'
   # left or right boundary  
   f1name = prefix+str(isite)
   if debug: print ' f1name=',f1name
   f1 = h5py.File(f1name,"w")
   bdim = 1
   kdim = 1
   cop = numpy.ones(1,dtype=dmrg_dtype).reshape(bdim,kdim)
   f1['mat'] = cop
   f1.close()
   return 0

# Dump <x1|O|x2> x=l,r into file    
def genBops(dmrg,fname,nop,isite,ifslc=False,debug=False):
   if debug: print '[mpo_dmrg_init.genBops] ifQt=',dmrg.ifQt,' ifslc=',ifslc
   if dmrg.ifQt:
      mpo_dmrg_initQt.genBopsQt(dmrg,fname,nop,isite,ifslc,debug)
   else:
      genBopsNQt(fname,nop,isite,debug)
   return 0

def genBopsNQt(fname,nop,isite,debug=False):
   if debug: print '[mpo_dmrg_init.genBopsNQt]'
   # Add two fake operators, just for consistenty with 
   # the latter determination of dimension of the CI space.
   # > nop for the number of operators Hx
   # > isite should be -1 for L and nsite for R.
   prefix = fname+'_site_'
   # left or right boundary  
   f1name = prefix+str(isite)
   if debug: print ' f1name=',f1name
   f1 = h5py.File(f1name,"w")
   odim = 1
   bdim = 1
   kdim = 1
   cop = numpy.ones(1,dtype=dmrg_dtype).reshape(odim,bdim,kdim)
   for iop in range(nop):
      f1['opers'+str(iop)] = cop
   f1.close()
   return 0

# Dump <x1|O|x2> x=l,r into file    
def genHops(dmrg,fbmps,fkmps,fname,status,debug=False):
   if dmrg.comm.rank == 0: print '\n[mpo_dmrg_init.genHops] ifQt=',dmrg.ifQt
   if dmrg.ifQt:
      exphop = mpo_dmrg_initQt.genHopsQt(dmrg,fbmps,fkmps,fname,status,debug)
   else:
      exphop = genHopsNQt(dmrg,fbmps,fkmps,fname,status,debug)
   return exphop

# Dump <x1|O|x2> x=l,r into file    
def genHopsNQt(dmrg,fbmps,fkmps,fname,status,debug=False):
   if dmrg.comm.rank == 0: 
      print '[mpo_dmrg_init.genHopsNQt] status=',status
      print ' fname = ',fname
   t0 = time.time()
   opIndices = dmrg.opers
   nop = dmrg.fhop['nops'].value
   # sites
   bnsite = fbmps['nsite'].value
   knsite = fkmps['nsite'].value
   assert bnsite == knsite
   nsite = bnsite
   prefix = fname+'_site_'
   if debug:
      print ' opIndices = ',opIndices
      print ' fname  = ',fname
      print ' nop    = ',nop
      print ' nsite  = ',nsite
   # L->R sweeps 
   if status == 'L':

      genBopsNQt(fname,nop,-1)
      for isite in range(0,nsite):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite-1),"r")
         f1name = prefix+str(isite)
	 f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 if isite == nsite-1: exphop = numpy.zeros(nop,dtype=dmrg_dtype)
         for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop
	    cop = dmrg.fhop['site'+str(isite)+'/op'+str(iop)].value
	    tmp = f0['opers'+str(iop)].value
	    #--- kernel ---
	    tmp = numpy.tensordot(bsite.conj(),tmp,axes=([0],[1])) # imj,pia->mjpa
	    tmp = numpy.tensordot(cop,tmp,axes=([0,2],[2,0]))      # pqmn,mjpa->qnja
	    tmp = numpy.tensordot(tmp,ksite,axes=([1,3],[1,0]))    # qnja,anb->qjb
	    #--- kernel ---
	    f1['opers'+str(iop)] = tmp
	    # Get the expectation value <psi|Hx|psi> at the boundary
	    if isite == nsite-1: exphop[iop] = tmp[0,0,0]
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	       	  ' nop =',nop,' t = %.2f s'%(tf-ti)

   elif status == 'R':

      genBopsNQt(fname,nop,nsite)
      for isite in range(nsite-1,-1,-1):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite+1),"r")
         f1name = prefix+str(isite)
         f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 if isite == 0: exphop = numpy.zeros(nop,dtype=dmrg_dtype)
         for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop
	    cop = dmrg.fhop['site'+str(isite)+'/op'+str(iop)].value
	    tmp = f0['opers'+str(iop)].value
	    #--- kernel ---
	    tmp = numpy.tensordot(bsite.conj(),tmp,axes=([2],[1])) # imj,qjb->imqb
	    tmp = numpy.tensordot(cop,tmp,axes=([1,2],[2,1]))      # pqmn,imqb->pnib
	    tmp = numpy.tensordot(tmp,ksite,axes=([1,3],[1,2]))    # pnib,anb->pia
	    #--- kernel ---
	    f1['opers'+str(iop)] = tmp
	    # Get the expectation value <psi|Hx|psi> at the boundary
	    if isite == 0: exphop[iop] = tmp[0,0,0]
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	          ' nop =',nop,' t = %.2f s'%(tf-ti)
   
   t1=time.time()
   dmrg.comm.Barrier()
   print ' time for genHops = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return exphop

# Dump <x1|x2> x=l,r into file    
def genSops(dmrg,fbmps,fkmps,fname,status,debug=False):
   if dmrg.comm.rank == 0: print '\n[mpo_dmrg_init.genSops] ifQt=',dmrg.ifQt
   if dmrg.ifQt:
      expsop = mpo_dmrg_initQt.genSopsQt(dmrg,fbmps,fkmps,fname,status,debug)
   else:
      expsop = genSopsNQt(dmrg,fbmps,fkmps,fname,status,debug)
   return expsop

def genSopsNQt(dmrg,fbmps,fkmps,fname,status,debug=False):
   if dmrg.comm.rank == 0: 
      print '[mpo_dmrg_init.genSopsNQt] status=',status
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

      genBmatNQt(fname,-1)
      for isite in range(0,nsite):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite-1),"r")
         f1name = prefix+str(isite)
	 f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 #
	 #   i-------j
	 #  /	 |
	 # *	 |m
	 #  \	 |
	 #   a-------b
	 #
	 tmp = f0['mat'].value
	 #--- kernel ---
	 tmp = numpy.tensordot(tmp,bsite.conj(),axes=([0],[0])) # ia,imj->amj
	 tmp = numpy.tensordot(tmp,ksite,axes=([0,1],[0,1]))    # amj,amb->jb
	 #--- kernel ---
	 f1['mat'] = tmp
	 # Get the expectation value <psi|Hx|psi> at the boundary,
	 # Here, unlike the exphop, expsop is just a scalar <l|r>.
	 if isite == nsite-1: expsop = tmp[0,0]
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	          ' t = %.2f s'%(tf-ti)

   elif status == 'R':

      genBmatNQt(fname,nsite)
      for isite in range(nsite-1,-1,-1):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite+1),"r")
         f1name = prefix+str(isite)
         f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 #
	 #   i-------j
	 #  	 |    \
	 #      m|     *
	 #  	 |    /
	 #   a-------b
	 #
	 tmp = f0['mat'].value
	 #--- kenerl ---
	 tmp = numpy.tensordot(bsite.conj(),tmp,axes=([2],[0])) # imj,jb->imb
	 tmp = numpy.tensordot(tmp,ksite,axes=([1,2],[1,2]))    # imb,amb->ia
	 #--- kenerl ---
	 f1['mat'] = tmp
	 # Get the expectation value <psi|Hx|psi> at the boundary
	 if isite == 0: expsop = tmp[0,0]
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	          ' t = %.2f s'%(tf-ti)

   t1=time.time()
   print ' time for genSops = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return expsop

# Dump <x1|P|x2> x=l,r into file   
# We could just treat P as a sum of MPO with nop operators and odim=1.
# Note that we do not need to put the weights into the first site in such rep.
def genPops(dmrg,fbmps,fkmps,fname,status,debug=False):
   if dmrg.comm.rank == 0: print '\n[mpo_dmrg_init.genPops] ifQt=',dmrg.ifQt
   if dmrg.ifQt:
      exppop = mpo_dmrg_initQt.genPopsQt(dmrg,fbmps,fkmps,fname,status,debug)
   else:
      exppop = genPopsNQt(dmrg,fbmps,fkmps,fname,status,debug)
   return exppop

def genPopsNQt(dmrg,fbmps,fkmps,fname,status,debug=False):
   if dmrg.comm.rank == 0: 
      print '[mpo_dmrg_init.genPopsNQt] status=',status
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

      genBopsNQt(fname,nop,-1)
      for isite in range(0,nsite):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite-1),"r")
         f1name = prefix+str(isite)
	 f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 if isite == nsite-1: exppop = numpy.zeros(nop,dtype=dmrg_dtype)
         for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop
	    cop = dmrg.fpop['op'+str(iop)].value
	    tmp = f0['opers'+str(iop)].value
	    #--- kernel ---
	    tmp = numpy.tensordot(bsite.conj(),tmp,axes=([0],[1])) # imj,pia->mjpa
	    tmp = numpy.tensordot(cop,tmp,axes=([0,2],[2,0]))      # pqmn,mjpa->qnja
	    tmp = numpy.tensordot(tmp,ksite,axes=([1,3],[1,0]))    # qnja,anb->qjb
	    #--- kernel ---
	    f1['opers'+str(iop)] = tmp
	    # Get the expectation value <psi|Hx|psi> at the boundary
	    if isite == nsite-1: exppop[iop] = tmp[0,0,0]
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	          ' nop =',nop,' t = %.2f s'%(tf-ti)

   elif status == 'R':

      genBopsNQt(fname,nop,nsite)
      for isite in range(nsite-1,-1,-1):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite+1),"r")
         f1name = prefix+str(isite)
         f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,dmrg.ifQt)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,dmrg.ifQt)
	 if isite == 0: exppop = numpy.zeros(nop,dtype=dmrg_dtype)
         for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop
	    cop = dmrg.fpop['op'+str(iop)].value
	    tmp = f0['opers'+str(iop)].value
	    #--- kernel --- 
	    tmp = numpy.tensordot(bsite.conj(),tmp,axes=([2],[1])) # imj,qjb->imqb
	    tmp = numpy.tensordot(cop,tmp,axes=([1,2],[2,1]))      # pqmn,imqb->pnib
	    tmp = numpy.tensordot(tmp,ksite,axes=([1,3],[1,2]))    # pnib,anb->pia
	    #--- kernel --- 
	    f1['opers'+str(iop)] = tmp
	    # Get the expectation value <psi|Hx|psi> at the boundary
	    if isite == 0: exppop[iop] = tmp[0,0,0]
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if dmrg.comm.rank == 0:
	    print ' isite =',os.path.split(f1name)[-1],\
	          ' nop =',nop,' t = %.2f s'%(tf-ti)

   t1=time.time()
   print ' time for genPops = %.2f s'%(t1-t0),' rank =',dmrg.comm.rank
   return exppop
