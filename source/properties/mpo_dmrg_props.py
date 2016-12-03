#!/usr/bin/env python
#
# Compute <bra|O|ket> given <bra|, |ket>, and O in MPS/MPO form.
# The code follows the codes for genHops in mpo_dmrg_init(Qt).
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
# 
# def evalProps(dmrg,fbmps,fkmps,fop,status='L',debug=False):
# def evalPropsNQt(dmrg,fbmps,fkmps,fop,status,debug=False):
# def evalPropsQt(dmrg,fbmps,fkmps,fop,status,debug=False):
#
import os
import time
import h5py
import numpy
from qcmpodmrg.source import sysutil_io
from qcmpodmrg.source import mpo_dmrg_io
from qcmpodmrg.source import mpo_dmrg_init
from qcmpodmrg.source import mpo_dmrg_initQt
from qcmpodmrg.source.qtensor import qtensor
from qcmpodmrg.source.sysutil_include import dmrg_dtype,dmrg_mtype

#
# General computation of <bra|O|ket> for NQt/Qt version
#
def evalProps(dmrg,fbmps,fkmps,fop,status='L',debug=False):
   if debug: print '\n[mpo_dmrg_props.evalProps] ifQt/ifs2proj=',(dmrg.ifQt,dmrg.ifs2proj)
   if not dmrg.ifQt:
      exphop = evalPropsNQt(dmrg,fbmps,fkmps,fop,status,debug)
   else:
      exphop = evalPropsQt(dmrg,fbmps,fkmps,fop,status,debug)
   return exphop

#
# NQt Version
#
def evalPropsNQt(dmrg,fbmps,fkmps,fop,status,debug=False):
   t0 = time.time()
   # sites
   bnsite = fbmps['nsite'].value
   knsite = fkmps['nsite'].value
   assert bnsite == knsite
   nsite = bnsite
   nop = fop['nop'].value
   if debug:
      print ' nop   = ',nop
      print ' nsite = ',nsite
   # Create directory
   path = './tmpdirProps'
   sysutil_io.createDIR(path,debug)
   fname = path+'/trop'
   prefix = fname+'_site_' 
   # L->R sweeps 
   if status == 'L':

      mpo_dmrg_init.genBopsNQt(fname,nop,-1)
      for isite in range(0,nsite):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite-1),"r")
         f1name = prefix+str(isite)
	 f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,False)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,False)
	 if isite == nsite-1: exphop = numpy.zeros(nop,dtype=dmrg_dtype)
         for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop
	    cop = fop['site'+str(isite)+'/op'+str(iop)].value
	    tmp = f0['opers'+str(iop)].value
	    #--- kernel ---
	    tmp = numpy.tensordot(bsite.conj(),tmp,axes=([0],[1]))
	    tmp = numpy.tensordot(cop,tmp,axes=([0,2],[2,0]))
	    tmp = numpy.tensordot(tmp,ksite,axes=([1,3],[1,0]))
	    #--- kernel ---
	    f1['opers'+str(iop)] = tmp
	    # Get the expectation value <psi|Hx|psi> at the boundary
	    if isite == nsite-1: exphop[iop] = tmp[0,0,0]
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
         if debug: print ' isite =',os.path.split(f1name)[-1],\
	       		 ' nop =',nop,' t = %.2f s'%(tf-ti)

   elif status == 'R':

      mpo_dmrg_init.genBopsNQt(fname,nop,nsite)
      for isite in range(nsite-1,-1,-1):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite+1),"r")
         f1name = prefix+str(isite)
         f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,False)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,False)
	 if isite == 0: exphop = numpy.zeros(nop,dtype=dmrg_dtype)
         for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop
	    cop = fop['site'+str(isite)+'/op'+str(iop)].value
	    tmp = f0['opers'+str(iop)].value
	    #--- kernel ---
	    tmp = numpy.tensordot(bsite.conj(),tmp,axes=([2],[1]))
	    tmp = numpy.tensordot(cop,tmp,axes=([1,2],[2,1]))
	    tmp = numpy.tensordot(tmp,ksite,axes=([1,3],[1,2]))
	    #--- kernel ---
	    f1['opers'+str(iop)] = tmp
	    # Get the expectation value <psi|Hx|psi> at the boundary
	    if isite == 0: exphop[iop] = tmp[0,0,0]
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if debug: print ' isite =',os.path.split(f1name)[-1],\
	       		 ' nop =',nop,' t = %.2f s'%(tf-ti)
   
   # final
   sysutil_io.deleteDIR(path,1,debug)
   t1=time.time()
   if debug: print ' time for evalProps = %.2f s'%(t1-t0)
   return exphop

#
# Qt Version: An important remark is that we donot use the 
#	      symmetry template for properties, as usually
# 	      the bond dimensions are very small. So for 
#	      simplicity, direct qtensor.tensordot is used!
#	      This also provides general treatments of <A|MPO|B>.
#
def evalPropsQt(dmrg,fbmps,fkmps,fop,status,debug=False):
   t0 = time.time()
   # sites
   bnsite = fbmps['nsite'].value
   knsite = fkmps['nsite'].value
   assert bnsite == knsite
   nsite = bnsite
   nop = fop['nop'].value
   if debug:
      print ' nop   = ',nop
      print ' nsite = ',nsite
   # Create directory
   path = './tmpdirProps'
   sysutil_io.createDIR(path,debug)
   fname = path+'/trop'
   prefix = fname+'_site_' 
   # L->R sweeps 
   if status == 'L':

      mpo_dmrg_initQt.genBopsQt(dmrg,fname,nop,-1,ifslc=False)
      lop = qtensor.qtensor()
      cop = qtensor.qtensor()
      for isite in range(0,nsite):
	 if debug: print ' isite=',isite,' of nsite=',nsite
	 ti = time.time()
	 f0 = h5py.File(prefix+str(isite-1),"r")
         f1name = prefix+str(isite)
	 f1 = h5py.File(f1name,"w")
	 bsite = mpo_dmrg_io.loadSite(fbmps,isite,True)
	 ksite = mpo_dmrg_io.loadSite(fkmps,isite,True)
	 # Lower the symmetry 
	 if dmrg.ifs2proj:
	    bsite = bsite.reduceQsymsToN()
	    ksite = ksite.reduceQsymsToN()
	 if isite == nsite-1: exphop = numpy.zeros(nop,dtype=dmrg_dtype)
	 for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop
	    # COP
	    cop.load(fop,'site'+str(isite)+'/op'+str(iop))
	    # LOP
	    lop.load(f0,'opers'+str(iop))
	    #--- kernel ---
	    tmp = qtensor.tensordot(bsite,lop,axes=([0],[1]),ifc1=True)
	    tmp = qtensor.tensordot(cop,tmp,axes=([0,2],[2,0]))
	    tmp = qtensor.tensordot(tmp,ksite,axes=([1,3],[1,0]))
	    #--- kernel ---
	    tmp.dump(f1,'opers'+str(iop))
	    # Get the expectation value <psi|Hx|psi> at the boundary
	    if isite == nsite-1: exphop[iop] = tmp.value
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if debug: print ' isite =',os.path.split(f1name)[-1],\
	       		 ' nop =',nop,' t = %.2f s'%(tf-ti)

   elif status == 'R':

      mpo_dmrg_initQt.genBopsQt(dmrg,fname,nop,nsite,ifslc=False)
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
	 # Lower the symmetry 
	 if dmrg.ifs2proj:
	    bsite = bsite.reduceQsymsToN()
	    ksite = ksite.reduceQsymsToN()
	 if isite == 0: exphop = numpy.zeros(nop,dtype=dmrg_dtype)
	 for iop in range(nop):
	    if debug: print '    iop=',iop,' of nop=',nop
	    # COP
	    cop.load(fop,'site'+str(isite)+'/op'+str(iop))
	    cop.qsyms[0]  = -cop.qsyms[0]
	    cop.qsyms[1]  = -cop.qsyms[1]
	    cop.status[0] = ~cop.status[0] 
	    cop.status[1] = ~cop.status[1] 
	    # ROP
	    rop.load(f0,'opers'+str(iop))
	    #--- kernel ---
	    tmp = qtensor.tensordot(bsite,rop,axes=([2],[1]),ifc1=True)
	    tmp = qtensor.tensordot(cop,tmp,axes=([1,2],[2,1]))
	    tmp = qtensor.tensordot(tmp,ksite,axes=([1,3],[1,2]))
	    #--- kernel ---
	    tmp.dump(f1,'opers'+str(iop))
	    # Get the expectation value <psi|Hx|psi> at the boundary
	    if isite == 0: exphop[iop] = tmp.value
	 f0.close()
	 f1.close()
         # final isite
	 tf = time.time()
	 if debug: print ' isite =',os.path.split(f1name)[-1],\
	       		 ' nop =',nop,' t = %.2f s'%(tf-ti)
 
   # final
   sysutil_io.deleteDIR(path,1,debug)
   t1=time.time()
   if debug: print ' time for evalProps = %.2f s'%(t1-t0)
   return exphop
