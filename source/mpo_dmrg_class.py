#!/usr/bin/env python
#
# Main class for performing various DMRG tasks
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
# 
# def __init__(self):
# def fdopXfhop(self):
# def s2quad(self,sval=0,sz=0):
# def partition(self):
# def loadInts(self,mol):
# def dumpMPO(self,debug=False):
# def dumpMPO_Model(self,name,debug=False):
#
# def build(self):
# def initStates(self):
#
# def default(self,sc,fmps=None):
#
# def ci(self,ifpt=False):
# def vpt(self):
# def detci(self):
#
# def final(self): 
#
# def checkMPS(self,fbra=None,fket=None,status='L',fname0='top',ifprt=True):
#
import os
import h5py
import time
import numpy
import mpo_dmrg_io
import mpo_dmrg_opt
import mpo_dmrg_prt
import mpo_dmrg_util
import mpo_dmrg_qphys
import mpo_dmrg_init
from mpi4py import MPI
from tools import parallel_util 

#
# Class for storage of basic information
#
class mpo_dmrg:
   def __init__(self):
      self.iprt  = 0
      self.comm  = None
      self.ifplot= False #True
      self.occun = None
      self.ecomp = None
      self.nsite = None       # number of sites
      self.sbas  = None       # number of spin-orbitals or operators (model H case)
      # This control the final energy convergence
      self.crit_tol = None    # crit_e for energy
      self.crit_vec = 1.e-8   # crit_vec in Davidson
      self.thresh   = -1.e-14 # -1.e-14
      self.ifdvdnz  = True    # False
      # Precise control of bond dimensions
      self.Dcut = None
      # TMP settings determined by the schedule
      self.Dmax = None     # Current maximal bond dimension 
      self.crit_e = 1.e-10 # Current energy convergence in Davidson 
      self.noise  = 1.e-4  # None
      self.inoise = 0      # default =0,pRDM;=1,random
      self.ifprecond = True
      self.schedule = None
      # Info
      self.isweep  = None
      self.ifconv  = None
      self.nsweep  = None
      self.deltaE  = None
      self.Energy  = 0.0  # Record energy at each sweep
      self.eav 	   = None # List
      self.dwt     = None # List
      self.esweeps = None # List
      # Files
      self.path   = None
      self.fname0 = None #"bfmpo.h5" #"cdmpo.h5" # File for integrals (CDmpo)
      self.fname1 = "opmpo.h5" # File for Hmpo used in dmrg
      # MPS - these are h5py file objects
      self.flmps = None 
      self.frmps = None
      # Guess
      self.ifguess= True #False
      self.psi0   = None
      self.ncsite = None
      self.status = None #twoSite or oneSite
      self.nmvp = []
      # solver
      self.solver = 0
      #------------------------------------------------------
      # Quantum numbers
      self.isym  = 2    # =0, no symmetry is used.
      			# =1, particle number - N.
			# =2, spin projection - Sz.
      self.qphys = None # quantum number of physical indices
      self.dphys = None
      self.qnuml = None # quantum number of virtual indices L
      self.qnumr = None # quantum number of virtual indices R
			# site :    0   1   2  
			# qnum : -0-*-1-*-2-*-3- (indices)
      self.sigsl = None
      self.sigsr = None
      self.nelec = 0.0 
      self.sz = 0.0
      # Multi-state support
      self.neig     = None
      self.wts      = None
      self.qsectors = None
      #------------------------------------------------------
      # Better initialization by increasing non-sym sweep
      self.initNsweep = 0
      self.initNcsite = 1
      #------------------------------------------------------
      # Just for printing everything looks good.
      self.const = 0.0
      #------------------------------------------------------
      # Spin projection
      self.quad = 'GaussLegendre' # 'Simpson'
      self.ifs2proj = False
      self.npts = None
      self.qwts = None
      self.qpts = None
      #------------------------------------------------------
      # Distribution of operators
      self.opers = []
      self.nops  = None
      #------------------------------------------------------
      # Direct Product Space for local optimizations
      self.dims = None # [ldim,cdim,rdim,ndim]
      #------------------------------------------------------
      # Qtensor
      self.ifQt = False
      self.qkey = None # Record symmetry of current targeted states!
      		       # This is used to change the quantum numbers.
      self.qlst = None # [ql,qc,qr] lists 
      self.qtmp = None # tmplate for CI vectors
      self.idlstCI   = None # temporay idlst for CI
      self.idlstPRDM = None # temporay idlst for PRDM
      #------------------------------------------------------
      # IO based implementation [standard]
      self.path = None
      self.fhop = None
      self.fpop = None
      self.maxslc = 1
      #------------------------------------------------------
      self.ifAllInts = False #True
      self.pdim  = None # no. of operators H[x]
      self.pidx  = None # indices of x
      self.pdic  = None # map x to idx for ints
      self.int1e = None
      self.int2e = None
      self.sint1e = None
      self.sint2e = None
      #------------------------------------------------------
      # state specific for excited states of same symmetry 
      self.ifex  = False
      self.nref  = 0
      self.wfex  = []
      self.wfex_canon = [] # Canonial forms: T/F => L/R
      #------------------------------------------------------
      self.ifpt  = False
      self.ifH0  = 4     # Use EN, otherwise full H.
      self.emix  = 0.0   # e0+e1-lambda*e1 
      self.e0    = 0.0   # E0 = <Psi0|H0|Psi0>
      self.e1    = 0.0   # E1 = <Psi0|H1|Psi0>
      self.et    = 0.0   # Et = <Psi0|Ht|Psi0>
      self.n0    = 1.0   # Normalization of P|Phi0> for SP-MPS
      self.coef  = None  # Refernce coefficients  
      self.fdop  = None  # File for diagonal operators
      self.ifE0  = False # =F: Use Etot as E0
      #------------------------------------------------------
      self.ifH0ortho = False
      #------------------------------------------------------
      self.ifptn = False # n-th order pt
      self.enlst = None
      #------------------------------------------------------
      self.ifcompression = False # Compressiong for H|psi>
      #------------------------------------------------------
      self.ifsym = True  # This allows nosym DMRG 
      #------------------------------------------------------
      self.ifsci = False # Selected CI like scheme
      self.trsci = 1.e-5 # Threshold for selection 
      #------------------------------------------------------
      # Hubbard model
      #------------------------------------------------------
      self.model_t = 1 
      self.model_u = 0
      self.h1e     = None # general hopping matrix
      #------------------------------------------------------

   # A temporary solution: 
   # this is because my genHops does not support input fhop yet.
   def fdopXfhop(self):
      self.fdop,self.fhop = self.fhop,self.fdop
      return 0

   # Generate quadrature for SP-MPS
   def s2quad(self,sval=0,sz=0):
      mpo_dmrg_util.s2quad(self,sval,sz)
      return 0

   # Partition operators
   def partition(self):
      if self.comm.rank == 0: print '\n[mpo_dmrg_class.partition]'
      if self.sbas is None:
	 print 'error: sbas is not defined!'
	 exit(1)
      rank = self.comm.rank
      size = self.comm.size
      # Normal MPS
      if self.npts is None: 
         assert self.sbas >= size
	 indices = parallel_util.partitionSites(self.sbas,size,rank)
	 self.opers = map(lambda x:[x,None],indices)
      # SP-MPS
      else:
	 assert self.npts*self.sbas >= self.comm.size
	 indices = parallel_util.partitionSites(self.npts*self.sbas,size,rank)
	 self.opers = map(lambda x:parallel_util.unrank(x,self.npts),indices)
      self.nops  = len(self.opers)
      # Set up mapping dictionary for integrals
      if self.ifAllInts:
	 self.pidx = range(self.sbas)
      else:
	 # One must be caution that both set and dic.keys()
	 # do not preserve the ordering when converted to list!
	 # {8: 2, 6: 0, 7: 1}
	 # set([6,7,8]) = [8,6,7]
	 self.pidx = sorted(list(set(map(lambda x:x[0],self.opers))))
      self.pdim = len(self.pidx) 
      self.pdic = dict([(self.pidx[i],i) for i in range(self.pdim)])
      assert self.nops>0
      # Weights for each bare MPO operators
      if self.npts is None:
         self.hpwts = numpy.ones(self.nops)
      else: 
	 self.hpwts = numpy.array(map(lambda x:self.qwts[x[1]],self.opers))
      # debug:
      for i in range(self.comm.size):
	 if i == self.comm.rank: 
	    print ' rank=',self.comm.rank,' nops=',self.nops,' opers=',self.opers
	 self.comm.Barrier()
      return 0

   # Load integrals
   def loadInts(self,mol):
      mpo_dmrg_io.loadInts(self,mol)
      return 0

   # Disk based  
   def dumpMPO(self,debug=False):
      rank = self.comm.rank
      if rank == 0: print '\n[mpo_dmrg_class.dumpMPO]'
      mpo_dmrg_io.dumpMPO_H(self,debug)
      mpo_dmrg_io.dumpMPO_R(self,debug)
      if self.ifpt and self.ifH0:
	 mpo_dmrg_io.dumpMPO_H0(self,debug)
      return 0

   def dumpMPO_Model(self,name,debug=False):
      rank = self.comm.rank
      if rank == 0: print '\n[mpo_dmrg_class.dumpMPO_Model] name =',name,' path =',self.path
      if name == 'Hubbard':
	 mpo_dmrg_io.dumpMPO_Hubbard(self)
      elif name == 'HubbardGeneral':
         mpo_dmrg_io.dumpMPO_H1e(self,debug)
         mpo_dmrg_io.dumpMPO_R(self,debug)
      else:
	 print 'error: Not implemented yet!'
         exit(1)
      return 0

   # Build some basic informations:
   # To make it consistent with the one site algorithm, 
   # we use nsite+1 places to keep all the symmety indices for qnuml/r.
   def build(self):
      if self.nsite < 2:
	 print 'error: mps is not supported for nsite < 2'
	 exit(1)
      assert self.isym in [1,2]
      # Initialize quantum numbers
      self.qphys = mpo_dmrg_qphys.initSpatialOrb(self.nsite,self.isym)
      self.dphys = map(lambda x:len(x),self.qphys)
      # Left
      self.qnuml = [None]*(self.nsite+1)
      self.qnuml[0] = numpy.array(mpo_dmrg_qphys.vacuum(self.isym))
      self.sigsl = [1.0]*(self.nsite+1)
      self.sigsl[0] = numpy.ones(1)
      # Right
      self.qnumr = [None]*(self.nsite+1)
      self.qnumr[-1] = numpy.array(mpo_dmrg_qphys.vacuum(self.isym))
      self.sigsr = [1.0]*(self.nsite+1)
      self.sigsr[-1] = numpy.ones(1)
      return 0

   def initStates(self):
      if self.qsectors is None and self.neig is None:
	 print 'error: no state is specified!'
         exit(1)
      if self.qsectors is not None:
         if self.isym == 0:
            print 'error: isym must be set in accordance with qsectors =',self.qsectors
            exit(1)
         neig = 0
         for key in self.qsectors:
            neig += self.qsectors[key]
         self.neig = neig
      else:
         # No symmetry case	 
         if self.isym == 0:
            key = str([])
         elif self.isym == 1:
            key = str([self.nelec])
         elif self.isym == 2:
            key = str([self.nelec,self.sz])
         else:
            print 'error: no such option for isym = ',self.isym
            exit(1)
         self.qsectors = {key:self.neig}
      # Generate weights, which can also be manually set.
      if self.wts is None:
         self.wts = numpy.ones(self.neig)/float(self.neig)
      return 0

   # Standard DMRG algorithm
   def default(self,sc,fmps=None):
      rank = self.comm.rank
      self.schedule = sc
      self.crit_tol = sc.tol
      if rank == 0: 
	 print '\n[mpo_dmrg_class.default]'
	 mpo_dmrg_prt.title(self)
      ti = time.time()
      # Count states
      self.initStates()
      if rank == 0: 
	 # Only set Dcut for rank-0 is sufficient, 
	 # as RDM truncation is done here.
	 if self.Dcut is not None:
	    print ' Dcut = ',self.Dcut
	    # Similar to qnuml and qnumr
	    assert len(self.Dcut) == self.nsite-1
	    self.Dcut = [1]+self.Dcut+[1]
         # LogFile in current directory
	 flog_name = 'log_'+os.path.split(self.path)[-1]
	 print ' flog_name =',flog_name
	 flog = open(flog_name,'w')
         flog.write('path:'+self.path+'\n')
      # MPSfile
      self.flmps = h5py.File(self.path+'/lmps','w')
      self.frmps = h5py.File(self.path+'/rmps','w')
      self.flmps['nsite'] = self.nsite
      self.frmps['nsite'] = self.nsite
      mpo_dmrg_io.saveQnum(self.flmps,0,self.qnuml[0])
      mpo_dmrg_io.saveQnum(self.frmps,self.nsite,self.qnumr[-1])
      # Initialize L-MPS by a single left sweep
      if fmps is None:
	 # This is not allowed for using Qt.
	 assert self.ifQt == False 
	 self.status = 'init'
         self.Dmax   = self.schedule.MaxMs[0]  
         self.crit_e = self.schedule.Tols[0]   
         self.noise  = self.schedule.Noises[0]
         mpo_dmrg_util.initMPS(self)
      # Use L-MPS from input
      else:
         self.comm.Barrier()
	 # Not only requires qnuml for setting up mpo_dmrg_util
	 self.qnuml = mpo_dmrg_io.loadQnums(fmps)
	 # but also the sites are needed to be copied to flmps!
	 mpo_dmrg_io.copyMPS(self.flmps,fmps,self.ifQt)
         self.comm.Barrier()
	 # Initialize operators
	 mpo_dmrg_util.initOps(self,fmps)	 
      # Optimize
      self.comm.Barrier()
      self.esweeps = []	
      self.psi0 = None
      self.eav = []
      self.dwt = []
      sitelst  = range(self.nsite)
      ifconv = False
      deltaE = 1.e3
      eold   = 1.e3
      isweep = -1
      # Optimize following the schedule
      for isweep in range(self.schedule.maxiter):
	 if rank == 0: print '\n'+'#'*30+' isweep = ',isweep,'#'*30
	 t0 = time.time()
	 # Record status
	 self.isweep = isweep
	 # Setup parameters
	 self.schedule.getParameters(self,isweep)
	 # Optimization
	 result1 = mpo_dmrg_opt.sweep(self,sitelst,self.ncsite,'R',ifsym=self.ifsym)
	 result2 = mpo_dmrg_opt.sweep(self,sitelst,self.ncsite,'L',ifsym=self.ifsym)
	 nmvp1,indx1,eav1,dwt1,elst1,dlst1 = result1
	 nmvp2,indx2,eav2,dwt2,elst2,dlst2 = result2
	 if rank == 0: mpo_dmrg_prt.flogWrite(self,flog,isweep,result1,result2) 	 
	 self.nmvp.append(nmvp1+nmvp2)
	 self.esweeps += elst1+elst2
	 if eav1 < eav2:
    	    self.eav.append(eav1)
    	    self.dwt.append(dwt1)
	    eav = eav1
	 else:
    	    self.eav.append(eav2)
    	    self.dwt.append(dwt2)
	    eav = eav2
         # Check convergence
	 t1 = time.time()
	 self.Energy = eav
    	 deltaE = eav-eold
	 if rank == 0: print 'Summary: isweep=',isweep,' Emin=',eav,\
			     ' dE=%9.2e  tol=%8.1e  t=%8.1e s'%(deltaE,self.crit_tol,t1-t0)
	 # Check convergence
	 ifconv = self.schedule.checkConv(self,isweep,deltaE)
         eold = eav
	 if ifconv: break
      # Save	 
      self.ifconv = ifconv
      self.nsweep = isweep + 1
      self.deltaE = deltaE
      tf = time.time()
      self.comm.Barrier()
      # Adjust a little bit to allow isweep = 0.
      # The only problem is the R-MPS is not initialized.
      if rank == 0 and isweep != -1: mpo_dmrg_prt.finalSweep(self,tf-ti)
      if rank == 0: flog.close()
      self.comm.Barrier()
      return 0
   
   # MPS-based CI
   def ci(self,ifpt=False):
      rank = self.comm.rank
      if rank == 0: print '\n[mpo_dmrg_class.ci] ifpt =',ifpt
      result = mpo_dmrg_util.ci(self)
      return result

   # MPS-based VPT
   def vpt(self):
      rank = self.comm.rank
      if rank == 0: print '\n[mpo_dmrg_class.vpt] solve Hylleraas functional in a given linear space'
      e2,c1 = mpo_dmrg_util.vpt(self)
      return e2,c1

   # MPS-based CI
   def detci(self):
      rank = self.comm.rank
      if rank == 0: print '\n[mpo_dmrg_class.detci]'
      self.dumpMPO()
      from mpsci import mixedci
      mixedci.main(self)
      return 0

   # The finalization is postponed, since checkMPS or the latter
   # initializations may use fhop/fpop and flmps/frmps.
   def final(self): 
      rank = self.comm.rank
      if rank == 0: print '\n[mpo_dmrg_class.final]'
      print ' path = ',self.path
      mpo_dmrg_io.finalMPS(self)
      mpo_dmrg_io.finalMPO(self)
      return 0

   # Generate {<A|H|B>,<A|B>,<A|P|B>}
   def checkMPS(self,fbra=None,fket=None,status='L',fname0='top',ifep=False):
      rank = self.comm.rank
      if rank == 0: print '\n[mpo_dmrg_class.checkMPS] size=',self.comm.size
      assert self.isym in [1,2] # Must have symmetry.
      assert not isinstance(fbra,basestring)
      assert not isinstance(fket,basestring)
      # Setup files
      if fbra is not None:
         fbmps = fbra
      else:
         if status == 'L':
            fbmps = h5py.File(self.path+'/lmps','r')
         elif status == 'R': 
            fbmps = h5py.File(self.path+'/rmps','r')
      if fket is not None: 
         fkmps = fket
      else:
	 fkmps = fbmps
      # bond dimension
      nsite = fbmps['nsite'].value
      bdim = map(lambda x:len(x),mpo_dmrg_io.loadQnums(fbmps))
      kdim = map(lambda x:len(x),mpo_dmrg_io.loadQnums(fkmps))
      # Quantities to be checked
      energy = None   
      ovlp = None
      psum = None
      fnameh = self.path+'/'+fname0+'_hop'
      fnames = self.path+'/'+fname0+'_sop'
      fnamep = self.path+'/'+fname0+'_pop'
      # Energy
      exphop = mpo_dmrg_init.genHops(self,fbmps,fkmps,fnameh,status)
      hpwts = self.fhop['wts'].value
      esum = numpy.dot(hpwts,exphop)
      esum = self.comm.reduce(esum,op=MPI.SUM,root=0)
      # S
      if rank == 0:
         ovlp = mpo_dmrg_init.genSops(self,fbmps,fkmps,fnames,status)
      # P
      if rank == 0 and self.ifs2proj: 
         exppop = mpo_dmrg_init.genPops(self,fbmps,fkmps,fnamep,status)
         psum = numpy.dot(self.qwts,exppop)
      # Final print
      if rank == 0:
         if psum is not None:
            energy = esum/psum
         else:
            energy = esum/ovlp
         print     
         print 'Summary for '+status+'-MPS:'
	 print ' npts   =',self.npts
         print ' <H(P)> =',esum
         print ' ovlp   =',ovlp
         print ' <(P)>  =',psum
         print ' Energy =',energy
         print ' Econst =',self.const
         print ' Etotal =',energy+self.const
         print ' bBdims =',bdim
         print ' kBdims =',kdim
      # Check Ecomponents
      if ifep:
	 if self.ifs2proj:
 	    npts = self.npts
	 else:
 	    npts = 1
         ecomp = self.comm.gather(exphop,root=0)
         opswt = self.comm.gather(hpwts,root=0)
	 if rank == 0:
	    ecomp = numpy.array(ecomp).flatten()
            opswt = numpy.array(opswt).flatten()
	    ecomp = ecomp*opswt
	    # Sum over grid points
            tmp = numpy.zeros(self.sbas)
            for ipt in range(npts):
               tmp += ecomp[ipt::npts]
            # Spatial orbitals
	    ecomp = numpy.zeros(self.nsite)
	    # Sum over spin cases
            for ispin in range(2):
               ecomp += tmp[ispin::2]
	    print '### Energy decompositions: <Hp> ###'
            print ' nsite =',self.nsite
	    print ' ecomp =',ecomp
            if psum is not None:
               self.ecomp = ecomp/psum
            else:
               self.ecomp = ecomp/ovlp
            print ' ecomp =',self.ecomp
            print ' etot  =',numpy.sum(ecomp)
      return energy,esum,ovlp,psum
