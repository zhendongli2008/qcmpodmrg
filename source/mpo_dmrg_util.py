#!/usr/bin/env python
#
# Initialization and some utils for DMRG
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def s2quad(dmrg,sval=0,sz=0):
#
# def mps0random(dphys,dmax):
# def mps0mixed(dphys):
# def mps0occun(dphys,occun,isym):
#
# def initMPS(dmrg,debug=False):
# def initOps(dmrg,fmps):
# def initE0pt(dmrg,fmps):
# def genCIHamiltonian(dmrg):
# def vpt(dmrg):
# def ci(dmrg):
# 
import h5py
import copy
import math
import numpy
import scipy.linalg
import mpo_dmrg_io
import mpo_dmrg_opt
import mpo_dmrg_init
from mpi4py import MPI
from tools import smalld
from sysutil_include import dmrg_dtype,dmrg_mtype

# Generation of quadrature (pts,wts)
def s2quad(dmrg,sval=0,sz=0):
   if dmrg.comm.rank == 0: 
      print '\n[mpo_dmrg_util.s2quad]'
      print ' sval =',sval,' sz =',sz,' npts =',dmrg.npts
      print ' quad =',dmrg.quad
   assert sval+1.e-4 > abs(sz)
   if dmrg.quad == 'Simpson':
      # Weights
      def wfun(phi,sval,sz):
         wt = (2.*sval+1.)/2.0*math.sin(phi)*smalld.value(sval,sz,sz,phi)
         return wt
      #
      # To use Simpson rules, we use even no. of grids
      # https://en.wikipedia.org/wiki/Simpson%27s_rule
      #
      # \int_a^b f(x) \, dx\approx
      # \tfrac{h}{3}\bigg[f(x_0)+4f(x_1)+2f(x_2)+4f(x_3)+2f(x_4)+\cdots+4f(x_{n-1})+f(x_n)\bigg]
      # =\tfrac{h}{3}\sum_{j=1}^{n/2}\bigg[f(x_{2j-2})+4f(x_{2j-1})+f(x_{2j})\bigg].
      #
      # Use shifted gauss-legendre quadrature in future?
      #
      npoints = dmrg.npts-1
      assert npoints%2 == 0 and npoints>0
      h = numpy.pi/(3.0*npoints)
      if dmrg.comm.rank == 0: print ' width of internal in [0,pi]=',h
      npts = npoints+1
      wts = numpy.zeros(npts)
      ###Trapezoidal rule
      #wts[:] = 2.0
      #wts[0] = 1.0
      #wts[-1] = 1.0
      #wts = wts*numpy.pi/(2.0*npoints)
      ###Simpson rule
      wts[0::2] = 2.0
      wts[1::2] = 4.0
      wts[0]  = 1.0
      wts[-1] = 1.0
      if dmrg.comm.rank == 0: print ' Simpson coeff=',wts
      wts = wts*h
      # Sites
      xts = numpy.linspace(0,numpy.pi,num=npts)
      wts = wts*numpy.array(map(lambda x:wfun(x,sval,sz),xts))
   elif dmrg.quad == 'GaussLegendre':
      xts,wts = numpy.polynomial.legendre.leggauss(dmrg.npts)
      xts = map(lambda x:math.acos(x),xts)
      fac = map(lambda x:smalld.value(sval,sz,sz,x),xts)
      wts = fac*wts
      wts = (2.*sval+1.)/2.0*wts
   dmrg.qpts = xts
   dmrg.qwts = wts
   if dmrg.comm.rank == 0: 
      print ' qpts =',dmrg.qpts
      print ' qwts =',dmrg.qwts
      print
   return xts,wts

# A random state
def mps0random(dphys,dmax):
   nsite = len(dphys)
   sites = [None]*nsite
   for isite in range(nsite):
      dloc = dphys[isite]
      if isite==0:
	 sites[isite] = numpy.random.uniform(-1,1,(1,dloc,dmax))
      elif isite==nsite-1:
	 sites[isite] = numpy.random.uniform(-1,1,(dmax,dloc,1))
      else:
	 sites[isite] = numpy.random.uniform(-1,1,(dmax,dloc,dmax))
   return sites

# A product state, |mixed>=Prod_i (|0i>+|1i>+...)/sqrt(Ni) 
def mps0mixed(dphys):
   nsite = len(dphys)
   sites = [None]*nsite
   for isite in range(nsite):
      dloc = dphys[isite]
      vals = 1.0/math.sqrt(dloc)
      tmp = numpy.zeros((1,dloc,1),dtype=dmrg_dtype)
      tmp[:] = vals
      sites[isite] = tmp.copy()
   return sites

# A product state with given spin-orbital occupation pattern 
def mps0occun(dphys,occun,isym=2):
   nsite = len(dphys)
   sites = [None]*nsite
   assert len(occun) == 2*nsite
   qnuml = [None]*(nsite+1)
   qnuml[0] = [[0.,0.]]
   nelec = 0.
   mspin = 0.0
   for isite in range(nsite):
      dloc = dphys[isite]
      na = occun[2*isite]
      nb = occun[2*isite+1]
      thresh = 1.e-4
      tmp = numpy.zeros(dloc,dtype=dmrg_dtype)
      if abs(na)<thresh and abs(nb)<thresh:
         tmp[0] = 1.0
      elif abs(na)<thresh and abs(nb-1.0)<thresh:
         nelec += 1.0
         mspin += -0.5
         tmp[1] = 1.0
      elif abs(na-1.0)<thresh and abs(nb)<thresh:
         nelec += 1.0
         mspin += 0.5
         tmp[2] = 1.0
      elif abs(na-1.0)<thresh and abs(nb-1.0)<thresh:
         nelec += 2.0
         tmp[3] = 1.0
      sites[isite] = tmp.reshape(1,dloc,1)
      qnuml[isite+1] = [[nelec,mspin]]
   # Right quantum numbers
   qnumr = [None]*(nsite+1)
   for isite in range(nsite+1):
      qnumr[isite] = [[nelec-qnuml[isite][0][0],mspin-qnuml[isite][0][1]]]
   # Particle number symmetry - N   
   if isym == 1:
      qnuml = [map(lambda x:[x[0]],qnumIbd) for qnumIbd in qnuml]
      qnumr = [map(lambda x:[x[0]],qnumIbd) for qnumIbd in qnumr]
   return qnuml,qnumr,sites

# initMPS
def initMPS(dmrg,debug=False):
   if dmrg.comm.rank == 0: print '[mpo_dmrg_util.initMPS]'
   # Product state MPS0
   if dmrg.occun is None:
      mps0 = mps0mixed(dmrg.dphys)
      ifsym = False
   else:
      qnuml,qnumr,mps0 = mps0occun(dmrg.dphys,dmrg.occun,dmrg.isym)
      dmrg.qnuml = copy.deepcopy(qnuml) 
      dmrg.qnumr = copy.deepcopy(qnumr) 
      ifsym = True
   # DUMP
   fmps0 = h5py.File(dmrg.path+'/mps0','w')
   mpo_dmrg_io.dumpMPS(fmps0,mps0,icase=0)
   if debug: dmrg.checkMPS(fmps0)
   # init operators from mps
   fnamel = dmrg.path+'/lop'
   fnamer = dmrg.path+'/rop'
   mpo_dmrg_init.genBops(dmrg,fnamel,dmrg.nops,-1,ifslc=False)
   mpo_dmrg_init.genHops(dmrg,fmps0,fmps0,fnamer,'R')
   # Initialize MPS
   sitelst = range(dmrg.nsite)
   if debug:
      mpo_dmrg_init.genHops(dmrg,fmps0,fmps0,fnamel,'L')
      mpo_dmrg_opt.sweep(dmrg,sitelst,dmrg.initNcsite,'R',ifsym=ifsym)
   # Initialization 
   mpo_dmrg_opt.sweep(dmrg,sitelst,dmrg.initNcsite,'L',ifsym=ifsym)
   # Nosym sweeps
   for isweep in range(dmrg.initNsweep):
      mpo_dmrg_opt.sweep(dmrg,sitelst,dmrg.initNcsite,'R',ifsym=ifsym)
      mpo_dmrg_opt.sweep(dmrg,sitelst,dmrg.initNcsite,'L',ifsym=ifsym)
   fmps0.close()
   return 0

# InitOperators for DMRG
def initOps(dmrg,fmps):
   rank = dmrg.comm.rank 
   if rank == 0: print '\n[mpo_dmrg_util.initOps]'
   #--------------------------------------------------------
   # Initialization-0: normal initialization (lrop)
   # H0 or H for LHS of eigenvalue or linear equations [PT]
   #--------------------------------------------------------
   if dmrg.ifpt and dmrg.ifH0: dmrg.fdopXfhop()
   fnamel = dmrg.path+'/lop'
   fnamer = dmrg.path+'/rop'
   mpo_dmrg_init.genBops(dmrg,fnamer,dmrg.nops,dmrg.nsite,ifslc=True)
   mpo_dmrg_init.genHops(dmrg,fmps,fmps,fnamel,'L')
   if dmrg.ifpt and dmrg.ifH0: dmrg.fdopXfhop()
   #--------------------------------------------------------
   # Initialization-0: normal initialization (lrpop)
   # Renormalized operators for <L|P|L>.
   #--------------------------------------------------------
   if rank == 0 and dmrg.ifs2proj: 
      fnamelp = dmrg.path+'/lpop'
      fnamerp = dmrg.path+'/rpop'
      mpo_dmrg_init.genBops(dmrg,fnamerp,dmrg.npts,dmrg.nsite,ifslc=False)
      mpo_dmrg_init.genPops(dmrg,fmps,fmps,fnamelp,'L')
   #--------------------------------------------------------
   if dmrg.ifpt: dmrg.ifex = True
   dmrg.nref = len(dmrg.wfex)
   # 2016.09.13: By default, all MPS in wfex is in left canonical form!
   if len(dmrg.wfex_canon) == 0: dmrg.wfex_canon = [True]*dmrg.nref
   fbra = fmps
   #--------------------------------------------------------
   # Initialization-1: initialization (lrsop/lrpop)
   # Excited states: Only rank-0 compute overlaps
   #--------------------------------------------------------
   if rank == 0 and dmrg.ifex:
      for iref in range(dmrg.nref):
	 print '\n ### Ovlps: iref =',iref,' rank =',rank,'###'
         fket = dmrg.wfex[iref]
	 # <A|B>
         if not dmrg.ifs2proj: 
            fnamel  = dmrg.path+'/ref'+str(iref)+'_lsop'
            fnamer  = dmrg.path+'/ref'+str(iref)+'_rsop'
            mpo_dmrg_init.genBmat(dmrg,fnamer,dmrg.nsite)
	    mpo_dmrg_init.genSops(dmrg,fbra,fket,fnamel,'L')
	 # <A|P|B>
	 else:
            fnamelp = dmrg.path+'/ref'+str(iref)+'_lpop'
            fnamerp = dmrg.path+'/ref'+str(iref)+'_rpop'
            mpo_dmrg_init.genBops(dmrg,fnamerp,dmrg.npts,dmrg.nsite,ifslc=False)
            mpo_dmrg_init.genPops(dmrg,fbra,fket,fnamelp,'L')
   #--------------------------------------------------------
   # RHS of PT equation: <fmps|H|MPS[i]>
   if dmrg.ifpt:
      # Generate E0	   
      initE0pt(dmrg,fmps)
      #--------------------------------------------------------
      # Initialization-2: initialization (lrhop)
      # Blocks for <Psi|H|Psi[i]>
      #--------------------------------------------------------
      ln = '#'*10
      for iref in range(dmrg.nref):
	 print '\n'+ln+' <Psi|H|Psi[i]>: iref =',iref,' rank=',rank,ln
         fket = dmrg.wfex[iref]
	 # <A|H|B>
         fnamel = dmrg.path+'/ref'+str(iref)+'_lhop'
         fnamer = dmrg.path+'/ref'+str(iref)+'_rhop'
	 mpo_dmrg_init.genBops(dmrg,fnamer,dmrg.nops,dmrg.nsite,ifslc=True)
	 mpo_dmrg_init.genHops(dmrg,fbra,fket,fnamel,'L')
      #--------------------------------------------------------
      # Initialization-3: initialization (lrdop)
      # Blocks for <Psi|H0|Psi[i]>
      #--------------------------------------------------------
      if dmrg.ifH0:
         dmrg.fdopXfhop()
         for iref in range(dmrg.nref):
   	    print '\n'+ln+' <Psi|H0|Psi[i]>: iref =',iref,' rank=',rank,ln
            fket = dmrg.wfex[iref]
   	    # <A|H|B>
            fnamel = dmrg.path+'/ref'+str(iref)+'_ldop'
            fnamer = dmrg.path+'/ref'+str(iref)+'_rdop'
   	    mpo_dmrg_init.genBops(dmrg,fnamer,dmrg.nops,dmrg.nsite,ifslc=True)
   	    mpo_dmrg_init.genHops(dmrg,fbra,fket,fnamel,'L')
         dmrg.fdopXfhop()
      #--------------------------------------------------------
   return 0

# E0 = {Edmrg/<Hd>}
def initE0pt(dmrg,fmps):
   rank = dmrg.comm.rank 
   # <I|H0|J>
   if dmrg.ifH0: dmrg.fdopXfhop()
   hmat,smat = genCIHamiltonian(dmrg)
   if dmrg.ifH0: dmrg.fdopXfhop()
   # ONLY rank-0 have Hmat and Smat
   if rank == 0:
      if dmrg.ifptn:
         dmrg.e0 = hmat[0,0]/smat[0,0]
      else:
         assert len(dmrg.coef) == dmrg.nref
         dmrg.coef = numpy.array(dmrg.coef)
         dmrg.e0 = reduce(numpy.dot,(dmrg.coef.T.conj(),hmat,dmrg.coef))/\
           	   reduce(numpy.dot,(dmrg.coef.T.conj(),smat,dmrg.coef))
   # For each rank, only the last one |n-1> 
   # has nonzero coeff used for BVec (V|n-1>).
   if dmrg.ifptn:
      dmrg.coef = numpy.zeros(dmrg.nref,dtype=dmrg_dtype)
      dmrg.coef[-1] = 1.0
   # Determine Et used in PT
   if dmrg.ifE0:
      dmrg.et = dmrg.e0
   else:
      energy,esum,ovlp,psum = dmrg.checkMPS(fmps,fmps,ifprt=False)
      if rank == 0:
         dmrg.e1 = energy-dmrg.e0
         dmrg.et = energy-dmrg.emix*dmrg.e1
         if dmrg.ifs2proj: 
            dmrg.n0 = 1/math.sqrt(smat[0,0])
         print  
	 print '='*40
         print 'Summary of partition: ifH0 =',dmrg.ifH0
	 print '='*40
	 print ' emix = ',dmrg.emix
         print ' e0 = ',dmrg.e0
         print ' e1 = ',dmrg.e1
	 print ' eH = ',energy
         print ' et = ',dmrg.et
         print ' n0 = ',dmrg.n0
   # Broadcast
   dmrg.e0 = dmrg.comm.bcast(dmrg.e0)
   dmrg.e1 = dmrg.comm.bcast(dmrg.e1)
   dmrg.et = dmrg.comm.bcast(dmrg.et)
   dmrg.n0 = dmrg.comm.bcast(dmrg.n0)
   return 0      

# <I|H|J> & <I|(P)|J> 
def genCIHamiltonian(dmrg):
   rank = dmrg.comm.rank 
   if rank == 0: print '\n[mpo_dmrg_ex.genCIHamiltonian] nref=',dmrg.nref	
   dmrg.nref = len(dmrg.wfex)
   hmat = numpy.zeros((dmrg.nref,dmrg.nref),dtype=dmrg_dtype)
   smat = numpy.zeros((dmrg.nref,dmrg.nref),dtype=dmrg_dtype)
   for iref in range(dmrg.nref):
      fbra = dmrg.wfex[iref]
      for jref in range(dmrg.nref):
         fket = dmrg.wfex[jref]
         energy,esum,ovlp,psum = dmrg.checkMPS(fbra,fket,ifprt=False)
	 if rank == 0:
	    print		 
	    print '-'*92
	    print '<i|O|j>:',(iref,jref),' esum,ovlp,psum=',esum,ovlp,psum
	    print '-'*92
	 hmat[iref,jref] = esum
	 if not dmrg.ifs2proj:
            smat[iref,jref] = ovlp
    	 else:
            smat[iref,jref] = psum
   return hmat,smat

# LHS & RHS of PT2 equation: <fmps|H0|fmps> & <fmps|H|MPS[i]>
def vpt(dmrg):
   rank = dmrg.comm.rank 
   # 1. LHS: Prepare operators for H0
   dmrg.fdopXfhop()
   hmat,smat = genCIHamiltonian(dmrg)
   dmrg.fdopXfhop()
   # 2. E0 = <Psi0|H|Psi0>
   dmrg.nref = len(dmrg.wfex)
   fbra = dmrg.wfex[0]
   energy,esum,ovlp,psum = dmrg.checkMPS(fbra,fbra,ifprt=False)
   # Normalization constant
   if rank==0: 
      dmrg.e0 = hmat[0,0]/smat[0,0]
      dmrg.e1 = energy-dmrg.e0
      dmrg.et = energy-dmrg.emix*dmrg.e1
      dmrg.n0 = 1.0/math.sqrt(smat[0,0])
   # 3. RHS = <0|H|psi1>
   v01 = numpy.zeros(dmrg.nref)
   for iref in range(1,dmrg.nref):
      print '\n ### Hams: iref =',iref,' rank= ',rank,'###'
      fket = dmrg.wfex[iref]
      # <A|H|B>
      fnamel = dmrg.path+'/ref'+str(iref)+'_lhop'
      fnamer = dmrg.path+'/ref'+str(iref)+'_rhop'
      mpo_dmrg_init.genBops(dmrg,fnamer,dmrg.nops,dmrg.nsite,ifslc=True)
      exphop = mpo_dmrg_init.genHops(dmrg,fbra,fket,fnamel,'L')
      hpwts = dmrg.fhop['wts'].value
      esum = numpy.dot(hpwts,exphop)
      esum = dmrg.comm.reduce(esum,op=MPI.SUM,root=0)
      if rank == 0: v01[iref] = dmrg.n0*esum.real
   # 4. Solve <Psi1|H0E0|Psi1> = -<Psi1|V|Psi0>   
   if rank == 0:
      ndim = hmat.shape[0]
      Amat = (hmat-dmrg.et*smat)[numpy.ix_(range(1,ndim),range(1,ndim))]
      bvec = v01[1:]
      cvec = scipy.linalg.solve(Amat,-bvec)
      e2o = cvec.dot(bvec)
      e2s = numpy.sum(bvec)
      lhs = numpy.sum(Amat)
      e2v = lhs+2.0*e2s
      # L = <Psi1|H0-E0|Psi1>+2<Psi1|V|0>
      print
      print '='*40
      print 'Summary of PT results: ifH0 =',dmrg.ifH0
      print '='*40
      print ' h0mat = \n',hmat
      print ' smat = \n',smat
      print ' emix =',dmrg.emix
      print ' e0 = ',dmrg.e0
      print ' e1 = ',dmrg.e1
      print ' eH = ',energy
      print ' et = ',dmrg.et
      print ' n0 = ',dmrg.n0
      print ' ndim =',ndim-1
      print ' Amat = \n',Amat
      print ' bvec = \n',bvec
      print ' (1) e2[ Sum of RHS ] =',e2s
      print ' (2) e2[ Functional ] =',e2v
      print '     * difference =',e2v-e2s
      print '     * <Psi1|H0-E0|Psi1> =',lhs
      print ' (3) e2[ ROptimized ] =',e2o
      print '     * difference =',e2o-e2s
      print '     * cvec =',cvec
      e2 = e2o
      #==========================================
      # If the difference is large, then this
      # indicate in the solution of psi1, the
      # accuracy [D1] is not sufficiently high.
      #==========================================
   else:
      e2 = None
      cvec = None
   return e2,cvec

def ci(dmrg):
   rank = dmrg.comm.rank 
   # MPSfile
   hmat,smat = genCIHamiltonian(dmrg)
   ndim = hmat.shape[0]
   #------------------------------------------
   if dmrg.ifpt and dmrg.ifH0:
      dmrg.fdopXfhop()
      hmat0,smat0 = genCIHamiltonian(dmrg)
      dmrg.fdopXfhop()
      if rank == 0: 
         print ' H0[I,J]=\n',hmat0
         print ' S0[I,J]=\n',smat0
   #------------------------------------------
   e = numpy.zeros(ndim,dtype=numpy.float_)
   v = numpy.zeros((ndim,ndim),dtype=dmrg_dtype)
   if rank == 0: 
      print ' H[I,J]=\n',hmat
      print ' S[I,J]=\n',smat
      e,v = scipy.linalg.eigh(hmat,smat)
      n0 = 1.0/math.sqrt(smat[0,0])
      e0 = hmat[0,0]/smat[0,0]
      e2 = hmat[0,1]*n0
      e3 = 0.0
      # PRINT
      print ' CI energies=',e
      print ' Coeff[0] =',v[:,0]
      print 'PT analysis:'
      print ' Normalization =',n0 
      print ' E[0-1] =',e0 
      print ' E[2]   =',e0+e2,e2
      if dmrg.ifpt and dmrg.ifH0:
         e3 = hmat[1,1]-hmat0[1,1]
         print ' E[3]   =',e0+e2+e3,e3
      print ' E[res] =',e[0],e[0]-e0-e2-e3
      print ' E[ful] =',e[0],e[0]-e0
   dmrg.comm.Bcast([e,MPI.DOUBLE])
   dmrg.comm.Bcast([v,dmrg_mtype])
   # CI case
   if not ifpt:
      return e,v
   # n-th order perturbation theory
   else:
      if rank == 0:
         # If H0 = PHP + QHQ is used.
         if not dmrg.ifH0: hmat0 = hmat.copy()
         # Note that in our case, we use PEP+QHdQ!
         if not dmrg.ifE0: hmat0[0,0] = hmat[0,0]-dmrg.emix*(hmat[0,0]-hmat0[0,0])
         hmat0[0,1:] = 0.
         hmat0[1:,0] = 0.
         vmat = hmat - hmat0
         print ' Hmat :\n',hmat
         print ' H0mat:\n',hmat0 
         print ' Vmat :\n',vmat 
         e0 = hmat0[0,0]
         e1 = hmat[0,0] - hmat0[0,0]
         # Higher-order pt energies
         enlst = [e0,e1] + [0]*(2*norder)
         for n in range(1,norder+1):
            # e[2n]
            enlst[2*n] = vmat[n-1,n]
            for k in range(1,n+1):
               for j in range(1,n):
                  #print 'e-k,j',(n,2*n),(2*n-k-j,k,j)     
                  enlst[2*n] -= enlst[2*n-k-j]*smat[k,j]
            # e[2n+1]
            enlst[2*n+1] = vmat[n,n]
            for k in range(1,n+1):
               for j in range(1,n+1):
                  #print 'o-k,j',(n,2*n+1),(2*n+1-k-j,k,j)
                  enlst[2*n+1] -= enlst[2*n+1-k-j]*smat[k,j]
         print '\n Analysis of PT series:'
         esum = 0.
         for n in range(2*norder+2):
            esum += enlst[n]	    
            print ' e[%d] = (%20.12f,%20.12f)'%(n,enlst[n],esum)
         enlst = numpy.array(enlst).real
      else:
         enlst = numpy.zeros(2*norder+2)
      dmrg.comm.Bcast([enlst,MPI.DOUBLE])
      enlst = list(enlst)
      return enlst
