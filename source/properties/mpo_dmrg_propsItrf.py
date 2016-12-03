#!/usr/bin/env python
#
# Interface to compute expectation value of MPS
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
# 
# def seff(s2exp):
# def s2quad(npts,sval,sz1,sz2):
# def eval_P(dmrg,flmps,spinfo):
#
# def eval_S2Global(dmrg,flmps,fname='top',spinfo=None):
# def eval_Global(dmrg,flmps,key,fname='top',spinfo=None):
# def eval_Local(dmrg,flmps,groups,key,fname='top',spinfo=None):
# def eval_Local2(dmrg,flmps,groups,ikey,jkey,fname='top',spinfo=None):
# def eval_SiSj(dmrg,flmps,groups,fname='top',spinfo=None):
# def eval_rdm1BF(dmrg,flmps,fname='top',spinfo=None):
#
import math
import numpy
import mpo_dmrg_props
import mpo_dmrg_propsMPO
import mpo_dmrg_propsMPORpt
from mpodmrg.source.sysutil_include import dmrg_dtype
from mpodmrg.source.tools import smalld

# Effective spin quantum number
def seff(s2exp):
   return 0.5*(-1.0+math.sqrt(1.0+4.0*s2exp))

# Spin Projected case
def s2quad(npts,sval,sz1,sz2,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPORpt.s2quad] npts=',npts,' (s,m1,m2)=',(sval,sz1,sz2)
   assert npts > 0
   assert sval+1.e-4 > abs(sz1)
   assert sval+1.e-4 > abs(sz2)
   xts,wts = numpy.polynomial.legendre.leggauss(npts)
   xts = map(lambda x:math.acos(x),xts)
   fac = map(lambda x:smalld.value(sval,sz1,sz2,x),xts)
   wts = fac*wts
   wts = (2.*sval+1.)/2.0*wts
   xts = numpy.asarray(xts)
   wts = numpy.asarray(wts)
   if debug: print ' xts =',xts
   if debug: print ' wts =',wts
   return xts,wts

# Overlap/Normalizaiton <P>
def eval_P(dmrg,flmps,spinfo,debug=False):
   npts,sval,sz = spinfo
   xts,wts = s2quad(npts,sval,sz,sz)
   nsite = dmrg.nsite
   fname = './top'
   fop = mpo_dmrg_propsMPORpt.genMPO_Rpt(nsite,fname,xts,dmrg.ifQt)
   exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
   fop.close()
   denorm = numpy.sum(wts*exphop)
   if debug: print '<P>=',denorm
   return denorm,xts,wts

# <S2>	 
def eval_S2Global(dmrg,flmps,fname='top',spinfo=None):
   print '\n[mpo_dmrg_propsItrf.eval_S2Global]'
   nsite = dmrg.nsite
   if not dmrg.ifs2proj: 
      fop = mpo_dmrg_propsMPO.genMPO_S2Global(nsite,fname,dmrg.ifQt)
      exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
      fop.close()
      expect = exphop[0]
   else:
      denorm,xts,wts = eval_P(dmrg,flmps,spinfo)
      fop = mpo_dmrg_propsMPORpt.genMPO_S2GlobalRpt(nsite,fname,xts,dmrg.ifQt)
      exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
      fop.close()
      expect = numpy.sum(wts*exphop)/denorm
   return expect
   
# Global properties like <Omega>
def eval_Global(dmrg,flmps,key,fname='top',spinfo=None):
   nsite = dmrg.nsite
   if not dmrg.ifs2proj: 
      fop = mpo_dmrg_propsMPO.genMPO_Global(nsite,key,fname,dmrg.ifQt)
      exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
      fop.close()
      expect = exphop[0]
   else:
      denorm,xts,wts = eval_P(dmrg,flmps,spinfo)
      fop = mpo_dmrg_propsMPORpt.genMPO_GlobalRpt(nsite,key,fname,xts,dmrg.ifQt)
      exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
      fop.close()
      expect = numpy.sum(wts*exphop)/denorm
   return expect

# Local properties like {<Ni>}
def eval_Local(dmrg,flmps,groups,key,fname='top',spinfo=None):
   nsite = dmrg.nsite
   ngroup = len(groups)
   expect = numpy.zeros(ngroup,dtype=dmrg_dtype)
   if not dmrg.ifs2proj: 
      for idx,ig in enumerate(groups):
         fop = mpo_dmrg_propsMPO.genMPO_Local(nsite,ig,key,fname,dmrg.ifQt)
         exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
         fop.close()
	 expect[idx] = exphop[0]
   else:
      denorm,xts,wts = eval_P(dmrg,flmps,spinfo)
      for idx,ig in enumerate(groups):
         fop = mpo_dmrg_propsMPORpt.genMPO_LocalRpt(nsite,ig,key,fname,xts,dmrg.ifQt)
         exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
         fop.close()
         expect[idx] = numpy.sum(wts*exphop)/denorm
   return expect

# Local2 properties like {<Ni*Nj>}
def eval_Local2(dmrg,flmps,groups,ikey,jkey,fname='top',spinfo=None):
   nsite = dmrg.nsite
   ngroup = len(groups)
   expect = numpy.zeros((ngroup,ngroup),dtype=dmrg_dtype)
   if not dmrg.ifs2proj:
      for idx,ig in enumerate(groups):
         for jdx,jg in enumerate(groups):
            fop = mpo_dmrg_propsMPO.genMPO_Local2(nsite,ig,jg,ikey,jkey,1.0,fname,dmrg.ifQt)
            exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
            fop.close()
	    expect[idx,jdx] = exphop[0]
   else:
      denorm,xts,wts = eval_P(dmrg,flmps,spinfo)
      for idx,ig in enumerate(groups):
         for jdx,jg in enumerate(groups):
            fop = mpo_dmrg_propsMPORpt.genMPO_Local2Rpt(nsite,ig,jg,ikey,jkey,1.0,fname,xts,dmrg.ifQt)
            exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
            fop.close()
            expect[idx,jdx] = numpy.sum(wts*exphop)/denorm
   return expect 
  
# <vec{SA}*vec{SB}> = 0.5*(<SA+SB-> + <SA-SB+>) + SAz*SBz
def eval_SiSj(dmrg,flmps,groups,fname='top',spinfo=None):
   nsite = dmrg.nsite
   ngroup = len(groups)
   expect = numpy.zeros((ngroup,ngroup),dtype=dmrg_dtype)
   if not dmrg.ifs2proj:
      for idx,ig in enumerate(groups):
         for jdx,jg in enumerate(groups):
            fop = mpo_dmrg_propsMPO.genMPO_SiSj(nsite,ig,jg,fname,dmrg.ifQt)
            exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
            fop.close()
            expect[idx,jdx] = numpy.sum(exphop)
   else:
      npts,sval,sz = spinfo
      denorm,xts,wts = eval_P(dmrg,flmps,spinfo)
      for idx,ig in enumerate(groups):
         for jdx,jg in enumerate(groups):
            fop = mpo_dmrg_propsMPORpt.genMPO_SiSjRpt(nsite,ig,jg,fname,xts,dmrg.ifQt)
            exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
            fop.close()
            expect[idx,jdx] = numpy.sum(wts*(exphop[:npts]+exphop[npts:2*npts]+exphop[2*npts:]))/denorm
   return expect

# Brute force Gamma[p,q]
def eval_rdm1BF(dmrg,flmps,fname='top',spinfo=None):
   nsite = dmrg.nsite
   if not dmrg.ifs2proj:
      rdm1 = numpy.zeros((nsite,nsite,2),dtype=dmrg_dtype)
      for p in range(nsite):
         for q in range(nsite):
            fop = mpo_dmrg_propsMPO.genMPO_Epq(nsite,p,q,fname,dmrg.ifQt)
            exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
            fop.close()
            rdm1[p,q] = exphop
      rdm1 = rdm1.transpose(2,0,1)
      rdm1t = rdm1[0]+rdm1[1]  
      rdm1s = 0.5*(rdm1[0]-rdm1[1])
   else:
      npts,sval,sz = spinfo
      denorm,xts,wts = eval_P(dmrg,flmps,spinfo)
      # RDM1 = <Epq>
      rdm1t = numpy.zeros((nsite,nsite),dtype=dmrg_dtype)
      for p in range(nsite):
         for q in range(nsite):
            fop = mpo_dmrg_propsMPORpt.genMPO_EpqRpt(nsite,p,q,fname,xts,dmrg.ifQt)
            exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
            fop.close()
            rdm1t[p,q] = numpy.sum(wts*(exphop[:npts]+exphop[npts:]))/denorm
      # Spin density matrix 0.5*<EpqA-EpqB>
      # Generation of additional quadrature points
      sz1 = sz-1.0
      xts1,wts1 = s2quad(npts,sval,sz1,sz)
      rdm1s = numpy.zeros((nsite,nsite),dtype=dmrg_dtype)
      for p in range(nsite):
         for q in range(nsite):
   	    # We use the same quadrature point		 
            fop = mpo_dmrg_propsMPORpt.genMPO_TpqRpt(nsite,p,q,fname,xts,dmrg.ifQt)
            exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
            fop.close()
            # Summarize results with different weights - wts,wts1
	    term1 = numpy.sum(wts*(exphop[:npts]-exphop[npts:2*npts]))/math.sqrt(2)
            term2 = numpy.sum(wts1*exphop[2*npts:])
            term1 = term1*sval/(sval+1.0)
            term2 = term2*math.sqrt(sval)/(sval+1.0)
            tpq = (term1+term2)/math.sqrt(2)
            rdm1s[p,q] = tpq/denorm
   # Check skewness
   skewt = numpy.linalg.norm(rdm1t-rdm1t.T.conj())
   print ' skewness of rdm1t =',skewt
   assert skewt < 1.e-12
   skews = numpy.linalg.norm(rdm1s-rdm1s.T.conj())
   print ' skewness of rdm1s =',skews
   assert skews < 1.e-12
   return rdm1t,rdm1s
