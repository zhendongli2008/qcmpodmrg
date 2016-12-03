#!/usr/bin/env python
#
# MPO for properties with SP-MPS
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def genMPO_Rpt(nsite,fname,xts,ifQt,debug=False):
# def genMPO_EpqRpt(nsite,p,q,fname,xts,ifQt,debug=False):
# def genMPO_TpqRpt(nsite,p,q,fname,xts,ifQt,debug=False):
#
# def genMPO_S2GlobalRpt(nsite,fname,xts,ifQt,debug=False):
# def genMPO_GlobalRpt(nsite,key,fname,xts,ifQt,debug=False):
#
# def genMPO_LocalRpt(nsite,ig,key,fname,xts,ifQt,debug=False):
# def genMPO_Local2Rpt(nsite,ig,jg,ikey,jkey,fac,fname,xts,ifQt,debug=False):
# def genMPO_SiSjRpt(nsite,ig,jg,fname,xts,ifQt,debug=False):
# 
import os
import time
import h5py
import math
import numpy
from mpodmrg.source import mpo_dmrg_opers
from mpodmrg.source import mpo_dmrg_spinopers
from mpodmrg.source.qtensor import qtensor_opers
from mpodmrg.source.qtensor import qtensor_spinopers

# DUMP Wx[isite] for R(theta)
# Although W is indepdent of isite, but for consistency
# we generate them for all sites.
def genMPO_Rpt(nsite,fname,xts,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPORpt.genMPO_Rpt] fname=',fname
   t0 = time.time()
   fop = h5py.File(fname,'w')
   npt = len(xts)
   nop = npt
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 for ipt in range(npt):
            rop = mpo_dmrg_opers.genExpISyPhi(xts[ipt])
	    grp['op'+str(ipt)] = rop
      else:
	 for ipt in range(npt):
	    rop = qtensor_opers.genExpISyPhiQt(xts[ipt])
	    rop.dump(grp,'op'+str(ipt))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_Rpt = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] for Epq = pa+*qa + pb+*qb
def genMPO_EpqRpt(nsite,p,q,fname,xts,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPORpt.genMPO_EpqRpt] fname=',fname
   t0 = time.time()
   fop = h5py.File(fname,'w')
   npt = len(xts)
   nop = 2*npt
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
         # {pa+*qa,pb+*qb} 
	 for is1 in [0,1]:
	    oplst = [[2*p+is1,1],[2*q+is1,0]]
	    for ipt in range(npt):
	       cop = mpo_dmrg_opers.genElemProductRSpatial(oplst,isite,xts[ipt])
	       grp['op'+str(is1*npt+ipt)] = cop
      else:
         # {pa+*qa,pb+*qb} 
	 for is1 in [0,1]:
	    oplst = [[2*p+is1,1],[2*q+is1,0]]
	    for ipt in range(npt):
	       wop = qtensor_opers.genElemProductRSpatialQt(oplst,isite,xts[ipt])
	       wop.dump(grp,'op'+str(is1*npt+ipt))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_EpqRpt = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] for Tpq = {pa+*qa, pb+*qb, pa+*qb}
def genMPO_TpqRpt(nsite,p,q,fname,xts,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPORpt.genMPO_TpqRpt] fname=',fname
   t0 = time.time()
   fop = h5py.File(fname,'w')
   npt = len(xts)
   nop = 3*npt
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 # pa+*qa
	 oplst = [[2*p+0,1],[2*q+0,0]]
         for ipt in range(npt):
	    cop = mpo_dmrg_opers.genElemProductRSpatial(oplst,isite,xts[ipt])
            grp['op'+str(0*npt+ipt)] = cop
	 # pb+*qb   
	 oplst = [[2*p+1,1],[2*q+1,0]]
         for ipt in range(npt):
	    cop = mpo_dmrg_opers.genElemProductRSpatial(oplst,isite,xts[ipt])
            grp['op'+str(1*npt+ipt)] = cop
	 # pa+*pb   
	 oplst = [[2*p+0,1],[2*q+1,0]]
         for ipt in range(npt):
	    cop = mpo_dmrg_opers.genElemProductRSpatial(oplst,isite,xts[ipt])
            grp['op'+str(2*npt+ipt)] = cop
      else:
	 # pa+*qa
	 oplst = [[2*p+0,1],[2*q+0,0]]
         for ipt in range(npt):
	    cop = qtensor_opers.genElemProductRSpatialQt(oplst,isite,xts[ipt])
            cop.dump(grp,'op'+str(0*npt+ipt))
	 # pb+*qb   
	 oplst = [[2*p+1,1],[2*q+1,0]]
         for ipt in range(npt):
	    cop = qtensor_opers.genElemProductRSpatialQt(oplst,isite,xts[ipt])
            cop.dump(grp,'op'+str(1*npt+ipt))
	 # pa+*pb   
	 oplst = [[2*p+0,1],[2*q+1,0]]
         for ipt in range(npt):
	    cop = qtensor_opers.genElemProductRSpatialQt(oplst,isite,xts[ipt])
            cop.dump(grp,'op'+str(2*npt+ipt))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_TpqRpt = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] 
def genMPO_S2GlobalRpt(nsite,fname,xts,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPORpt.genMPO_S2GlobalRpt] fname=',fname
   t0 = time.time()
   fop = h5py.File(fname,'w')
   npt = len(xts)
   nop = 1*npt
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 cop = mpo_dmrg_spinopers.genS2GlobalSpatial(nsite,isite)
         for ipt in range(npt):
            rop = mpo_dmrg_opers.genExpISyPhiMat(xts[ipt])
      	    wop = numpy.tensordot(cop,rop,axes=([3],[0]))
            grp['op'+str(ipt)] = wop
      else:
         for ipt in range(npt):
	    cop = qtensor_spinopers.genS2GlobalRSpatialQt(nsite,isite,xts[ipt])
            cop.dump(grp,'op'+str(ipt))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_S2GlobalRpt = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] 
def genMPO_GlobalRpt(nsite,key,fname,xts,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPORpt.genMPO_GlobalRpt] fname=',fname
   t0 = time.time()
   fop = h5py.File(fname,'w')
   npt = len(xts)
   nop = 1*npt
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 cop = mpo_dmrg_spinopers.genGlobalSpatial(nsite,isite,key)
         for ipt in range(npt):
            rop = mpo_dmrg_opers.genExpISyPhiMat(xts[ipt])
      	    wop = numpy.tensordot(cop,rop,axes=([3],[0]))
            grp['op'+str(ipt)] = wop
      else:
         for ipt in range(npt):
	    cop = qtensor_spinopers.genGlobalRSpatialQt(nsite,isite,key,xts[ipt])
            cop.dump(grp,'op'+str(ipt))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_GlobalRpt = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] 
def genMPO_LocalRpt(nsite,ig,key,fname,xts,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPORpt.genMPO_LocalRpt] fname=',fname,' ig=',ig,' key=',key
   t0 = time.time()
   fop = h5py.File(fname,'w')
   npt = len(xts)
   nop = 1*npt
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 cop = mpo_dmrg_spinopers.genLocalSpatial(nsite,isite,ig,key)
         for ipt in range(npt):
            rop = mpo_dmrg_opers.genExpISyPhiMat(xts[ipt])
      	    wop = numpy.tensordot(cop,rop,axes=([3],[0]))
            grp['op'+str(ipt)] = wop
      else:
         for ipt in range(npt):
	    cop = qtensor_spinopers.genLocalRSpatialQt(nsite,isite,ig,key,xts[ipt])
            cop.dump(grp,'op'+str(ipt))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_LocalRpt = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] 
def genMPO_Local2Rpt(nsite,ig,jg,ikey,jkey,fac,fname,xts,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPORpt.genMPO_Local2Rpt] fname=',fname
   if debug: print ' ig=',ig,' jg=',jg,' ikey/jkey=',(ikey,jkey)
   t0 = time.time()
   fop = h5py.File(fname,'w')
   npt = len(xts)
   nop = 1*npt
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,ikey,jkey,fac)
         for ipt in range(npt):
            rop = mpo_dmrg_opers.genExpISyPhiMat(xts[ipt])
      	    wop = numpy.tensordot(cop,rop,axes=([3],[0]))
            grp['op'+str(ipt)] = wop
      else:
         for ipt in range(npt):
	    cop = qtensor_spinopers.genLocal2RSpatialQt(nsite,isite,ig,jg,ikey,jkey,fac,xts[ipt])
            cop.dump(grp,'op'+str(ipt))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_Local2Rpt = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] for SiSj = 0.5*(Si+*Sj-+Si-*Sj) + Szi*Szj
def genMPO_SiSjRpt(nsite,ig,jg,fname,xts,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPORpt.genMPO_SiSjRpt] fname=',fname
   t0 = time.time()
   fop = h5py.File(fname,'w')
   npt = len(xts)
   nop = 3*npt
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      # 0.5*(Si+*Sj-+Si-*Sj) + Szi*Szj
      if not ifQt:
	 # (a) 0.5*Si+*Sj-
	 if isite == 0:
	    cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,'Sp','Sm',0.5)
         else:
	    cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,'Sp','Sm',1.0)
	 for ipt in range(npt):
	    rop = mpo_dmrg_opers.genExpISyPhi(xts[ipt])
	    wop = mpo_dmrg_opers.prodTwoOpers(cop,rop)
            grp['op'+str(0*npt+ipt)] = wop
	 # (b) 0.5*Si-*Sj+
	 if isite == 0:
	    cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,'Sm','Sp',0.5)
         else:
	    cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,'Sm','Sp',1.0)
	 for ipt in range(npt):
	    rop = mpo_dmrg_opers.genExpISyPhi(xts[ipt])
	    wop = mpo_dmrg_opers.prodTwoOpers(cop,rop)
 	    grp['op'+str(1*npt+ipt)] = wop
	 # (c) Szi*Szj
	 cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,'Sz','Sz',1.0)
	 for ipt in range(npt):
	    rop = mpo_dmrg_opers.genExpISyPhi(xts[ipt])
	    wop = mpo_dmrg_opers.prodTwoOpers(cop,rop)
 	    grp['op'+str(2*npt+ipt)] = wop
      else:
	 # Sip*Sjm
	 for ipt in range(npt):
	    if isite == 0:
	       cop = qtensor_spinopers.genLocal2RSpatialQt(nsite,isite,ig,jg,'Sp','Sm',0.5,xts[ipt])
            else:
	       cop = qtensor_spinopers.genLocal2RSpatialQt(nsite,isite,ig,jg,'Sp','Sm',1.0,xts[ipt])
            cop.dump(grp,'op'+str(0*npt+ipt))
	 # Sim*Sjp
	 for ipt in range(npt):
	    if isite == 0:
	       cop = qtensor_spinopers.genLocal2RSpatialQt(nsite,isite,ig,jg,'Sm','Sp',0.5,xts[ipt])
            else:
	       cop = qtensor_spinopers.genLocal2RSpatialQt(nsite,isite,ig,jg,'Sm','Sp',1.0,xts[ipt])
            cop.dump(grp,'op'+str(1*npt+ipt))
	 # Siz*Sjz
	 for ipt in range(npt):
	    cop = qtensor_spinopers.genLocal2RSpatialQt(nsite,isite,ig,jg,'Sz','Sz',1.0,xts[ipt])
 	    cop.dump(grp,'op'+str(2*npt+ipt))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_SiSjRpt = %.2f s'%(t1-t0)
   return fop
