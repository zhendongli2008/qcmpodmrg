#!/usr/bin/env python
#
# General subroutine for evaluation of <bra|O|ket> given <bra|, |ket>, 
# and O in MPS/MPO form, where O is simple enough such that calculation 
# can be done just on single core => bond dim is small.
# Currently, O = {Epq,Tpq,SA*SB} for MPS and SP-MPS.
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
# 
# def genMPO_Apq(nsite,islst,fname,ifQt,debug=False):
# def genMPO_Apqrs(nsite,islst,plst,fname,ifQt,debug=False):
# def genMPO_Epq(nsite,p,q,fname,ifQt,debug=False):
#
# def genMPO_S2Global(nsite,fname,ifQt,debug=False):
# def genMPO_Global(nsite,key,fname,ifQt,debug=False):
#
# def genMPO_Local(nsite,ig,key,fname,ifQt,debug=False):
# def genMPO_Local2(nsite,ig,jg,ikey,jkey,fac,fname,ifQt,debug=False):
# def genMPO_SiSj(nsite,ig,jg,fname,ifQt,debug=False):
# 
import os
import time
import h5py
import math
import numpy
from qcmpodmrg.source import sysutil_io
from qcmpodmrg.source import mpo_dmrg_opers
from qcmpodmrg.source import mpo_dmrg_spinopers
from qcmpodmrg.source import mpo_dmrg_io
from qcmpodmrg.source.qtensor import qtensor_opers
from qcmpodmrg.source.qtensor import qtensor_spinopers

# Apq = p[is1]^+*q[is2], where islst = [is1,is2] contains spin cases,
# for p,q for all spatial sites, thus dump 'MPO' list of nsite**2.
def genMPO_Apq(nsite,islst,fname,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPO.genMPO_Apq] fname=',fname
   t0 = time.time()
   fop = h5py.File(fname,'w')
   nop = nsite**2
   fop['nop'] = nop
   is1,is2 = islst
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 # Loop over operators
	 idx = 0
	 for p in range(nsite):
	    for q in range(nsite):
	       oplst = [[2*p+is1,1],[2*q+is2,0]]
	       cop = mpo_dmrg_opers.genElemProductSpatial(oplst,isite)
               grp['op'+str(idx)] = cop
	       idx += 1
      else:
	 # Loop over operators
	 idx = 0
	 for p in range(nsite):
	    for q in range(nsite):
	       oplst = [[2*p+is1,1],[2*q+is2,0]]
	       cop = qtensor_opers.genElemProductSpatialQt(oplst,isite)
               cop.dump(grp,'op'+str(idx))
	       idx += 1
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_Apq = %.2f s'%(t1-t0)
   return fop

# p[is1]^+*q[is2]^+*r[is3]*s[is4]
def genMPO_Apqrs(nsite,islst,plst,fname,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPO.genMPO_Apqrs] fname=',fname
   t0 = time.time()
   fop = h5py.File(fname,'w')
   nop = len(plst)*nsite**3
   fop['nop'] = nop
   is1,is2,is3,is4 = islst
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 # Loop over operators
	 idx = 0
	 for p in plst:
	    for q in range(nsite):
	       for r in range(nsite):
	          for s in range(nsite):
	             oplst = [[2*p+is1,1],[2*q+is2,1],[2*r+is3,0],[2*s+is4,0]]
	             cop = mpo_dmrg_opers.genElemProductSpatial(oplst,isite)
                     grp['op'+str(idx)] = cop
	             idx += 1
      else:
	 # Loop over operators
	 idx = 0
	 for p in plst:
	    for q in range(nsite):
	       for r in range(nsite):
	          for s in range(nsite):
	             oplst = [[2*p+is1,1],[2*q+is2,1],[2*r+is3,0],[2*s+is4,0]]
	             cop = qtensor_opers.genElemProductSpatialQt(oplst,isite)
                     cop.dump(grp,'op'+str(idx))
	             idx += 1
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_Apqrs = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] for Epq = pa+*qa + pb+*qb
# Instead of doing the sum directly, we separate them 
# into two component similar to the concept for total H. 
def genMPO_Epq(nsite,p,q,fname,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPO.genMPO_Epq] fname=',fname
   t0 = time.time()
   fop = h5py.File(fname,'w')
   nop = 2
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
         # {pa+*qa,pb+*qb} 
	 for is1 in [0,1]:
	    oplst = [[2*p+is1,1],[2*q+is1,0]]
	    cop = mpo_dmrg_opers.genElemProductSpatial(oplst,isite)
            grp['op'+str(is1)] = cop
      else:
         # {pa+*qa,pb+*qb} 
	 for is1 in [0,1]:
	    oplst = [[2*p+is1,1],[2*q+is1,0]]
	    cop = qtensor_opers.genElemProductSpatialQt(oplst,isite)
            cop.dump(grp,'op'+str(is1))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_Epq = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] 
def genMPO_S2Global(nsite,fname,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPO.genMPO_S2Global] fname=',fname
   t0 = time.time()
   fop = h5py.File(fname,'w')
   nop = 1
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 cop = mpo_dmrg_spinopers.genS2GlobalSpatial(nsite,isite)
         grp['op'+str(0)] = cop
      else:
	 cop = qtensor_spinopers.genS2GlobalSpatialQt(nsite,isite)
         cop.dump(grp,'op'+str(0))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_S2Global = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] 
def genMPO_Global(nsite,key,fname,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPO.genMPO_Global] fname=',fname,' key=',key
   t0 = time.time()
   fop = h5py.File(fname,'w')
   nop = 1
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 cop = mpo_dmrg_spinopers.genGlobalSpatial(nsite,isite,key)
         grp['op'+str(0)] = cop
      else:
	 cop = qtensor_spinopers.genGlobalSpatialQt(nsite,isite,key)
         cop.dump(grp,'op'+str(0))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_Global = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] 
def genMPO_Local(nsite,ig,key,fname,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPO.genMPO_Local] fname=',fname,' ig=',ig,' key=',key
   t0 = time.time()
   fop = h5py.File(fname,'w')
   nop = 1
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 cop = mpo_dmrg_spinopers.genLocalSpatial(nsite,isite,ig,key)
         grp['op'+str(0)] = cop
      else:
	 cop = qtensor_spinopers.genLocalSpatialQt(nsite,isite,ig,key)
         cop.dump(grp,'op'+str(0))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_Local = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] 
def genMPO_Local2(nsite,ig,jg,ikey,jkey,fac,fname,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPO.genMPO_Local2] fname=',fname
   if debug: print ' ig=',ig,' jg=',jg,' ikey/jkey=',(ikey,jkey)
   t0 = time.time()
   fop = h5py.File(fname,'w')
   nop = 1
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      if not ifQt:
	 cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,ikey,jkey,fac)
         grp['op'+str(0)] = cop
      else:
	 cop = qtensor_spinopers.genLocal2SpatialQt(nsite,isite,ig,jg,ikey,jkey,fac)
         cop.dump(grp,'op'+str(0))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_Local2 = %.2f s'%(t1-t0)
   return fop

# DUMP Wx[isite] for SiSj = 0.5*(Si+*Sj-+Si-*Sj) + Szi*Szj
def genMPO_SiSj(nsite,ig,jg,fname,ifQt,debug=False):
   if debug: print '\n[mpo_dmrg_propsMPO.genMPO_SiSj] fname=',fname
   t0 = time.time()
   fop = h5py.File(fname,'w')
   nop = 3
   fop['nop'] = nop
   for isite in range(nsite):
      ti = time.time()
      gname = 'site'+str(isite)
      grp = fop.create_group(gname)
      # vec{S}_i*vec{S}_j = 0.5*(Si+*Sj-+Si-*Sj) + Siz*Sjz 
      # (only need to put the factor on the first site)
      if not ifQt:
	 # Sip*Sjm
	 if isite == 0:
	    cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,'Sp','Sm',0.5)
         else:
	    cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,'Sp','Sm',1.0)
         grp['op'+str(0)] = cop
	 # Sim*Sjp
	 if isite == 0:
	    cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,'Sm','Sp',0.5)
         else:
	    cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,'Sm','Sp',1.0)
         grp['op'+str(1)] = cop
	 # Siz*Sjz
	 cop = mpo_dmrg_spinopers.genLocal2Spatial(nsite,isite,ig,jg,'Sz','Sz',1.0)
 	 grp['op'+str(2)] = cop
      else:
	 # Sip*Sjm
	 if isite == 0:
	    cop = qtensor_spinopers.genLocal2SpatialQt(nsite,isite,ig,jg,'Sp','Sm',0.5)
         else:
	    cop = qtensor_spinopers.genLocal2SpatialQt(nsite,isite,ig,jg,'Sp','Sm',1.0)
         cop.dump(grp,'op'+str(0))
	 # Sim*Sjp
	 if isite == 0:
	    cop = qtensor_spinopers.genLocal2SpatialQt(nsite,isite,ig,jg,'Sm','Sp',0.5)
         else:
	    cop = qtensor_spinopers.genLocal2SpatialQt(nsite,isite,ig,jg,'Sm','Sp',1.0)
         cop.dump(grp,'op'+str(1))
	 # Siz*Sjz
	 cop = qtensor_spinopers.genLocal2SpatialQt(nsite,isite,ig,jg,'Sz','Sz',1.0)
 	 cop.dump(grp,'op'+str(2))
      tf = time.time()
      if debug: print ' isite =',isite,' time = %.2f s'%(tf-ti) 
   t1 = time.time()
   if debug: print ' time for genMPO_SiSj = %.2f s'%(t1-t0)
   return fop
