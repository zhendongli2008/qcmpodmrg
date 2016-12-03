#!/usr/bin/env python
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def HDiagQt(info,ndim0,prjmap):
# def HVecQt(vec0,info,ndim0,prjmap,ifile=0,ioff=0):
# def SDiagQt(info,ndim0,prjmap):
# def SVecQt(vec0,info,ndim0,prjmap):
# def pRDMQt(cimat,info):
# def renormQt_H00(dmrg,isite,ncsite,flst,flstN,site,status):
# def renormQt_S00(dmrg,isite,ncsite,flst,flstN,site,status):
# def renormQt_S0I(dmrg,isite,ncsite,flst,flstN,site,status):
# def renormQt_H0I(dmrg,isite,ncsite,flst,flstN,site,status,iHd=0,thresh=1.e-12):
#
# def PBasSopsQt(info,ndim0,prjmap,nref):
# def PBasPopsQt(info,ndim0,prjmap,nref):
# def BVecQt(info,ndim0,prjmap,iHd,thresh=1.e-12):
# 
import time
import numpy
import mpo_dmrg_io
import mpo_dmrg_dotutil
from qtensor import qtensor
from qtensor import qtensor_opers
from sysutil_include import dmrg_dtype,dmrg_mtype

# Diagonal term of H in Vl*Vr
def HDiagQt(info,ndim0,prjmap):
   dmrg,isite,ncsite,flst,status,ifsym = info
   fL,fR = flst[0][0:2]
   ldim,cdim,rdim,ndim = dmrg.dims
   if dmrg.iprt > 0: print '[mpo_dmrg_kernel.HDiag] (l,c,r,n,n0)=',(ldim,cdim,rdim,ndim,ndim0) 
   # Loop over operators
   diag = numpy.zeros(ndim,dtype=dmrg_dtype)
   lops = qtensor.Qt()
   rops = qtensor.Qt()
   cops = qtensor.Qt()
   wops = qtensor.Qt()
   nop = dmrg.fhop['nops'].value
   if ncsite == 1:

      for iop in range(nop):
         # L/R-OP
	 lops.loadInfo(fL,'opers'+str(iop))
         rops.loadInfo(fR,'opers'+str(iop))
         # COP
         pindx = dmrg.opers[iop]
         cops.load(dmrg.fhop,'site'+str(isite)+'/op'+str(iop))
         # Start to construct the diagonal terms
	 #
	 #   /       |	     \
	 #  |--- i --|-- j ---|
	 #   \	     |	     /
	 #
	 # LOOP OVER islc
	 for islc in range(cops.maxslc[0]):
	    if lops.size[islc] == 0: continue
	    lop = lops.loadSLC(fL,'opers'+str(iop),islc)
      	    ldiag = lop.toDenseDiag(lop.idlst)
	    lop = None
	    # LOOP OVER jslc
	    for jslc in range(cops.maxslc[1]):
	       ijdx = cops.ravel([islc,jslc])
	       if cops.size[ijdx] == 0 or rops.size[jslc] == 0: continue
	       cop = cops.dic[ijdx]
      	       cdiag = cop.toDenseDiag(cop.idlst)
	       rop = lops.loadSLC(fR,'opers'+str(iop),jslc)
      	       rdiag = rop.toDenseDiag(rop.idlst)
	       rop = None
      	       # contract bonds: l[p,dl]*c[p,q,dc]*r[q,dr]
      	       tmp = numpy.tensordot(ldiag,cdiag,axes=([0],[0])) # lc[dl,q,dc]
      	       tmp = numpy.tensordot(tmp,rdiag,axes=([1],[0]))   # lcr[dl,dc,dr]
      	       diag += dmrg.hpwts[iop]*tmp.reshape(ndim)

   elif ncsite == 2:

      jsite = isite+1
      for iop in range(nop):
         # L/R-OP
	 lops.loadInfo(fL,'opers'+str(iop))
         rops.loadInfo(fR,'opers'+str(iop))
         # COP
         pindx = dmrg.opers[iop]
         cops.load(dmrg.fhop,'site'+str(isite)+'/op'+str(iop))
         wops.load(dmrg.fhop,'site'+str(jsite)+'/op'+str(iop))
         # Start to construct the diagonal terms
	 #
	 #   /       |	     |       \
	 #  |--- i --|-- j --|-- k ---|
	 #   \	     |	     |       /
	 #
	 # LOOP OVER islc
	 for islc in range(cops.maxslc[0]):
	    if lops.size[islc] == 0: continue
	    lop = lops.loadSLC(fL,'opers'+str(iop),islc)
      	    ldiag = lop.toDenseDiag(lop.idlst)
	    lop = None
	    # LOOP OVER kslc
	    for kslc in range(wops.maxslc[1]):
	       if rops.size[kslc] == 0: continue
	       rop = lops.loadSLC(fR,'opers'+str(iop),kslc)
      	       rdiag = rop.toDenseDiag(rop.idlst)
	       rop = None
	       # LOOP OVER jslc
	       for jslc in range(cops.maxslc[1]):
	          ijdx = cops.ravel([islc,jslc])
	          jkdx = wops.ravel([jslc,kslc])
	          if cops.size[ijdx] == 0 or wops.size[jkdx] == 0: continue
	          cop = cops.dic[ijdx]
      	       	  cdiag = cop.toDenseDiag(cop.idlst)
	          wop = wops.dic[jkdx]
         	  wdiag = wop.toDenseDiag(wop.idlst)
      	          # contract bonds: l[p,dl]*c[p,q,dc]*w[q,s,dw]*r[s,dr]
      	       	  tmp = numpy.tensordot(ldiag,cdiag,axes=([0],[0])) # lc[dl,q,dc]
	       	  tmp = numpy.tensordot(tmp,wdiag,axes=([1],[0]))   # lcw[dl,dc,s,dw]
		  tmp = numpy.tensordot(tmp,rdiag,axes=([2],[0]))   # lcwr[dl,dc,dw,dr]
      	       	  diag += dmrg.hpwts[iop]*tmp.reshape(ndim)

   # Reshaping
   diag = diag[prjmap].real.copy()

   # TEST
   debug = False 
   if debug and ifsym: checkDiagQt(diag,HVecQt,info,ndim0,prjmap)
   return diag

def checkDiagQt(diag,funQt,vec,info,ndim0,prjmap):
   print ' checkDiagQt: debuging diag and funQt ...'
   dmrg = info[0]
   assert dmrg.comm.size == 1
   Hmat = numpy.zeros((ndim0,ndim0),dtype=dmrg_dtype)
   for i in range(ndim0):
      vec = numpy.zeros(ndim0,dtype=dmrg_dtype)
      vec[i] = 1.0
      Hmat[:,i] = funQt(vec,info,ndim0,prjmap)
      print ' idx=',i,'diag[i]=',diag[i],'Hii=',Hmat[i,i],'diff=',Hmat[i,i]-diag[i]
   diff1 = numpy.linalg.norm(Hmat-Hmat.T.conj())
   diff2 = numpy.linalg.norm(numpy.diag(Hmat)-diag)
   if dmrg.comm.size == 1:
      print ' diff1(H-H.T.conj)=',diff1 # diff1= 8.21020104329e-08
      if diff1 > 1.e-10:
         print ' error: Fmat is not Hermitian!'
         exit(1)
   print ' diff2(H.diag-D)=',diff2 # diff2= 1.47061633021e-14
   if diff2 > 1.e-10:
      print ' error: diag and Fmat are not consistent!'
      exit(1)
   return 0

# H*Vec
def HVecQt(vec0,info,ndim0,prjmap,ifile=0,ioff=0):
   dmrg,isite,ncsite,flst,status,ifsym = info
   fL,fR = flst[ifile][ioff:ioff+2]
   dmrg.qtmp.value = vec0.copy()
   if not dmrg.ifs2proj:
      qvec = dmrg.qtmp 
   else:
      qvec = dmrg.qtmp.reduceQsymsToN()
   # Loop over operators
   hvec = numpy.zeros(ndim0,dtype=dmrg_dtype)
   lops = qtensor.Qt()
   rops = qtensor.Qt()
   cops = qtensor.Qt()
   # cases for symmetry template depending on ap^+Vp
   dims1 = numpy.array([8]+[dmrg.maxslc]*1)
   dims2 = numpy.array([8]+[dmrg.maxslc]*2)
   ncase1 = numpy.prod(dims1)
   ncase2 = numpy.prod(dims2)
   ista1 = dict([(i,-1) for i in range(ncase1)])
   ista2 = dict([(i,-1) for i in range(ncase2)])
   top1 = dict([(i,qtensor.qtensor()) for i in range(ncase1)])
   top2 = dict([(i,qtensor.qtensor()) for i in range(ncase2)])
   nop = dmrg.fhop['nops'].value
   if ncsite == 1:

      # Initializaiton
      top3 = dict([(i,qtensor.qtensor()) for i in range(ncase2)])
      # Loop over operators
      for iop in range(nop):
         # L/R-OP
	 lops.loadInfo(fL,'opers'+str(iop))
         rops.loadInfo(fR,'opers'+str(iop))
         # WOP
	 pindx = dmrg.opers[iop]
         cops.load(dmrg.fhop,'site'+str(isite)+'/op'+str(iop))
	 # Symmetry & Buffer
	 icase0 = qtensor_opers.case(pindx[0],isite)
	 #
	 #   /       |	     \
	 #  |--- i --|-- j ---|
	 #   \	     |	     /
	 #
	 # LOOP OVER islc
	 for islc in range(cops.maxslc[0]):
	    if lops.size[islc] == 0: continue
	    lop = lops.loadSLC(fL,'opers'+str(iop),islc)
	    icase1 = numpy.ravel_multi_index([icase0,islc],dims1)
	    if ista1[icase1] == -1:
	       ilen1 = top1[icase1].tensordotSYM(lop,qvec,axes=([2],[0]))
	       ista1[icase1] = ilen1
	    if ista1[icase1] <= 0: continue
 	    top1[icase1].tensordotCAL(lop,qvec)
	    lop = None

	    # LOOP OVER jslc
	    for jslc in range(cops.maxslc[1]):
	       ijdx = cops.ravel([islc,jslc])
	       if cops.size[ijdx] == 0 or rops.size[jslc] == 0: continue
	       cop = cops.dic[ijdx]
	       rop = lops.loadSLC(fR,'opers'+str(iop),jslc)
	       icase2 = numpy.ravel_multi_index([icase0,islc,jslc],dims2)
	       if ista2[icase2] == -1:
                  top2[icase2].tensordotSYM(top1[icase1],cop,axes=([0,2],[0,3]))
                  # Flip symmetry
                  if dmrg.ifs2proj: 
                     rop.qsyms = [-rop.qsyms[0],dmrg.qkey[:1]-rop.qsyms[1],dmrg.qkey[:1]-rop.qsyms[2]] 
                  else:
                     rop.qsyms = [-rop.qsyms[0],dmrg.qkey-rop.qsyms[1],dmrg.qkey-rop.qsyms[2]] 
	    	  ilen2 = top3[icase2].tensordotSYM(top2[icase2],rop,axes=([1,2],[2,0]))
	          ista2[icase2] = ilen2
	       if ista2[icase2] <= 0: continue
               top2[icase2].tensordotCAL(top1[icase1],cop)
	       top3[icase2].tensordotCAL(top2[icase2],rop)
	       rop = None
	       top2[icase2].value = None # jslc
	       if dmrg.ifs2proj:
	          # Assuming projectionNMs does not take too much time!
                  tmp = top3[icase2].projectionNMs(dmrg.qtmp.qsyms)
                  hvec += dmrg.hpwts[iop]*tmp.value
	          tmp = None
               else:
                  hvec += dmrg.hpwts[iop]*top3[icase2].value
	       top3[icase2].value = None # jslc
	    top1[icase1].value = None # islc

   elif ncsite == 2:

      # Initialization
      wops = qtensor.Qt()
      dims3 = numpy.array([8]+[dmrg.maxslc]*3)
      ncase3 = numpy.prod(dims3)
      ista3 = dict([(i,-1) for i in range(ncase3)])
      top3 = dict([(i,qtensor.qtensor()) for i in range(ncase3)])
      top4 = dict([(i,qtensor.qtensor()) for i in range(ncase3)])
      jsite = isite+1
      # Loop over operators
      for iop in range(nop):
         # L/R-OP
         lops.loadInfo(fL,'opers'+str(iop))
         rops.loadInfo(fR,'opers'+str(iop))
         # WOP
         pindx = dmrg.opers[iop]
         cops.load(dmrg.fhop,'site'+str(isite)+'/op'+str(iop))
         wops.load(dmrg.fhop,'site'+str(jsite)+'/op'+str(iop))
	 # Symmetry & Buffer
	 icase0 = qtensor_opers.case(pindx[0],isite,ncsite)
	 #
	 #   /       |	     |       \
	 #  |--- i --|-- j --|-- k ---|
	 #   \	     |	     |       /
	 #
	 # LOOP OVER islc
	 for islc in range(cops.maxslc[0]):
	    if lops.size[islc] == 0: continue
	    lop = lops.loadSLC(fL,'opers'+str(iop),islc)
	    icase1 = numpy.ravel_multi_index([icase0,islc],dims1)
	    if ista1[icase1] == -1:
	       ilen1 = top1[icase1].tensordotSYM(lop,qvec,axes=([2],[0]))
	       ista1[icase1] = ilen1
	    if ista1[icase1] <= 0: continue
 	    top1[icase1].tensordotCAL(lop,qvec)
	    lop = None

	    # LOOP OVER jslc
	    for jslc in range(cops.maxslc[1]):
	       ijdx = cops.ravel([islc,jslc])
	       if cops.size[ijdx] == 0: continue
	       cop = cops.dic[ijdx]
	       icase2 = numpy.ravel_multi_index([icase0,islc,jslc],dims2)
	       if ista2[icase2] == -1:
                  ilen2 = top2[icase2].tensordotSYM(top1[icase1],cop,axes=([0,2],[0,3]))
	          ista2[icase2] = ilen2
	       if ista2[icase2] <= 0: continue
               top2[icase2].tensordotCAL(top1[icase1],cop)
	    
	       # LOOP OVER kslc
	       for kslc in range(wops.maxslc[1]):
	          jkdx = wops.ravel([jslc,kslc])
	          if wops.size[jkdx] == 0 or rops.size[kslc] == 0: continue
	          wop = wops.dic[jkdx]
	    	  rop = lops.loadSLC(fR,'opers'+str(iop),kslc)
	          icase3 = numpy.ravel_multi_index([icase0,islc,jslc,kslc],dims3)
	          if ista3[icase3] == -1:
	             # Flip symmetry (only once)
                     if dmrg.ifs2proj: 
                        rop.qsyms = [-rop.qsyms[0],dmrg.qkey[:1]-rop.qsyms[1],dmrg.qkey[:1]-rop.qsyms[2]] 
                     else:
                        rop.qsyms = [-rop.qsyms[0],dmrg.qkey-rop.qsyms[1],dmrg.qkey-rop.qsyms[2]] 
	    	     top3[icase3].tensordotSYM(top2[icase2],wop,axes=([1,3],[3,0]))
            	     ilen3 = top4[icase3].tensordotSYM(top3[icase3],rop,axes=([1,3],[2,0]))
	             ista3[icase3] = ilen3
	          if ista3[icase3] <= 0: continue
	 	  top3[icase3].tensordotCAL(top2[icase2],wop)
         	  top4[icase3].tensordotCAL(top3[icase3],rop)
		  rop = None
	 	  top3[icase3].value = None # kslc
         	  if dmrg.ifs2proj: 
         	     tmp = top4[icase3].projectionNMs(dmrg.qtmp.qsyms)
	 	     hvec += dmrg.hpwts[iop]*tmp.value
	 	     tmp = None
         	  else:
	 	     hvec += dmrg.hpwts[iop]*top4[icase3].value
	 	  top4[icase3].value = None # kslc
	       top2[icase2].value = None # jslc
	    top1[icase1].value = None # islc

   return hvec

# Diagonal term of H in Vl*Vr
def SDiagQt(info,ndim0,prjmap):
   dmrg,isite,ncsite,flst,status,ifsym = info
   fL,fR,fLp,fRp = flst[0][0:4]
   ldim,cdim,rdim,ndim = dmrg.dims
   diag = numpy.zeros(ndim,dtype=dmrg_dtype)
   # Only rank-0 deal with <k|s2proj|k>
   if dmrg.comm.rank == 0 and dmrg.ifs2proj:
      lop = qtensor.qtensor()
      rop = qtensor.qtensor()
      cop = qtensor.qtensor()
      wop = qtensor.qtensor()
      nop = dmrg.npts
      if ncsite == 1:
         # Loop over operators
         for iop in range(nop):
            cop.load(dmrg.fpop,'op'+str(iop))
            lop.load(fLp,'opers'+str(iop))
            rop.load(fRp,'opers'+str(iop))
            # Start to construct the diagonal terms
            cdiag = cop.toDenseDiag(cop.idlst)
            ldiag = lop.toDenseDiag(lop.idlst)
            rdiag = rop.toDenseDiag(rop.idlst)
            # contract bonds: l[p,dl]*c[p,q,dc]*r[q,dr]
            tmp = numpy.tensordot(ldiag,cdiag,axes=([0],[0])) # lc[dl,q,dc]
            tmp = numpy.tensordot(tmp,rdiag,axes=([1],[0]))   # lcr[dl,dc,dr]
            diag += dmrg.qwts[iop]*tmp.reshape(ndim)
      elif ncsite == 2:
         jsite = isite+1
         for iop in range(nop):
            cop.load(dmrg.fpop,'op'+str(iop))
            wop.load(dmrg.fpop,'op'+str(iop))
            lop.load(fLp,'opers'+str(iop))
            rop.load(fRp,'opers'+str(iop))
            # Start to construct the diagonal terms
            cdiag = cop.toDenseDiag(cop.idlst)
            wdiag = cop.toDenseDiag(wop.idlst)
            ldiag = lop.toDenseDiag(lop.idlst)
            rdiag = rop.toDenseDiag(rop.idlst)
            # contract bonds: l[p,dl]*c[p,q,dc]*w[q,s,dw]*r[s,dr]
            tmp = numpy.tensordot(ldiag,cdiag,axes=([0],[0])) # lc[dl,q,dc]
            tmp = numpy.tensordot(tmp,wdiag,axes=([1],[0]))   # lcw[dl,dc,s,dw]
            tmp = numpy.tensordot(tmp,rdiag,axes=([2],[0]))   # lcwr[dl,dc,dw,dr]
            diag += dmrg.qwts[iop]*tmp.reshape(ndim)
   # Reshaping
   diag = diag[prjmap].real.copy()
   # TEST
   debug = False 
   if debug and ifsym: checkDiagQt(diag,SVecQt,info,ndim0,prjmap)
   return diag

# S*Vec
def SVecQt(vec0,info,ndim0,prjmap):
   dmrg,isite,ncsite,flst,status,ifsym = info
   fL,fR,fLp,fRp = flst[0][0:4]
   svec = numpy.zeros(ndim0,dtype=dmrg_dtype)
   if dmrg.comm.rank == 0 and dmrg.ifs2proj:
      dmrg.qtmp.value = vec0.copy()
      if not dmrg.ifs2proj:
         qvec = dmrg.qtmp 
      else:
         qvec = dmrg.qtmp.reduceQsymsToN()
      nop = dmrg.npts
      lop = qtensor.qtensor()
      rop = qtensor.qtensor()
      cop = qtensor.qtensor()
      # cases for symmetry template depending on ap^+Vp
      ncase = 1 
      ista = dict([(i,0) for i in range(ncase)])
      top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
      top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
      top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
      top4 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
      icase = 0
      if ncsite == 1:
      
         # Loop over operators
         for iop in range(nop):
            cop.load(dmrg.fpop,'op'+str(iop))
            # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
            lop.load(fLp,'opers'+str(iop))
            rop.load(fRp,'opers'+str(iop))
            # Flip symmetry
            rop.qsyms = [-rop.qsyms[0],dmrg.qkey[:1]-rop.qsyms[1],dmrg.qkey[:1]-rop.qsyms[2]] 
	    # Symmetry & Buffer
	    if ista[icase] == 0:
	       top1[icase].tensordotSYM(lop,qvec,axes=([2],[0]))
	       top2[icase].tensordotSYM(top1[icase],cop,axes=([0,2],[0,3]))
	       top3[icase].tensordotSYM(top2[icase],rop,axes=([1,2],[2,0]))
	       ista[icase] = 1
	    top1[icase].tensordotCAL(lop,qvec)
	    top2[icase].tensordotCAL(top1[icase],cop)
	    top1[icase].value = None
	    top3[icase].tensordotCAL(top2[icase],rop)
	    top2[icase].value = None
            tmp = top3[icase].projectionNMs(dmrg.qtmp.qsyms)
	    top3[icase].value = None
            svec += dmrg.qwts[iop]*tmp.value
	    tmp = None

      elif ncsite == 2:

         # Loop over operators
         for iop in range(nop):
            cop.load(dmrg.fpop,'op'+str(iop))
            # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
            lop.load(fLp,'opers'+str(iop))
            rop.load(fRp,'opers'+str(iop))
            # Flip symmetry
            rop.qsyms = [-rop.qsyms[0],dmrg.qkey[:1]-rop.qsyms[1],dmrg.qkey[:1]-rop.qsyms[2]]
	    # Symmetry & Buffer
	    if ista[icase] == 0:
	       top1[icase].tensordotSYM(lop,qvec,axes=([2],[0]))
	       top2[icase].tensordotSYM(top1[icase],cop,axes=([0,2],[0,3]))
	       top3[icase].tensordotSYM(top2[icase],cop,axes=([1,3],[3,0]))
	       top4[icase].tensordotSYM(top3[icase],rop,axes=([1,3],[2,0]))
	       ista[icase] = 1
	    top1[icase].tensordotCAL(lop,qvec)
	    top2[icase].tensordotCAL(top1[icase],cop)
	    top1[icase].value = None
	    top3[icase].tensordotCAL(top2[icase],cop)
	    top2[icase].value = None
	    top4[icase].tensordotCAL(top3[icase],rop)
	    top3[icase].value = None
            tmp = top4[icase].projectionNMs(dmrg.qtmp.qsyms)
	    top4[icase].value = None
            svec += dmrg.qwts[iop]*tmp.value
	    tmp = None

   return svec

# pRDM
def pRDMQt(cimat,info):
   dmrg,isite,ncsite,flst,status,ifsym = info
   fL,fR = flst[0][0:2]
   ldim,cdim,rdim,ndim = dmrg.dims
   if dmrg.iprt > 0: print '[mpo_dmrg_kernel.pRDM] (l,c,r,n)=',(ldim,cdim,rdim,ndim)
   neig,diml,dimc,dimr = cimat.shape
   nop = dmrg.nops
   dims = numpy.array([6]+[dmrg.maxslc]*2)
   ncase = numpy.prod(dims)
   if status == 'L':

      rdm1 = numpy.zeros((diml,dimc,diml,dimc),dtype=dmrg_dtype)
      lops = qtensor.Qt()
      cops = qtensor.Qt()
      # LOOP OVER QSECTORS
      idx = 0
      for qkey in dmrg.qsectors:
         neig = dmrg.qsectors[qkey]
         key = mpo_dmrg_dotutil.floatKey(qkey)
         mpo_dmrg_dotutil.setupQtmpPRDM(dmrg,ncsite,key,'L')            
         # cases for symmetry template depending on ap^+Vp
         ista = dict([(i,-1) for i in range(ncase)])
         top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         # LOOP OVER STATES
         for ieig in range(neig):	
            dmrg.qtmp.fromDense(cimat[idx],dmrg.idlstCI)
            idx += 1

            # Left: C[l',r]*C[l,r]=P[l',l]
            if dmrg.comm.rank == 0:
               qt = qtensor.tensordot(dmrg.qtmp,dmrg.qtmp,axes=([2],[2]),ifc2=True) 
               rdm1 += qt.toDenseTensor(dmrg.idlstPRDM)
	    
	    # Add noise: Hl*|L><L|*Hl
	    if dmrg.inoise == 0 and dmrg.noise >= 1.e-10:
               if not dmrg.ifs2proj:
                  qvec = dmrg.qtmp
               else:
      	          qvec = dmrg.qtmp.reduceQsymsToN()
               # LOOP OVER OPERTORS
               icsPRDM = None
               tmpPRDM = None
               for iop in range(nop):
                  # LOP
      	          lops.loadInfo(fL,'opers'+str(iop))
                  # COP
                  pindx = dmrg.opers[iop]
                  cops.load(dmrg.fhop,'site'+str(isite)+'/op'+str(iop))
                  # Symmetry & Buffer
                  icase0 = qtensor_opers.case(pindx[0],isite)
                  # COMPUTE
                  for jslc in range(cops.maxslc[1]):
                     for islc in range(cops.maxslc[0]):
                        ijdx = cops.ravel([islc,jslc])
                        if cops.size[ijdx] == 0 or lops.size[islc] == 0: continue
                        cop = cops.dic[ijdx]
                        lop = lops.loadSLC(fL,'opers'+str(iop),islc)
                        # Symmetry & Buffer
                        icase = numpy.ravel_multi_index([icase0,jslc,islc],dims)
                        if ista[icase] == -1:
                           ilen1 = top1[icase].tensordotSYM(lop,qvec,axes=([2],[0]))
                           ilen2 = top2[icase].tensordotSYM(top1[icase],cop,axes=([0,2],[0,3]))
                           ilen3 = top3[icase].tensordotSYM(top2[icase],top2[icase],axes=([1,2],[1,2]))
                           ista[icase] = ilen3
                        if ista[icase] <= 0: continue
                        top1[icase].tensordotCAL(lop,qvec)
                        lop = None
                        top2[icase].tensordotCAL(top1[icase],cop)
                        top1[icase].value = None
                        top3[icase].tensordotCAL(top2[icase],top2[icase],ifc2=True)
                        top2[icase].value = None
                        if tmpPRDM is None:
                           icsPRDM = icase
                   	   tmpPRDM = top3[icase].value.copy()
                        else:
                           tmpPRDM += top3[icase].value
                        top3[icase].value = None
               # Finish operator [iop] loops
               top3[icsPRDM].value = tmpPRDM.copy()
               tmpPRDM = None
               rdm1 += dmrg.noise*top3[icsPRDM].toDenseTensor(dmrg.idlstPRDM)
               top3[icsPRDM].value = None
      # 5000*5000*4*4*8/1024^3 ~ 3G
      rdm1 = rdm1.reshape(diml*dimc,diml*dimc)

   elif status == 'R':

      rdm1 = numpy.zeros((dimc,dimr,dimc,dimr),dtype=dmrg_dtype)
      jsite = isite+ncsite-1
      # Loop over operators
      rops = qtensor.Qt()
      cops = qtensor.Qt()
      # LOOP OVER QSECTORS
      idx = 0
      for qkey in dmrg.qsectors:
         neig = dmrg.qsectors[qkey]
         key = mpo_dmrg_dotutil.floatKey(qkey)
         mpo_dmrg_dotutil.setupQtmpPRDM(dmrg,ncsite,key,'R')
         # cases for symmetry template depending on ap^+Vp
         ista = dict([(i,-1) for i in range(ncase)])
         top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         # LOOP OVER STATES
         for ieig in range(neig):	 
            dmrg.qtmp.fromDense(cimat[idx],dmrg.idlstCI)
            idx += 1

            # Right: C[l,r']*C[l,r]=P[r',r]
            if dmrg.comm.rank == 0:
               qt = qtensor.tensordot(dmrg.qtmp,dmrg.qtmp,axes=([0],[0]),ifc1=True) 
               rdm1 += qt.toDenseTensor(dmrg.idlstPRDM)
            
	    if dmrg.inoise == 0 and dmrg.noise >= 1.e-10:
               if not dmrg.ifs2proj:
                  qvec = dmrg.qtmp
               else:
      	          qvec = dmrg.qtmp.reduceQsymsToN()
               # LOOP OVER OPERTORS
               icsPRDM = None
               tmpPRDM = None
               for iop in range(nop):
                  # ROP
                  rops.loadInfo(fR,'opers'+str(iop))
                  # COP
                  pindx = dmrg.opers[iop]
      	          cops.load(dmrg.fhop,'site'+str(jsite)+'/op'+str(iop))
                  # Symmetry & Buffer
                  icase0 = qtensor_opers.case(pindx[0],jsite)
                  # COMPUTE
                  for jslc in range(cops.maxslc[0]):
                     for islc in range(cops.maxslc[1]):
                        ijdx = cops.ravel([jslc,islc])
                        if cops.size[ijdx] == 0 or rops.size[islc] == 0: continue
                        cop = cops.dic[ijdx]
                        rop = rops.loadSLC(fR,'opers'+str(iop),islc)
                        # Symmetry & Buffer
                        icase = numpy.ravel_multi_index([icase0,jslc,islc],dims)
                  	if ista[icase] == -1:
                           # Flip symmetry
                           if dmrg.ifs2proj: 
                              rop.qsyms = [-rop.qsyms[0],dmrg.qkey[:1]-rop.qsyms[1],dmrg.qkey[:1]-rop.qsyms[2]] 
                           else:
                              rop.qsyms = [-rop.qsyms[0],dmrg.qkey-rop.qsyms[1],dmrg.qkey-rop.qsyms[2]] 
                           rop.status[1] = ~rop.status[1]
                           rop.qsyms[1]  = -rop.qsyms[1]
                           ilen1 = top1[icase].tensordotSYM(qvec,rop,axes=([2],[2]))
      	                   ilen2 = top2[icase].tensordotSYM(cop,top1[icase],axes=([1,3],[2,1]))
                           ilen3 = top3[icase].tensordotSYM(top2[icase],top2[icase],axes=([0,2],[0,2])) # T[m'j',mj]
                           ista[icase] = ilen3
                        if ista[icase] <= 0: continue
                        top1[icase].tensordotCAL(qvec,rop)
                        rop = None
      	                top2[icase].tensordotCAL(cop,top1[icase])
                        top1[icase].value = None
                        top3[icase].tensordotCAL(top2[icase],top2[icase],ifc1=True)
                        top2[icase].value = None
                        if tmpPRDM is None:
                           icsPRDM = icase
                   	   tmpPRDM = top3[icase].value.copy()
                        else:
                           tmpPRDM += top3[icase].value
                        top3[icase].value = None
               # Finish operator [iop] loops
               top3[icsPRDM].value = tmpPRDM.copy()
               tmpPRDM = None
               rdm1 += dmrg.noise*top3[icsPRDM].toDenseTensor(dmrg.idlstPRDM)
               top3[icsPRDM].value = None
      # 5000*5000*4*4*8/1024^3 ~ 3G
      rdm1 = rdm1.reshape(dimc*dimr,dimc*dimr)

   return rdm1

# Renormalizaiton of operators: local operations without communication!
# The implementation is very much the same as initQt, except the bra and
# ket is the same here, isite is given, and fL,fR,fN are used.
#
# Construct renormalized operators for <0|H*R(theta)|0> or <0|H0|0> 
def renormQt_H00(dmrg,isite,ncsite,flst,flstN,site,status):
   # Loop over operators
   if not dmrg.ifs2proj:
      qsite = site
   else:
      qsite = site.reduceQsymsToN()
   dims = numpy.array([6]+[dmrg.maxslc]*2)
   ncase = numpy.prod(dims)
   nop = dmrg.fhop['nops'].value
   if status == 'L':

      # Hamiltoinan operators
      fL = flst[0][0] 
      fN = flstN[0][0]
      # Loop over operators
      lops = qtensor.Qt()
      cops = qtensor.Qt()
      # cases for symmetry template depending on ap^+Vp
      ista = dict([(i,-1) for i in range(ncase)])
      top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
      top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
      top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
      for iop in range(nop):
         # LOP
	 lops.loadInfo(fL,'opers'+str(iop))
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
	       lop = lops.loadSLC(fL,'opers'+str(iop),islc)
	       # Symmetry & Buffer
	       icase = numpy.ravel_multi_index([icase0,jslc,islc],dims)
	       if ista[icase] == -1:
	          ilen1 = top1[icase].tensordotSYM(qsite,lop,axes=([0],[1]))
	          ilen2 = top2[icase].tensordotSYM(cop,top1[icase],axes=([0,2],[2,0]))
	          ilen3 = top3[icase].tensordotSYM(top2[icase],qsite,axes=([1,3],[1,0]))
	          ista[icase] = ilen3
	       if ista[icase] <= 0: continue
	       top1[icase].tensordotCAL(qsite,lop,ifc1=True)
	       lop = None
	       top2[icase].tensordotCAL(cop,top1[icase])
	       top1[icase].value = None
	       top3[icase].tensordotCAL(top2[icase],qsite)
	       top2[icase].value = None
	       if qops.size[jslc] == 0:
	          qops.dic[jslc].copy(top3[icase])
	          qops.size[jslc] = qops.dic[jslc].size_allowed
	          # We dump the idlst information for operators & MPS
	          qops.dic[jslc].idlst = [cop.idlst[1],qsite.idlst[2],qsite.idlst[2]]
	       else:
	          qops.dic[jslc].value += top3[icase].value  
	       top3[icase].value = None
	       lop = None
	    if qops.dic[jslc].size > 0: qops.dumpSLC(fN,'opers'+str(iop),jslc)
	    qops.dic[jslc] = None
	 # DUMP information
	 qops.dumpInfo(fN,'opers'+str(iop))

   elif status == 'R':

      jsite = isite+ncsite-1
      # Hamiltoinan operators
      fR = flst[0][1]
      fN = flstN[0][0]
      # Loop over operators
      rops = qtensor.Qt()
      cops = qtensor.Qt()
      # cases for symmetry template depending on ap^+Vp
      ista = dict([(i,-1) for i in range(ncase)])
      top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
      top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
      top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
      for iop in range(nop):
	 # ROP
	 rops.loadInfo(fR,'opers'+str(iop))
         # COP
         pindx = dmrg.opers[iop]
         cops.load(dmrg.fhop,'site'+str(jsite)+'/op'+str(iop))
	 # Symmetry & Buffer
	 icase0 = qtensor_opers.case(pindx[0],jsite)
	 # COMPUTE
	 qops = qtensor.Qt(1,cops.maxslc[0])
	 for jslc in range(cops.maxslc[0]):
	    for islc in range(cops.maxslc[1]):
	       ijdx = cops.ravel([jslc,islc])
	       if cops.size[ijdx] == 0 or rops.size[islc] == 0: continue
	       cop = cops.dic[ijdx]
	       rop = rops.loadSLC(fR,'opers'+str(iop),islc)
	       # Symmetry & Buffer
	       icase = numpy.ravel_multi_index([icase0,jslc,islc],dims)
	       if ista[icase] == -1:
	  	  # Flip horizontal directions
	          cop.qsyms[0]  = -cop.qsyms[0]
	          cop.qsyms[1]  = -cop.qsyms[1]
	          cop.status[0] = ~cop.status[0] 
	          cop.status[1] = ~cop.status[1] 
	          ilen1 = top1[icase].tensordotSYM(qsite,rop,axes=([2],[1]))
	          ilen2 = top2[icase].tensordotSYM(cop,top1[icase],axes=([1,2],[2,1]))
	          ilen3 = top3[icase].tensordotSYM(top2[icase],qsite,axes=([1,3],[1,2]))
	          ista[icase] = ilen3
	       if ista[icase] <= 0: continue
	       #
	       # Originally, the memory usage can be large, since top=O(KD^2) per each H[x]. 
	       # 535 1517.465 MiB  801.098 MiB top1[icase].tensordotCAL(qsite,top)
	       # 536 2423.266 MiB  905.801 MiB top2[icase].tensordotCAL(cop,top1[icase])
	       # Now the bond is sliced, such that the rop is smaller by 1/maxslc.
	       #
	       top1[icase].tensordotCAL(qsite,rop,ifc1=True)
	       rop = None
	       top2[icase].tensordotCAL(cop,top1[icase])
	       top1[icase].value = None
	       top3[icase].tensordotCAL(top2[icase],qsite)
	       top2[icase].value = None
	       if qops.size[jslc] == 0:
	          qops.dic[jslc].copy(top3[icase])
	          qops.size[jslc] = qops.dic[jslc].size_allowed
	          # We dump the idlst information for operators & MPS
	          qops.dic[jslc].idlst = [cop.idlst[0],qsite.idlst[0],qsite.idlst[0]]
	       else:
	          qops.dic[jslc].value += top3[icase].value  
	       top3[icase].value = None
	    if qops.dic[jslc].size > 0: qops.dumpSLC(fN,'opers'+str(iop),jslc)
	    qops.dic[jslc] = None
	 # DUMP information
	 qops.dumpInfo(fN,'opers'+str(iop))
      
   return 0

# Construct renormalized operators for <0|R(theta)|0> (SP-MPS)
def renormQt_S00(dmrg,isite,ncsite,flst,flstN,site,status):
   # Loop over operators
   if not dmrg.ifs2proj:
      qsite = site
   else:
      qsite = site.reduceQsymsToN()
   dims = numpy.array([6]+[dmrg.maxslc]*2)
   ncase = numpy.prod(dims)
   if status == 'L':

      # Deal with the projector	on rank-0 
      if dmrg.comm.rank == 0 and dmrg.ifs2proj:
	 fLp = flst[0][2]
	 fNp = flstN[0][1]
         nop = dmrg.npts
   	 top = qtensor.qtensor()
   	 cop = qtensor.qtensor()
         # cases for symmetry template depending on ap^+Vp
         ncase = 1 
         ista = dict([(i,0) for i in range(ncase)])
         top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         icase = 0
         for iop in range(nop):
	    cop.load(dmrg.fpop,'op'+str(iop))
            top.load(fLp,'opers'+str(iop))
	    # Symmetry & Buffer
	    if ista[icase] == 0:
	       top1[icase].tensordotSYM(qsite,top,axes=([0],[1]))
	       top2[icase].tensordotSYM(cop,top1[icase],axes=([0,2],[2,0]))
	       top3[icase].tensordotSYM(top2[icase],qsite,axes=([1,3],[1,0]))
	       # We dump the idlst information for operators & MPS
	       top3[icase].idlst = [cop.idlst[1],qsite.idlst[2],qsite.idlst[2]]
	       ista[icase] = 1
	    top1[icase].tensordotCAL(qsite,top,ifc1=True)
	    top2[icase].tensordotCAL(cop,top1[icase])
	    top1[icase].value = None
	    top3[icase].tensordotCAL(top2[icase],qsite)
	    top2[icase].value = None
	    top3[icase].dump(fNp,'opers'+str(iop))
	    top3[icase].value = None

   elif status == 'R':

      jsite = isite+ncsite-1
      # Deal with the projector	on rank-0 
      if dmrg.comm.rank == 0 and dmrg.ifs2proj:
	 fRp = flst[0][3]
	 fNp = flstN[0][1]
         nop = dmrg.npts
   	 top = qtensor.qtensor()
   	 cop = qtensor.qtensor()
         # cases for symmetry template depending on ap^+Vp
         ncase = 1 
         ista = dict([(i,0) for i in range(ncase)])
         top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         icase = 0
         for iop in range(nop):
	    cop.load(dmrg.fpop,'op'+str(iop))
	    # Flip horizontal directions
	    cop.qsyms[0]  = -cop.qsyms[0]
	    cop.qsyms[1]  = -cop.qsyms[1]
	    cop.status[0] = ~cop.status[0] 
	    cop.status[1] = ~cop.status[1] 
            top.load(fRp,'opers'+str(iop))
	    # Symmetry & Buffer
	    if ista[icase] == 0:
	       top1[icase].tensordotSYM(qsite,top,axes=([2],[1]))
	       top2[icase].tensordotSYM(cop,top1[icase],axes=([1,2],[2,1]))
	       top3[icase].tensordotSYM(top2[icase],qsite,axes=([1,3],[1,2]))
	       # We dump the idlst information for operators & MPS
	       top3[icase].idlst = [cop.idlst[0],qsite.idlst[0],qsite.idlst[0]]
	       ista[icase] = 1
	    top1[icase].tensordotCAL(qsite,top,ifc1=True)
	    top2[icase].tensordotCAL(cop,top1[icase])
	    top1[icase].value = None
	    top3[icase].tensordotCAL(top2[icase],qsite)
	    top2[icase].value = None
	    top3[icase].dump(fNp,'opers'+str(iop))
	    top3[icase].value = None

   return 0

# Construct renormalized operators for <0|I> or <0|R(theta)|I> (SP-MPS)
def renormQt_S0I(dmrg,isite,ncsite,flst,flstN,site,status):
   # Loop over operators
   if not dmrg.ifs2proj:
      qsite = site
   else:
      qsite = site.reduceQsymsToN()
   dims = numpy.array([6]+[dmrg.maxslc]*2)
   ncase = numpy.prod(dims)
   if status == 'L':

      # Excited states
      if dmrg.comm.rank == 0 and dmrg.ifex:
         lop = qtensor.qtensor()
         for iref in range(dmrg.nref):
      	    fket = dmrg.wfex[iref]
	    ksite = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
	    if not dmrg.ifs2proj:
	       fL = flst[1][2*iref]
	       fN = flstN[1][iref]
               # SOP
	       lop.load(fL,'mat')
               # For SOP, we only need to use site, rather the symmetry reduced one!
	       tmp = qtensor.tensordot(site,lop,axes=([0],[0]),ifc1=True)
               tmp = qtensor.tensordot(tmp,ksite,axes=([0,2],[1,0]))
	       tmp.dump(fN,'mat')
	    # POP
    	    else:
      	       qksite = ksite.reduceQsymsToN()
	       fLp = flst[1][2*iref]
	       fNp = flstN[1][iref]
               nop = dmrg.npts
   	       top = qtensor.qtensor()
   	       cop = qtensor.qtensor()
               # cases for symmetry template depending on ap^+Vp
               ncase = 1 
               ista = dict([(i,0) for i in range(ncase)])
               top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
               top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
               top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
               icase = 0
               for iop in range(nop):
	          cop.load(dmrg.fpop,'op'+str(iop))
                  top.load(fLp,'opers'+str(iop))
	          # Symmetry & Buffer
	          if ista[icase] == 0:
	             top1[icase].tensordotSYM(qsite,top,axes=([0],[1]))
	             top2[icase].tensordotSYM(cop,top1[icase],axes=([0,2],[2,0]))
	             top3[icase].tensordotSYM(top2[icase],qksite,axes=([1,3],[1,0]))
	             # Note that the idlst is not needed for POPs!
		     ## We dump the idlst information for operators & MPS
	             #top3[icase].idlst = [cop.idlst[1],qsite.idlst[2],qsite.idlst[2]]
	             ista[icase] = 1
	          top1[icase].tensordotCAL(qsite,top,ifc1=True)
	          top2[icase].tensordotCAL(cop,top1[icase])
	          top1[icase].value = None
	          top3[icase].tensordotCAL(top2[icase],qksite)
	          top2[icase].value = None
	          top3[icase].dump(fNp,'opers'+str(iop))
	          top3[icase].value = None

   elif status == 'R':

      jsite = isite+ncsite-1
      # Excited states
      if dmrg.comm.rank == 0 and dmrg.ifex:
         rop = qtensor.qtensor()
         for iref in range(dmrg.nref):
      	    fket = dmrg.wfex[iref]
	    ksite = mpo_dmrg_io.loadSite(fket,jsite,dmrg.ifQt)
            qout = fket['qnum'+str(dmrg.nsite)].value[0]
	    # L-MPS to R-MPS (convention) by changing qsyms of two bonds
	    # Adjust the symmetry for ksite, which also makes the sp-mps case work!
	    ksite.qsyms[0] = qout - ksite.qsyms[0]
	    ksite.qsyms[2] = qout - ksite.qsyms[2]
	    ksite.status[0] = ~ksite.status[0]
	    ksite.status[2] = ~ksite.status[2]
	    if not dmrg.ifs2proj:
	       fR = flst[1][2*iref+1]
	       fN = flstN[1][iref]
	       # SOP
	       rop.load(fR,'mat')
	       tmp = qtensor.tensordot(site,rop,axes=([2],[0]),ifc1=True)
	       tmp = qtensor.tensordot(tmp,ksite,axes=([1,2],[1,2]))
	       tmp.dump(fN,'mat')
	    # POP
    	    else:
      	       qksite = ksite.reduceQsymsToN()
	       fRp = flst[1][2*iref+1]
	       fNp = flstN[1][iref]
               nop = dmrg.npts
   	       top = qtensor.qtensor()
   	       cop = qtensor.qtensor()
               # cases for symmetry template depending on ap^+Vp
               ncase = 1 
               ista = dict([(i,0) for i in range(ncase)])
               top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
               top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
               top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
               icase = 0
               for iop in range(nop):
	          cop.load(dmrg.fpop,'op'+str(iop))
	          # Flip horizontal directions
	          cop.qsyms[0]  = -cop.qsyms[0]
	          cop.qsyms[1]  = -cop.qsyms[1]
	          cop.status[0] = ~cop.status[0] 
	          cop.status[1] = ~cop.status[1] 
                  top.load(fRp,'opers'+str(iop))
	          # Symmetry & Buffer
	          if ista[icase] == 0:
	             top1[icase].tensordotSYM(qsite,top,axes=([2],[1]))
	             top2[icase].tensordotSYM(cop,top1[icase],axes=([1,2],[2,1]))
	             top3[icase].tensordotSYM(top2[icase],qksite,axes=([1,3],[1,2]))
	             ## We dump the idlst information for operators & MPS
	             #top3[icase].idlst = [cop.idlst[0],qsite.idlst[0],qsite.idlst[0]]
	             ista[icase] = 1
	          top1[icase].tensordotCAL(qsite,top,ifc1=True)
	          top2[icase].tensordotCAL(cop,top1[icase])
	          top1[icase].value = None
	          top3[icase].tensordotCAL(top2[icase],qksite)
	          top2[icase].value = None
	          top3[icase].dump(fNp,'opers'+str(iop))
	          top3[icase].value = None

   return 0

# Construct renormalized operators for PT treatment:
# RHS = <0|H|I> or <0|H*R(theta)|I> (SP-MPS)
def renormQt_H0I(dmrg,isite,ncsite,flst,flstN,site,status,iHd=0,thresh=1.e-12):
   # Loop over operators
   if not dmrg.ifs2proj:
      qsite = site
   else:
      qsite = site.reduceQsymsToN()
   dims = numpy.array([6]+[dmrg.maxslc]*2)
   ncase = numpy.prod(dims)
   nop = dmrg.fhop['nops'].value
   if status == 'L':

      # Hamiltoinan operators for PT.
      if dmrg.ifpt:
         for iref in range(dmrg.nref):
	    if iHd == 0:	 
	       if abs(dmrg.coef[iref])<thresh: continue
               fL = flst[2][2*iref]
               fN = flstN[2][iref]
	    elif iHd == 1: 
               fL = flst[3][2*iref]
               fN = flstN[3][iref]
      	    fket = dmrg.wfex[iref]
	    ksite = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
	    if not dmrg.ifs2proj:
	       qksite = ksite	    
	    else:
      	       qksite = ksite.reduceQsymsToN()
            # Loop over operators
            lops = qtensor.Qt()
            cops = qtensor.Qt()
            # cases for symmetry template depending on ap^+Vp
            ista = dict([(i,-1) for i in range(ncase)])
            top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
            top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
            top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
            for iop in range(nop):
               # LOP
               lops.loadInfo(fL,'opers'+str(iop))
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
                     lop = lops.loadSLC(fL,'opers'+str(iop),islc)
                     # Symmetry & Buffer
                     icase = numpy.ravel_multi_index([icase0,jslc,islc],dims)
                     if ista[icase] == -1:
                        ilen1 = top1[icase].tensordotSYM(qsite,lop,axes=([0],[1]))
                        ilen2 = top2[icase].tensordotSYM(cop,top1[icase],axes=([0,2],[2,0]))
                        ilen3 = top3[icase].tensordotSYM(top2[icase],qksite,axes=([1,3],[1,0]))
                        ista[icase] = ilen3
                     if ista[icase] <= 0: continue
                     top1[icase].tensordotCAL(qsite,lop,ifc1=True)
                     lop = None
                     top2[icase].tensordotCAL(cop,top1[icase])
                     top1[icase].value = None
                     top3[icase].tensordotCAL(top2[icase],qksite)
                     top2[icase].value = None
                     if qops.size[jslc] == 0:
                        qops.dic[jslc].copy(top3[icase])
                        qops.size[jslc] = qops.dic[jslc].size_allowed
	                #*** Note that the idlst is not needed for HOPs! No need for diag ***
                        ## We dump the idlst information for operators & MPS
                        #qops.dic[jslc].idlst = [cop.idlst[1],qsite.idlst[2],qsite.idlst[2]]
                     else:
                        qops.dic[jslc].value += top3[icase].value  
                     top3[icase].value = None
                     lop = None
                  if qops.dic[jslc].size > 0: qops.dumpSLC(fN,'opers'+str(iop),jslc)
                  qops.dic[jslc] = None
               # DUMP information
               qops.dumpInfo(fN,'opers'+str(iop))

   elif status == 'R':

      jsite = isite+ncsite-1
      # Hamiltoinan operators for PT.
      if dmrg.ifpt:
         for iref in range(dmrg.nref):
	    if iHd == 0:
	       if abs(dmrg.coef[iref])<thresh: continue
	       fR = flst[2][2*iref+1]
	       fN = flstN[2][iref]
	    elif iHd == 1:
	       fR = flst[3][2*iref+1]
	       fN = flstN[3][iref]
      	    fket = dmrg.wfex[iref]
	    ksite = mpo_dmrg_io.loadSite(fket,jsite,dmrg.ifQt)
            qout = fket['qnum'+str(dmrg.nsite)].value[0]
	    # L-MPS to R-MPS (convention) by changing qsyms of two bonds
	    # Adjust the symmetry for ksite, which also make the sp-mps case work!
	    ksite.qsyms[0] = qout - ksite.qsyms[0]
	    ksite.qsyms[2] = qout - ksite.qsyms[2]
	    ksite.status[0] = ~ksite.status[0]
	    ksite.status[2] = ~ksite.status[2]
	    if not dmrg.ifs2proj:
	       qksite = ksite	    
	    else:
      	       qksite = ksite.reduceQsymsToN()
            # Loop over operators
            rops = qtensor.Qt()
            cops = qtensor.Qt()
            # cases for symmetry template depending on ap^+Vp
            ista = dict([(i,-1) for i in range(ncase)])
            top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
            top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
            top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
            for iop in range(nop):
               # ROP
               rops.loadInfo(fR,'opers'+str(iop))
               # COP
               pindx = dmrg.opers[iop]
               cops.load(dmrg.fhop,'site'+str(jsite)+'/op'+str(iop))
               # Symmetry & Buffer
               icase0 = qtensor_opers.case(pindx[0],jsite)
               # COMPUTE
               qops = qtensor.Qt(1,cops.maxslc[0])
               for jslc in range(cops.maxslc[0]):
                  for islc in range(cops.maxslc[1]):
                     ijdx = cops.ravel([jslc,islc])
                     if cops.size[ijdx] == 0 or rops.size[islc] == 0: continue
                     cop = cops.dic[ijdx]
                     rop = rops.loadSLC(fR,'opers'+str(iop),islc)
                     # Symmetry & Buffer
                     icase = numpy.ravel_multi_index([icase0,jslc,islc],dims)
                     if ista[icase] == -1:
                	# Flip horizontal directions
                        cop.qsyms[0]  = -cop.qsyms[0]
                        cop.qsyms[1]  = -cop.qsyms[1]
                        cop.status[0] = ~cop.status[0] 
                        cop.status[1] = ~cop.status[1] 
                        ilen1 = top1[icase].tensordotSYM(qsite,rop,axes=([2],[1]))
                        ilen2 = top2[icase].tensordotSYM(cop,top1[icase],axes=([1,2],[2,1]))
                        ilen3 = top3[icase].tensordotSYM(top2[icase],qksite,axes=([1,3],[1,2]))
                        ista[icase] = ilen3
                     if ista[icase] <= 0: continue
                     #
                     # Originally, the memory usage can be large, since top=O(KD^2) per each H[x]. 
                     # 535 1517.465 MiB  801.098 MiB top1[icase].tensordotCAL(qsite,top)
                     # 536 2423.266 MiB  905.801 MiB top2[icase].tensordotCAL(cop,top1[icase])
                     # Now the bond is sliced, such that the rop is smaller by 1/maxslc.
                     #
                     top1[icase].tensordotCAL(qsite,rop,ifc1=True)
                     rop = None
                     top2[icase].tensordotCAL(cop,top1[icase])
                     top1[icase].value = None
                     top3[icase].tensordotCAL(top2[icase],qksite)
                     top2[icase].value = None
                     if qops.size[jslc] == 0:
                        qops.dic[jslc].copy(top3[icase])
                        qops.size[jslc] = qops.dic[jslc].size_allowed
	                # Note that the idlst is not needed for HOPs!
                        ## We dump the idlst information for operators & MPS
                        #qops.dic[jslc].idlst = [cop.idlst[0],qsite.idlst[0],qsite.idlst[0]]
                     else:
                        qops.dic[jslc].value += top3[icase].value  
                     top3[icase].value = None
                  if qops.dic[jslc].size > 0: qops.dumpSLC(fN,'opers'+str(iop),jslc)
                  qops.dic[jslc] = None
               # DUMP information
               qops.dumpInfo(fN,'opers'+str(iop))
      
   return 0

# <lnr|Psi>
def PBasSopsQt(info,ndim0,prjmap,nref):
   dmrg,isite,ncsite,flst,status,ifsym = info
   ldim,cdim,rdim,ndim = dmrg.dims
   vt = numpy.zeros((nref,ndim0),dtype=dmrg_dtype)
   lop = qtensor.qtensor()
   rop = qtensor.qtensor()
   if ncsite == 1:
   
      for iref in range(nref):
	 fL = flst[1][2*iref]
	 fR = flst[1][2*iref+1]
         lop.load(fL,'mat')
         rop.load(fR,'mat')
	 fket = dmrg.wfex[iref]
         qout = fket['qnum'+str(dmrg.nsite)].value[0]
      	 rop.qsyms[0] = qout - rop.qsyms[0]
      	 rop.qsyms[1] = qout - rop.qsyms[1]
	 ksite1 = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
	 # L[u1,b]*A[b,n,c]*R[u2,c] = vec[u1,n,u2]
	 tmp = qtensor.tensordot(lop,ksite1,axes=([1],[0])) # LA[u1,n,c]
	 tmp = qtensor.tensordot(tmp,rop,axes=([2],[1]))    # LAR[u1,n,u2]
         vt[iref] = tmp.value

   elif ncsite == 2:

      assert cdim == 4*4
      for iref in range(nref):
	 fL = flst[1][2*iref]
	 fR = flst[1][2*iref+1]
         lop.load(fL,'mat')
         rop.load(fR,'mat')
      	 fket = dmrg.wfex[iref]
         qout = fket['qnum'+str(dmrg.nsite)].value[0]
      	 rop.qsyms[0] = qout - rop.qsyms[0]
      	 rop.qsyms[1] = qout - rop.qsyms[1]
	 ksite1 = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
	 ksite2 = mpo_dmrg_io.loadSite(fket,isite+1,dmrg.ifQt)
	 # L[u1,b]*A[b,n,c]*B[c,m,d]*R[u2,d] = vec[u1,n,m,u2]
	 tmp1 = qtensor.tensordot(lop,ksite1,axes=([1],[0])) # LA[u1,n,c]
	 tmp2 = qtensor.tensordot(ksite2,rop,axes=([2],[1])) # BR[c,m,u2]
	 tmp  = qtensor.tensordot(tmp1,tmp2,axes=([2],[0]))  # LABR[u1,n,m,u2]
	 vt[iref] = tmp.value

   return vt

# <lnr|P|Psi>
def PBasPopsQt(info,ndim0,prjmap,nref):
   dmrg,isite,ncsite,flst,status,ifsym = info
   ldim,cdim,rdim,ndim = dmrg.dims
   vt = numpy.zeros((nref,ndim0),dtype=dmrg_dtype)
   nop = dmrg.npts
   if ncsite == 1:

      for iref in range(nref):
	 fLp = flst[1][2*iref]
	 fRp = flst[1][2*iref+1]
	 fket = dmrg.wfex[iref]
	 ksite = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
	 vtensor = ksite.reduceQsymsToN()
         # temp operators
	 lop = qtensor.qtensor()
         rop = qtensor.qtensor()
         cop = qtensor.qtensor()
         # cases for symmetry template depending on ap^+Vp
         ncase = 1 
         ista = dict([(i,0) for i in range(ncase)])
         top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top4 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         icase = 0
         # Loop over operators
         for iop in range(nop):
            cop.load(dmrg.fpop,'op'+str(iop))
            # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
            lop.load(fLp,'opers'+str(iop))
            rop.load(fRp,'opers'+str(iop))
            # Flip symmetry
            rop.qsyms = [-rop.qsyms[0],dmrg.qkey[:1]-rop.qsyms[1],dmrg.qkey[:1]-rop.qsyms[2]] 
            # Symmetry & Buffer
            if ista[icase] == 0:
               top1[icase].tensordotSYM(lop,vtensor,axes=([2],[0]))
               top2[icase].tensordotSYM(top1[icase],cop,axes=([0,2],[0,3]))
               top3[icase].tensordotSYM(top2[icase],rop,axes=([1,2],[2,0]))
               ista[icase] = 1
            top1[icase].tensordotCAL(lop,vtensor)
            top2[icase].tensordotCAL(top1[icase],cop)
            top1[icase].value = None
            top3[icase].tensordotCAL(top2[icase],rop)
            top2[icase].value = None
            tmp = top3[icase].projectionNMs(dmrg.qtmp.qsyms)
            top3[icase].value = None
            vt[iref] += dmrg.qwts[iop]*tmp.value
            tmp = None

   elif ncsite == 2:

      assert cdim == 4*4 
      for iref in range(nref):
	 fLp = flst[1][2*iref]
	 fRp = flst[1][2*iref+1]
	 fket = dmrg.wfex[iref]
	 ksite1 = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
	 ksite2 = mpo_dmrg_io.loadSite(fket,isite+1,dmrg.ifQt)
         vtensor = qtensor.tensordot(ksite1,ksite2,axes=([2],[0]))
	 vtensor = vtensor.reduceQsymsToN()
         # temp operators
	 lop = qtensor.qtensor()
         rop = qtensor.qtensor()
         cop = qtensor.qtensor()
         # cases for symmetry template depending on ap^+Vp
         ncase = 1 
         ista = dict([(i,0) for i in range(ncase)])
         top1 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top2 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top3 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         top4 = dict([(i,qtensor.qtensor()) for i in range(ncase)])
         icase = 0
         # Loop over operators
         for iop in range(nop):
            cop.load(dmrg.fpop,'op'+str(iop))
            # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
            lop.load(fLp,'opers'+str(iop))
            rop.load(fRp,'opers'+str(iop))
            # Flip symmetry
            rop.qsyms = [-rop.qsyms[0],dmrg.qkey[:1]-rop.qsyms[1],dmrg.qkey[:1]-rop.qsyms[2]]
	    # Symmetry & Buffer
	    if ista[icase] == 0:
	       top1[icase].tensordotSYM(lop,vtensor,axes=([2],[0]))
	       top2[icase].tensordotSYM(top1[icase],cop,axes=([0,2],[0,3]))
	       top3[icase].tensordotSYM(top2[icase],cop,axes=([1,3],[3,0]))
	       top4[icase].tensordotSYM(top3[icase],rop,axes=([1,3],[2,0]))
	       ista[icase] = 1
	    top1[icase].tensordotCAL(lop,vtensor)
	    top2[icase].tensordotCAL(top1[icase],cop)
	    top1[icase].value = None
	    top3[icase].tensordotCAL(top2[icase],cop)
	    top2[icase].value = None
	    top4[icase].tensordotCAL(top3[icase],rop)
	    top3[icase].value = None
            tmp = top4[icase].projectionNMs(dmrg.qtmp.qsyms)
	    top4[icase].value = None
            vt[iref] += dmrg.qwts[iop]*tmp.value
	    tmp = None

   return vt

# RHS of PT = \sum_{m}<lnr|H|chi[m]>c[m], iHd=0
# or the set <lnr|Hd|chi[m]>, iHd=1.
def BVecQt(info,ndim0,prjmap,iHd,thresh=1.e-12):
   dmrg,isite,ncsite,flst,status,ifsym = info
   nop = dmrg.nops
   if iHd == 0:
      bvec = numpy.zeros(ndim0,dtype=dmrg_dtype)
   elif iHd == 1:
      bvec = numpy.zeros((dmrg.nref,ndim0),dtype=dmrg_dtype)
   if ncsite == 1:

      for iref in range(dmrg.nref):
	 if iHd == 0:
	    if abs(dmrg.coef[iref])<thresh: continue
	    fL = flst[2][2*iref]
	    fR = flst[2][2*iref+1]
	 elif iHd == 1:
	    fL = flst[3][2*iref]
	    fR = flst[3][2*iref+1]
	 fket = dmrg.wfex[iref]
	 qvec = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
	 if dmrg.ifs2proj: qvec = qvec.reduceQsymsToN()
         # temp operators
         lops = qtensor.Qt()
         rops = qtensor.Qt()
         cops = qtensor.Qt()
         # cases for symmetry template depending on ap^+Vp
         dims1 = numpy.array([8]+[dmrg.maxslc]*1)
         dims2 = numpy.array([8]+[dmrg.maxslc]*2)
         ncase1 = numpy.prod(dims1)
         ncase2 = numpy.prod(dims2)
         ista1 = dict([(i,-1) for i in range(ncase1)])
         ista2 = dict([(i,-1) for i in range(ncase2)])
         top1 = dict([(i,qtensor.qtensor()) for i in range(ncase1)])
         top2 = dict([(i,qtensor.qtensor()) for i in range(ncase2)])
         # Initializaiton
         top3 = dict([(i,qtensor.qtensor()) for i in range(ncase2)])
         # Loop over operators
         hvec = numpy.zeros(ndim0,dtype=dmrg_dtype)
         for iop in range(nop):
	    porb,ipop = dmrg.opers[iop]
	    # L/R-OP
            lops.loadInfo(fL,'opers'+str(iop))
            rops.loadInfo(fR,'opers'+str(iop))
            # WOP
            pindx = dmrg.opers[iop]
            cops.load(dmrg.fhop,'site'+str(isite)+'/op'+str(iop))
            # Symmetry & Buffer
            icase0 = qtensor_opers.case(pindx[0],isite)
            #   /       |	\
            #  |--- i --|-- j ---|
            #   \	|	/
            # LOOP OVER islc
            for islc in range(cops.maxslc[0]):
               if lops.size[islc] == 0: continue
               lop = lops.loadSLC(fL,'opers'+str(iop),islc)
               icase1 = numpy.ravel_multi_index([icase0,islc],dims1)
               if ista1[icase1] == -1:
                  ilen1 = top1[icase1].tensordotSYM(lop,qvec,axes=([2],[0]))
                  ista1[icase1] = ilen1
               if ista1[icase1] <= 0: continue
               top1[icase1].tensordotCAL(lop,qvec)
               lop = None

               # LOOP OVER jslc
               for jslc in range(cops.maxslc[1]):
                  ijdx = cops.ravel([islc,jslc])
                  if cops.size[ijdx] == 0 or rops.size[jslc] == 0: continue
                  cop = cops.dic[ijdx]
                  rop = lops.loadSLC(fR,'opers'+str(iop),jslc)
                  icase2 = numpy.ravel_multi_index([icase0,islc,jslc],dims2)
                  if ista2[icase2] == -1:
                     top2[icase2].tensordotSYM(top1[icase1],cop,axes=([0,2],[0,3]))
                     # Flip symmetry
                     if dmrg.ifs2proj: 
                        rop.qsyms = [-rop.qsyms[0],dmrg.qkey[:1]-rop.qsyms[1],dmrg.qkey[:1]-rop.qsyms[2]] 
                     else:
                        rop.qsyms = [-rop.qsyms[0],dmrg.qkey-rop.qsyms[1],dmrg.qkey-rop.qsyms[2]] 
               	     ilen2 = top3[icase2].tensordotSYM(top2[icase2],rop,axes=([1,2],[2,0]))
                     ista2[icase2] = ilen2
                  if ista2[icase2] <= 0: continue
                  top2[icase2].tensordotCAL(top1[icase1],cop)
                  top3[icase2].tensordotCAL(top2[icase2],rop)
                  rop = None
                  top2[icase2].value = None # jslc
                  if dmrg.ifs2proj:
                     # Assuming projectionNMs does not take too much time!
                     tmp = top3[icase2].projectionNMs(dmrg.qtmp.qsyms)
                     hvec += dmrg.hpwts[iop]*tmp.value
                     tmp = None
                  else:
                     hvec += dmrg.hpwts[iop]*top3[icase2].value
                  top3[icase2].value = None # jslc
               top1[icase1].value = None # islc

	 if iHd == 0:
	    bvec += hvec*dmrg.coef[iref]
         elif iHd == 1:
	    bvec[iref] = hvec.copy()
	   
   elif ncsite == 2:

      jsite = isite+1
      for iref in range(dmrg.nref):
	 if iHd == 0:
	    if abs(dmrg.coef[iref])<thresh: continue
	    fL = flst[2][2*iref]
	    fR = flst[2][2*iref+1]
	 elif iHd == 1:
	    fL = flst[3][2*iref]
	    fR = flst[3][2*iref+1]
	 fket = dmrg.wfex[iref]
	 ksite1 = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
	 ksite2 = mpo_dmrg_io.loadSite(fket,isite+1,dmrg.ifQt)
         qvec = qtensor.tensordot(ksite1,ksite2,axes=([2],[0]))
	 if dmrg.ifs2proj: qvec = qvec.reduceQsymsToN()
         # temp operators
         lops = qtensor.Qt()
         rops = qtensor.Qt()
         cops = qtensor.Qt()
         # cases for symmetry template depending on ap^+Vp
         dims1 = numpy.array([8]+[dmrg.maxslc]*1)
         dims2 = numpy.array([8]+[dmrg.maxslc]*2)
         ncase1 = numpy.prod(dims1)
         ncase2 = numpy.prod(dims2)
         ista1 = dict([(i,-1) for i in range(ncase1)])
         ista2 = dict([(i,-1) for i in range(ncase2)])
         top1 = dict([(i,qtensor.qtensor()) for i in range(ncase1)])
         top2 = dict([(i,qtensor.qtensor()) for i in range(ncase2)])
         # Initializaiton
         wops = qtensor.Qt()
         dims3 = numpy.array([8]+[dmrg.maxslc]*3)
         ncase3 = numpy.prod(dims3)
         ista3 = dict([(i,-1) for i in range(ncase3)])
         top3 = dict([(i,qtensor.qtensor()) for i in range(ncase3)])
         top4 = dict([(i,qtensor.qtensor()) for i in range(ncase3)])
         # Loop over operators
         hvec = numpy.zeros(ndim0,dtype=dmrg_dtype)
         for iop in range(nop):
	    porb,ipop = dmrg.opers[iop]
	    # L/R-OP
            lops.loadInfo(fL,'opers'+str(iop))
            rops.loadInfo(fR,'opers'+str(iop))
            # WOP
            pindx = dmrg.opers[iop]
            cops.load(dmrg.fhop,'site'+str(isite)+'/op'+str(iop))
            wops.load(dmrg.fhop,'site'+str(jsite)+'/op'+str(iop))
            # Symmetry & Buffer
            icase0 = qtensor_opers.case(pindx[0],isite,ncsite)
            #   /       |	|       \
            #  |--- i --|-- j --|-- k ---|
            #   \	|	|       /
            # LOOP OVER islc
            for islc in range(cops.maxslc[0]):
               if lops.size[islc] == 0: continue
               lop = lops.loadSLC(fL,'opers'+str(iop),islc)
               icase1 = numpy.ravel_multi_index([icase0,islc],dims1)
               if ista1[icase1] == -1:
                  ilen1 = top1[icase1].tensordotSYM(lop,qvec,axes=([2],[0]))
                  ista1[icase1] = ilen1
               if ista1[icase1] <= 0: continue
               top1[icase1].tensordotCAL(lop,qvec)
               lop = None

               # LOOP OVER jslc
               for jslc in range(cops.maxslc[1]):
                  ijdx = cops.ravel([islc,jslc])
                  if cops.size[ijdx] == 0: continue
                  cop = cops.dic[ijdx]
                  icase2 = numpy.ravel_multi_index([icase0,islc,jslc],dims2)
                  if ista2[icase2] == -1:
                     ilen2 = top2[icase2].tensordotSYM(top1[icase1],cop,axes=([0,2],[0,3]))
                     ista2[icase2] = ilen2
                  if ista2[icase2] <= 0: continue
                  top2[icase2].tensordotCAL(top1[icase1],cop)
               
                  # LOOP OVER kslc
                  for kslc in range(wops.maxslc[1]):
                     jkdx = wops.ravel([jslc,kslc])
                     if wops.size[jkdx] == 0 or rops.size[kslc] == 0: continue
                     wop = wops.dic[jkdx]
               	     rop = lops.loadSLC(fR,'opers'+str(iop),kslc)
                     icase3 = numpy.ravel_multi_index([icase0,islc,jslc,kslc],dims3)
                     if ista3[icase3] == -1:
                        # Flip symmetry (only once)
                        if dmrg.ifs2proj: 
                           rop.qsyms = [-rop.qsyms[0],dmrg.qkey[:1]-rop.qsyms[1],dmrg.qkey[:1]-rop.qsyms[2]] 
                        else:
                           rop.qsyms = [-rop.qsyms[0],dmrg.qkey-rop.qsyms[1],dmrg.qkey-rop.qsyms[2]] 
               	        top3[icase3].tensordotSYM(top2[icase2],wop,axes=([1,3],[3,0]))
               	        ilen3 = top4[icase3].tensordotSYM(top3[icase3],rop,axes=([1,3],[2,0]))
                        ista3[icase3] = ilen3
                     if ista3[icase3] <= 0: continue
            	     top3[icase3].tensordotCAL(top2[icase2],wop)
            	     top4[icase3].tensordotCAL(top3[icase3],rop)
           	     rop = None
            	     top3[icase3].value = None # kslc
            	     if dmrg.ifs2proj: 
            	        tmp = top4[icase3].projectionNMs(dmrg.qtmp.qsyms)
            	        hvec += dmrg.hpwts[iop]*tmp.value
            	        tmp = None
            	     else:
            	        hvec += dmrg.hpwts[iop]*top4[icase3].value
            	     top4[icase3].value = None # kslc
                  top2[icase2].value = None # jslc
               top1[icase1].value = None # islc

	 if iHd == 0:
	    bvec += hvec*dmrg.coef[iref]
         elif iHd == 1:
	    bvec[iref] = hvec.copy()

   return bvec
