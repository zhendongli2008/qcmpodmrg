#!/usr/bin/env python
#
# Using MPO formalism, this part is extremely simple.
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def HDiag(info,ndim0,prjmap):
# def HDiagNQt(info,ndim0,prjmap):
# def linkDiag(cdiag,wdiag):
# def HVec(vec0,info,ndim0,prjmap):
# def HVecNQt(vec0,info,ndim0,prjmap,ifile=0,ioff=0):
# def SDiag(info,ndim0,prjmap):
# def SDiagNQt(info,ndim0,prjmap):
# def SVec(vec0,info,ndim0,prjmap):
# def SVecNQt(vec0,info,ndim0,prjmap):
# def pRDM(cimat,info):
# def pRDMNQt(cimat,info):
# def renorm(dmrg,isite,ncsite,flst,flstN,site,status):
# def renormNQt_H00(dmrg,isite,ncsite,flst,flstN,site,status):
# def renormNQt_S00(dmrg,isite,ncsite,flst,flstN,site,status):
# def renormNQt_S0I(dmrg,isite,ncsite,flst,flstN,site,status):
# def renormNQt_H0I(dmrg,isite,ncsite,flst,flstN,site,status,iHd=0,thresh=1.e-12):
#
# def PBas(info,ndim0,prjmap):
# def PBasOrtho(pbas,thresh=1.e-10):   
# def PBasSopsNQt(info,ndim0,prjmap,nref):
# def PBasPopsNQt(info,ndim0,prjmap,nref):
# def BVec(info,ndim0,prjmap,iHd=0):
# def BVecNQt(info,ndim0,prjmap,iHd,thresh=1.e-12):
# 
import time
import numpy
import scipy.linalg
import mpo_dmrg_io
import mpo_dmrg_kernelQt
from sysutil_include import dmrg_dtype,dmrg_mtype

# Diagonal term of H in Vl*Vr
def HDiag(info,ndim0,prjmap):
   dmrg = info[0]
   #------------------------------------------
   if dmrg.ifpt and dmrg.ifH0: dmrg.fdopXfhop()
   if dmrg.ifQt:
      diag = mpo_dmrg_kernelQt.HDiagQt(info,ndim0,prjmap) 
   else:
      diag = HDiagNQt(info,ndim0,prjmap)
   if dmrg.ifpt and dmrg.ifH0: dmrg.fdopXfhop()
   #------------------------------------------
   return diag

def HDiagNQt(info,ndim0,prjmap):
   dmrg,isite,ncsite,flst,status,ifsym = info
   fL,fR = flst[0][0:2]
   ldim,cdim,rdim,ndim = dmrg.dims
   if dmrg.iprt > 0: print '[mpo_dmrg_kernel.HDiag] (l,c,r,n,n0)=',(ldim,cdim,rdim,ndim,ndim0) 
   diag = numpy.zeros((ldim,cdim,rdim),dtype=dmrg_dtype)
   # Loop over operators
   nop = dmrg.fhop['nops'].value
   for iop in range(nop):
      cop = dmrg.fhop['site'+str(isite)+'/op'+str(iop)].value
      cdiag = numpy.einsum('abii->abi',cop)
      if ncsite == 2:
         jsite = isite+1
	 wop = dmrg.fhop['site'+str(jsite)+'/op'+str(iop)].value
   	 wdiag = numpy.einsum('bcjj->bcj',wop)
	 cdiag = linkDiag(cdiag,wdiag)
      # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
      lop = fL['opers'+str(iop)].value
      rop = fR['opers'+str(iop)].value
      ldiag = lop[:,range(ldim),range(ldim)]
      rdiag = rop[:,range(rdim),range(rdim)]
      # contract bonds
      tmp = numpy.tensordot(ldiag,cdiag,axes=([0],[0])) # al,abn->lnb  
      tmp = numpy.tensordot(tmp,rdiag,axes=([1],[0]))   # lnb,br->lnr  
      diag += dmrg.hpwts[iop]*tmp
   # Reshaping
   diag = diag.reshape(ndim)[prjmap].real.copy()
   # TEST
   debug = False #True
   if debug and ifsym: checkDiagNQt(diag,HVecNQt,info,ndim0,prjmap)
   return diag

def checkDiagNQt(diag,funNQt,vec,info,ndim0,prjmap):
   print ' checkDiagNQt: Debuging diag and funNQt ...'
   dmrg = info[0]
   assert dmrg.comm.size == 1
   Hmat = numpy.zeros((ndim0,ndim0),dtype=dmrg_dtype)
   for i in range(ndim0):
      vec = numpy.zeros(ndim0,dtype=dmrg_dtype)
      vec[i] = 1.0
      Hmat[:,i] = HVecNQt(vec,info,ndim0,prjmap)
      print ' idx=',i,'diag[i]=',diag[i],'Hii=',Hmat[i,i],'diff=',Hmat[i,i]-diag[i]
   diff1 = numpy.linalg.norm(Hmat-Hmat.T.conj())
   diff2 = numpy.linalg.norm(numpy.diag(Hmat)-diag)
   if dmrg.comm.size == 1:
      print ' diff1(H-H.T.conj)=',diff1 # diff1= 8.21020104329e-08
      if diff1 > 1.e-10:
         print ' error: Fmat is not symmetry!'
         exit(1)
   print ' diff2(H.diag-D)=',diff2 # diff2= 1.47061633021e-14
   if diff2 > 1.e-10:
      print ' error: diag and Fmat are not consistent!'
      exit(1)
   return 0

def linkDiag(cdiag,wdiag):
   #cdiag = numpy.einsum('abi,bcj->acij',cdiag,wdiag)
   cdiag = numpy.tensordot(cdiag,wdiag,axes=([1],[0])) # abi,bcj->aicj
   cdiag = cdiag.transpose(0,2,1,3) # aicj->acij
   s = cdiag.shape
   cdiag = cdiag.reshape(s[0],s[1],s[2]*s[3])
   return cdiag

# H*Vec
def HVec(vec0,info,ndim0,prjmap):
   dmrg = info[0]
   #------------------------------------------
   if dmrg.ifpt and dmrg.ifH0: dmrg.fdopXfhop()
   if dmrg.ifQt:
      hvec = mpo_dmrg_kernelQt.HVecQt(vec0,info,ndim0,prjmap) 
   else:
      hvec = HVecNQt(vec0,info,ndim0,prjmap)
   if dmrg.ifpt and dmrg.ifH0: dmrg.fdopXfhop()
   #------------------------------------------
   # PT case: [H0-E0](*P)
   if dmrg.ifpt and dmrg.comm.rank == 0:
      # Heff = H0-E0
      if not dmrg.ifs2proj:
 	 hvec -= dmrg.et*vec0
      # Heff = H0*P-E0*P
      else:
         svec = SVec(vec0,info,ndim0,prjmap)
	 hvec -= dmrg.et*svec
   return hvec

def HVecNQt(vec0,info,ndim0,prjmap,ifile=0,ioff=0):
   dmrg,isite,ncsite,flst,status,ifsym = info
   fL,fR = flst[ifile][ioff:ioff+2]
   ldim,cdim,rdim,ndim = dmrg.dims
   vtensor = numpy.zeros(ndim,dtype=dmrg_dtype)
   vtensor[prjmap] = vec0
   nop = dmrg.fhop['nops'].value
   if ncsite == 1:

      vtensor = vtensor.reshape(ldim,cdim,rdim)
      hvec = numpy.zeros(vtensor.shape,dtype=dmrg_dtype)
      # Loop over operators
      for iop in range(nop):
         cop = dmrg.fhop['site'+str(isite)+'/op'+str(iop)].value
         # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
         lop = fL['opers'+str(iop)].value
         rop = fR['opers'+str(iop)].value
         tmp = numpy.tensordot(lop,vtensor,axes=([2],[0])) # L[pia]*B[anb]=>T[pinb]
         tmp = numpy.tensordot(tmp,cop,axes=([0,2],[0,3])) # T[pinb]*W[pqmn]=>T[ibqm]
         tmp = numpy.tensordot(tmp,rop,axes=([1,2],[2,0])) # T[ibqm]*R[qjb]=>T[imj]
         hvec += dmrg.hpwts[iop]*tmp

   elif ncsite == 2:

      assert cdim == 4*4
      vtensor = vtensor.reshape(ldim,4,4,rdim)
      hvec = numpy.zeros(vtensor.shape,dtype=dmrg_dtype)
      jsite = isite+1
      # Loop over operators
      for iop in range(nop):
         cop = dmrg.fhop['site'+str(isite)+'/op'+str(iop)].value
         wop = dmrg.fhop['site'+str(jsite)+'/op'+str(iop)].value
         # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
         lop = fL['opers'+str(iop)].value
         rop = fR['opers'+str(iop)].value
         tmp = numpy.tensordot(lop,vtensor,axes=([2],[0])) # L[pia]*B[anmb]=>T[pinmb] (a)
         tmp = numpy.tensordot(tmp,cop,axes=([0,2],[0,3])) # T[pinmb]*W[pqen]=>T[imbqe] (p,n)
	 tmp = numpy.tensordot(tmp,wop,axes=([1,3],[3,0])) # T[imbqe]*W[qrfm]=>T[iberf] (m,q)
         tmp = numpy.tensordot(tmp,rop,axes=([1,3],[2,0])) # T[iberf]*R[rjb]=>T[iefj] (b,r)
         hvec += dmrg.hpwts[iop]*tmp

   # reshaping
   hvec = hvec.reshape(ndim)[prjmap]
   return hvec

# Diagonal term of P in Vl*Vr, similar to Diag
def SDiag(info,ndim0,prjmap):
   dmrg = info[0]
   if dmrg.ifQt:
      diag = mpo_dmrg_kernelQt.SDiagQt(info,ndim0,prjmap) 
   else:
      diag = SDiagNQt(info,ndim0,prjmap)
   return diag

def SDiagNQt(info,ndim0,prjmap):
   dmrg,isite,ncsite,flst,status,ifsym = info
   fL,fR,fLp,fRp = flst[0][0:4]
   ldim,cdim,rdim,ndim = dmrg.dims
   diag = numpy.zeros((ldim,cdim,rdim),dtype=dmrg_dtype)
   # Only rank-0 deal with <k|s2proj|k>
   if dmrg.comm.rank == 0 and dmrg.ifs2proj:
      nop = dmrg.npts
      # Loop over operators
      for iop in range(nop):
         cop = dmrg.fpop['op'+str(iop)].value
         cdiag = numpy.einsum('abii->abi',cop)
	 if ncsite == 2:
            jsite = isite+1
	    wop = dmrg.fpop['op'+str(iop)].value
   	    wdiag = numpy.einsum('bcjj->bcj',wop)
	    cdiag = linkDiag(cdiag,wdiag)
         # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
         lop = fLp['opers'+str(iop)].value
         rop = fRp['opers'+str(iop)].value
         ldiag = lop[:,range(ldim),range(ldim)]
         rdiag = rop[:,range(rdim),range(rdim)]
         # contract bonds
         tmp = numpy.tensordot(ldiag,cdiag,axes=([0],[0])) # lbn
         tmp = numpy.tensordot(tmp,rdiag,axes=([1],[0])) # lnr
         diag += dmrg.qwts[iop]*tmp
   # reshaping
   diag = diag.reshape(ndim)[prjmap].real.copy()
   # TEST
   debug = False #True
   if debug and ifsym: checkDiagNQt(diag,SVecNQt,info,ndim0,prjmap)
   return diag

# S*Vec
def SVec(vec0,info,ndim0,prjmap):
   if info[0].ifQt:
      svec = mpo_dmrg_kernelQt.SVecQt(vec0,info,ndim0,prjmap) 
   else:
      svec = SVecNQt(vec0,info,ndim0,prjmap) 
   return svec

def SVecNQt(vec0,info,ndim0,prjmap):
   dmrg,isite,ncsite,flst,status,ifsym = info
   fL,fR,fLp,fRp = flst[0][0:4]
   ldim,cdim,rdim,ndim = dmrg.dims
   svec = numpy.zeros((ldim,cdim,rdim),dtype=dmrg_dtype)
   if dmrg.comm.rank == 0 and dmrg.ifs2proj:
      vtensor = numpy.zeros(ndim,dtype=dmrg_dtype)
      vtensor[prjmap] = vec0
      nop = dmrg.npts
      if ncsite == 1:
      
	 vtensor = vtensor.reshape(ldim,cdim,rdim)
         # Loop over operators
         for iop in range(nop):
            cop = dmrg.fpop['op'+str(iop)].value
            # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
            lop = fLp['opers'+str(iop)].value
            rop = fRp['opers'+str(iop)].value
            tmp = numpy.tensordot(lop,vtensor,axes=([2],[0])) # L[pia]*B[anb]=>T[pinb]
            tmp = numpy.tensordot(tmp,cop,axes=([0,2],[0,3])) # T[pinb]*W[pqmn]=>T[ibqm]
            tmp = numpy.tensordot(tmp,rop,axes=([1,2],[2,0])) # T[ibqm]*R[qjb]=>T[imj]
            svec += dmrg.qwts[iop]*tmp

      elif ncsite == 2:

         assert cdim == 4*4
	 vtensor = vtensor.reshape(ldim,4,4,rdim)
         svec = svec.reshape(ldim,4,4,rdim)
         # Loop over operators
         for iop in range(nop):
            cop = dmrg.fpop['op'+str(iop)].value
            # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
            lop = fLp['opers'+str(iop)].value
            rop = fRp['opers'+str(iop)].value
            tmp = numpy.tensordot(lop,vtensor,axes=([2],[0])) # L[pia]*B[anmb]=>T[pinmb] (a)
            tmp = numpy.tensordot(tmp,cop,axes=([0,2],[0,3])) # T[pinmb]*W[pqen]=>T[imbqe] (p,n)
	    tmp = numpy.tensordot(tmp,cop,axes=([1,3],[3,0])) # T[imbqe]*W[qrfm]=>T[iberf] (m,q)
            tmp = numpy.tensordot(tmp,rop,axes=([1,3],[2,0])) # T[iberf]*R[rjb]=>T[iefj] (b,r)
            svec += dmrg.qwts[iop]*tmp

   # reshaping
   svec = svec.reshape(ndim)[prjmap]
   return svec

# perturbed RDM on superblock
def pRDM(cimat,info):
   if info[0].ifQt:
      rdm1 = mpo_dmrg_kernelQt.pRDMQt(cimat,info)
   else:
      rdm1 = pRDMNQt(cimat,info)
   return rdm1

def pRDMNQt(cimat,info):
   dmrg,isite,ncsite,flst,status,ifsym = info
   fL,fR,fLp,fRp = flst[0][0:4]
   ldim,cdim,rdim,ndim = dmrg.dims
   if dmrg.iprt > 0: print '[mpo_dmrg_kernel.pRDM] (l,c,r,n)=',(ldim,cdim,rdim,ndim)
   neig,diml,dimc,dimr = cimat.shape
   if status == 'L':
      
      rdm1 = numpy.zeros((diml*dimc,diml*dimc),dtype=dmrg_dtype)
      # pRDM as noise
      if dmrg.inoise == 0 and dmrg.noise >= 1.e-10:
	 # Loop over operators
         nop = dmrg.nops
         for iop in range(nop):
            cop = dmrg.fhop['site'+str(isite)+'/op'+str(iop)].value
            lop = fL['opers'+str(iop)].value
            for ieig in range(neig):	 
               # L[pia]*B[anb]=>T[pinb]
               tmp = numpy.tensordot(lop,cimat[ieig],axes=([2],[0]))
               # T[pinb]*W[pqmn]=>T[ibqm]
               tmp = numpy.tensordot(tmp,cop,axes=([0,2],[0,3]))
               # T[ibqm]->T[im,bq] - O(KD2) memory
               tmp = tmp.transpose(0,3,1,2)
               s = tmp.shape
               tmp = tmp.reshape(s[0]*s[1],s[2]*s[3])
               rdm1 += numpy.dot(tmp,tmp.T.conj())*dmrg.noise
      if dmrg.comm.rank == 0:
  	 s = cimat[0].shape
	 for ieig in range(neig):	 
	    tmp = cimat[ieig].reshape(s[0]*s[1],s[2])
	    rdm1 += numpy.dot(tmp,tmp.T.conj())

   elif status == 'R':
      
      rdm1 = numpy.zeros((dimc*dimr,dimc*dimr),dtype=dmrg_dtype)
      jsite = isite+ncsite-1
      # pRDM as noise
      if dmrg.inoise == 0 and dmrg.noise >= 1.e-10:
         # Loop over operators
         nop = dmrg.nops
         for iop in range(nop):
            cop = dmrg.fhop['site'+str(jsite)+'/op'+str(iop)].value
            rop = fR['opers'+str(iop)].value
            for ieig in range(neig):	 
               # B[anb]*R[qjb]=>T[anqj]
               tmp = numpy.tensordot(cimat[ieig],rop,axes=([2],[2]))
               # W[pqmn]*T[anqj]=>T[pmaj]
               tmp = numpy.tensordot(cop,tmp,axes=([1,3],[2,1]))
               # T[pmaj]->T[pa,mj] - O(KD2) memory
               tmp = tmp.transpose(0,2,1,3)
               s = tmp.shape
               tmp = tmp.reshape(s[0]*s[1],s[2]*s[3])
               rdm1 += numpy.dot(tmp.T.conj(),tmp)*dmrg.noise
      if dmrg.comm.rank == 0:
  	 s = cimat[0].shape
	 for ieig in range(neig):	 
	    tmp = cimat[ieig].reshape(s[0],s[1]*s[2])
	    rdm1 += numpy.dot(tmp.T.conj(),tmp)

   return rdm1

# Renormalizaiton of operators: local operations without communication!
def renorm(dmrg,isite,ncsite,flst,flstN,site,status):
   if dmrg.iprt > 0: print '[mpo_dmrg_kernel.renorm] status=',status
   #------------------------------------------
   if dmrg.ifpt and dmrg.ifH0: dmrg.fdopXfhop()
   if not dmrg.ifQt:
      renormNQt_H00(dmrg,isite,ncsite,flst,flstN,site,status)
   else:
      mpo_dmrg_kernelQt.renormQt_H00(dmrg,isite,ncsite,flst,flstN,site,status)
   if dmrg.ifpt and dmrg.ifH0: dmrg.fdopXfhop()
   #------------------------------------------
   if not dmrg.ifQt:
      # lrpop
      renormNQt_S00(dmrg,isite,ncsite,flst,flstN,site,status)
      # lrsop
      renormNQt_S0I(dmrg,isite,ncsite,flst,flstN,site,status)
      # lrhop
      renormNQt_H0I(dmrg,isite,ncsite,flst,flstN,site,status)
      if dmrg.ifpt and dmrg.ifH0:
         # lrdop
	 dmrg.fdopXfhop()
	 renormNQt_H0I(dmrg,isite,ncsite,flst,flstN,site,status,iHd=1)
	 dmrg.fdopXfhop()
   else:
      mpo_dmrg_kernelQt.renormQt_S00(dmrg,isite,ncsite,flst,flstN,site,status)
      mpo_dmrg_kernelQt.renormQt_S0I(dmrg,isite,ncsite,flst,flstN,site,status)
      mpo_dmrg_kernelQt.renormQt_H0I(dmrg,isite,ncsite,flst,flstN,site,status)
      if dmrg.ifpt and dmrg.ifH0:
	 dmrg.fdopXfhop()
	 mpo_dmrg_kernelQt.renormQt_H0I(dmrg,isite,ncsite,flst,flstN,site,status,iHd=1)
	 dmrg.fdopXfhop()
   return 0

# Construct renormalized operators for <0|H*R(theta)|0> or <0|H0|0> 
def renormNQt_H00(dmrg,isite,ncsite,flst,flstN,site,status):
   nop = dmrg.fhop['nops'].value
   if status == 'L':

      # Hamiltoinan operators
      fL = flst[0][0]
      fN = flstN[0][0]
      for iop in range(nop):
         cop = dmrg.fhop['site'+str(isite)+'/op'+str(iop)].value
         lop = fL['opers'+str(iop)].value
	 tmp = numpy.tensordot(site.conj(),lop,axes=([0],[1]))
	 tmp = numpy.tensordot(cop,tmp,axes=([0,2],[2,0]))
	 tmp = numpy.tensordot(tmp,site,axes=([1,3],[1,0]))
	 fN['opers'+str(iop)] = tmp

   elif status == 'R':

      jsite = isite+ncsite-1
      # Hamiltoinan operators
      fR = flst[0][1]
      fN = flstN[0][0]
      for iop in range(nop):
         cop = dmrg.fhop['site'+str(jsite)+'/op'+str(iop)].value
         rop = fR['opers'+str(iop)].value
	 tmp = numpy.tensordot(site.conj(),rop,axes=([2],[1]))
	 tmp = numpy.tensordot(cop,tmp,axes=([1,2],[2,1]))
	 tmp = numpy.tensordot(tmp,site,axes=([1,3],[1,2]))
	 fN['opers'+str(iop)] = tmp

   return 0

# Construct renormalized operators for <0|R(theta)|0> (SP-MPS)
def renormNQt_S00(dmrg,isite,ncsite,flst,flstN,site,status):
   if status == 'L':

      # Deal with the projector	on rank-0 
      if dmrg.comm.rank == 0 and dmrg.ifs2proj:
	 fLp = flst[0][2]
	 fNp = flstN[0][1]
         nop = dmrg.npts
         for iop in range(nop):
            cop = dmrg.fpop['op'+str(iop)].value
            lop = fLp['opers'+str(iop)].value
            tmp = numpy.tensordot(site.conj(),lop,axes=([0],[1]))
            tmp = numpy.tensordot(cop,tmp,axes=([0,2],[2,0]))
            tmp = numpy.tensordot(tmp,site,axes=([1,3],[1,0]))
	    fNp['opers'+str(iop)] = tmp

   elif status == 'R':

      jsite = isite+ncsite-1
      # Deal with the projector	on rank-0 
      if dmrg.comm.rank == 0 and dmrg.ifs2proj:
	 fRp = flst[0][3]
	 fNp = flstN[0][1]
         nop = dmrg.npts
         for iop in range(nop):
            cop = dmrg.fpop['op'+str(iop)].value
            rop = fRp['opers'+str(iop)].value
	    tmp = numpy.tensordot(site.conj(),rop,axes=([2],[1]))
	    tmp = numpy.tensordot(cop,tmp,axes=([1,2],[2,1]))
	    tmp = numpy.tensordot(tmp,site,axes=([1,3],[1,2]))
	    fNp['opers'+str(iop)] = tmp

   return 0

# Construct renormalized operators for <0|I> or <0|R(theta)|I> (SP-MPS)
def renormNQt_S0I(dmrg,isite,ncsite,flst,flstN,site,status):
   if status == 'L':

      # Excited states
      if dmrg.comm.rank == 0 and dmrg.ifex:
         for iref in range(dmrg.nref):
      	    fket = dmrg.wfex[iref]
	    ksite = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
	    if not dmrg.ifs2proj:
	       fL = flst[1][2*iref]
	       fN = flstN[1][iref]
               # SOP
	       tmp = fL['mat'].value
               tmp = numpy.tensordot(site.conj(),tmp,axes=([0],[0]))
               tmp = numpy.tensordot(tmp,ksite,axes=([0,2],[1,0]))
	       fN['mat'] = tmp
	    # POP
    	    else:
	       fLp = flst[1][2*iref]
	       fNp = flstN[1][iref]
               nop = dmrg.npts
               for iop in range(nop):
                  cop = dmrg.fpop['op'+str(iop)].value
                  lop = fLp['opers'+str(iop)].value
                  tmp = numpy.tensordot(site.conj(),lop,axes=([0],[1]))
                  tmp = numpy.tensordot(cop,tmp,axes=([0,2],[2,0]))
                  tmp = numpy.tensordot(tmp,ksite,axes=([1,3],[1,0]))
	          fNp['opers'+str(iop)] = tmp

   elif status == 'R':

      jsite = isite+ncsite-1
      # Excited states
      if dmrg.comm.rank == 0 and dmrg.ifex:
         for iref in range(dmrg.nref):
      	    fket = dmrg.wfex[iref]
	    ksite = mpo_dmrg_io.loadSite(fket,jsite,dmrg.ifQt)
	    if not dmrg.ifs2proj:
	       fR = flst[1][2*iref+1]
	       fN = flstN[1][iref]
               # SOP
	       tmp = fR['mat'].value
	       tmp = numpy.tensordot(site.conj(),tmp,axes=([2],[0]))
	       tmp = numpy.tensordot(tmp,ksite,axes=([1,2],[1,2]))
	       fN['mat'] = tmp
	    # POP
    	    else:
	       fRp = flst[1][2*iref+1]
	       fNp = flstN[1][iref]
               nop = dmrg.npts
               for iop in range(nop):
                  cop = dmrg.fpop['op'+str(iop)].value
                  rop = fRp['opers'+str(iop)].value
	          tmp = numpy.tensordot(site.conj(),rop,axes=([2],[1]))
	          tmp = numpy.tensordot(cop,tmp,axes=([1,2],[2,1]))
	          tmp = numpy.tensordot(tmp,ksite,axes=([1,3],[1,2]))
	          fNp['opers'+str(iop)] = tmp

   return 0

# Construct renormalized operators for PT treatment:
# RHS = <0|H|I> or <0|H*R(theta)|I> (SP-MPS)
def renormNQt_H0I(dmrg,isite,ncsite,flst,flstN,site,status,iHd=0,thresh=1.e-12):
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
            for iop in range(nop):
               cop = dmrg.fhop['site'+str(isite)+'/op'+str(iop)].value
               lop = fL['opers'+str(iop)].value
               tmp = numpy.tensordot(site.conj(),lop,axes=([0],[1]))
               tmp = numpy.tensordot(cop,tmp,axes=([0,2],[2,0]))
               tmp = numpy.tensordot(tmp,ksite,axes=([1,3],[1,0]))
               fN['opers'+str(iop)] = tmp

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
            for iop in range(nop):
               cop = dmrg.fhop['site'+str(jsite)+'/op'+str(iop)].value
               rop = fR['opers'+str(iop)].value
               tmp = numpy.tensordot(site.conj(),rop,axes=([2],[1]))
               tmp = numpy.tensordot(cop,tmp,axes=([1,2],[2,1]))
               tmp = numpy.tensordot(tmp,ksite,axes=([1,3],[1,2]))
               fN['opers'+str(iop)] = tmp

   return 0

# PBas - project |0> on the renormalized basis {|l1n1r1>}
def PBas(info,ndim0,prjmap):
   dmrg = info[0]
   # No. of reference to construct projector <0|b>
   nref = dmrg.nref
   # <A|B> metric
   if not dmrg.ifs2proj:
      if dmrg.ifQt:
         pbas = mpo_dmrg_kernelQt.PBasSopsQt(info,ndim0,prjmap,nref) 
      else:
         pbas = PBasSopsNQt(info,ndim0,prjmap,nref)
   # <A|P|B> metric
   else:
      if dmrg.ifQt:
         pbas = mpo_dmrg_kernelQt.PBasPopsQt(info,ndim0,prjmap,nref) 
      else:
         pbas = PBasPopsNQt(info,ndim0,prjmap,nref)
   print '[mpo_dmrg_kenrel.PBas] nref =',dmrg.nref,' pbas.shape =',pbas.shape
   return pbas

# From the projected vectors to orthonormal basis
def PBasOrtho(pbas,thresh=1.e-12):
   # QR factorization - (ndim,nstate) => [nstate,ndim] to be in accordance with dvdson.
   u,sigs,vt = scipy.linalg.svd(pbas,full_matrices=False)
   ntot = len(sigs)
   nsig = len(numpy.argwhere(sigs>thresh))
   print ' nsig(>1.e-12) =',nsig,' ntot =',ntot,' singular values = ',sigs
   u = u[:,:nsig].copy()
   vt = vt[:nsig].copy()
   return u,sigs,vt

# <lnr|Psi>
def PBasSopsNQt(info,ndim0,prjmap,nref):
   dmrg,isite,ncsite,flst,status,ifsym = info
   ldim,cdim,rdim,ndim = dmrg.dims
   vt = numpy.zeros((nref,ndim0),dtype=dmrg_dtype)
   if ncsite == 1:
   
      for iref in range(nref):
	 fL = flst[1][2*iref]
	 fR = flst[1][2*iref+1]
         lop = fL['mat'].value
         rop = fR['mat'].value
      	 fket = dmrg.wfex[iref]
	 ksite1 = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
	 # L[u1,b]*A[b,n,c]*R[u2,c] = vec[u1,n,u2]
	 tmp1 = numpy.tensordot(lop,ksite1,axes=([1],[0])) # LA[u1,n,c]
	 tmp  = numpy.tensordot(tmp1,rop,axes=([2],[1]))   # LAR[u1,n,u2]
	 vt[iref] = tmp.reshape(ndim)[prjmap]

   elif ncsite == 2:

      assert cdim == 4*4
      for iref in range(nref):
	 fL = flst[1][2*iref]
	 fR = flst[1][2*iref+1]
         lop = fL['mat'].value
         rop = fR['mat'].value
      	 fket = dmrg.wfex[iref]
	 ksite1 = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
	 ksite2 = mpo_dmrg_io.loadSite(fket,isite+1,dmrg.ifQt)
	 # L[u1,b]*A[b,n,c]*B[c,m,d]*R[u2,d] = vec[u1,n,m,u2]
	 tmp1 = numpy.tensordot(lop,ksite1,axes=([1],[0])) # LA[u1,n,c]
	 tmp2 = numpy.tensordot(ksite2,rop,axes=([2],[1])) # BR[c,m,u2]
	 tmp  = numpy.tensordot(tmp1,tmp2,axes=([2],[0])) # LABR[u1,n,m,u2]
	 vt[iref] = tmp.reshape(ndim)[prjmap]

   return vt

# <lnr|P|Psi>
def PBasPopsNQt(info,ndim0,prjmap,nref):
   dmrg,isite,ncsite,flst,status,ifsym = info
   ldim,cdim,rdim,ndim = dmrg.dims
   vt = numpy.zeros((nref,ndim0),dtype=dmrg_dtype)
   nop = dmrg.npts
   if ncsite == 1:
     
      for iref in range(nref):
         fLp = flst[1][2*iref]
	 fRp = flst[1][2*iref+1]
         fket = dmrg.wfex[iref]
	 vtensor = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
         # Loop over operators
         for iop in range(nop):
	    #    /    |    \
	    # L *==  W[p] ==* R
	    #    \    |    /
 	    #     ----*----
	    #        Ket
            cop = dmrg.fpop['op'+str(iop)].value
            lop = fLp['opers'+str(iop)].value
            rop = fRp['opers'+str(iop)].value
            tmp = numpy.tensordot(lop,vtensor,axes=([2],[0])) # L[pia]*B[anb]=>T[pinb]
            tmp = numpy.tensordot(tmp,cop,axes=([0,2],[0,3])) # T[pinb]*W[pqmn]=>T[ibqm]
            tmp = numpy.tensordot(tmp,rop,axes=([1,2],[2,0])) # T[ibqm]*R[qjb]=>T[imj]
            vt[iref] += dmrg.qwts[iop]*tmp.reshape(ndim)[prjmap]

   elif ncsite == 2:

      assert cdim == 4*4
      for iref in range(nref):
         fLp = flst[1][2*iref]
         fRp = flst[1][2*iref+1]
         fket = dmrg.wfex[iref]
         ksite1 = mpo_dmrg_io.loadSite(fket,isite,dmrg.ifQt)
         ksite2 = mpo_dmrg_io.loadSite(fket,isite+1,dmrg.ifQt)
         vtensor = numpy.tensordot(ksite1,ksite2,axes=([2],[0]))
         # Loop over operators
         for iop in range(nop):
            cop = dmrg.fpop['op'+str(iop)].value
            lop = fLp['opers'+str(iop)].value
            rop = fRp['opers'+str(iop)].value
            tmp = numpy.tensordot(lop,vtensor,axes=([2],[0])) # L[pia]*B[anmb]=>T[pinmb] (a)
            tmp = numpy.tensordot(tmp,cop,axes=([0,2],[0,3])) # T[pinmb]*W[pqen]=>T[imbqe] (p,n)
            tmp = numpy.tensordot(tmp,cop,axes=([1,3],[3,0])) # T[imbqe]*W[qrfm]=>T[iberf] (m,q)
            tmp = numpy.tensordot(tmp,rop,axes=([1,3],[2,0])) # T[iberf]*R[rjb]=>T[iefj] (b,r)
            vt[iref] += dmrg.qwts[iop]*tmp.reshape(ndim)[prjmap]

   return vt

# RHS of PT = \sum_{m}<lnr|H|chi[m]>c[m], iHd=0
# or the set <lnr|Hd|chi[m]>, iHd=1.
def BVec(info,ndim0,prjmap,iHd=0):
   dmrg = info[0]
   #------------------------------------------
   # Construct RHS for Hd
   if iHd == 1: dmrg.fdopXfhop()
   if dmrg.ifQt:
      bvec = mpo_dmrg_kernelQt.BVecQt(info,ndim0,prjmap,iHd) 
   else:
      bvec = BVecNQt(info,ndim0,prjmap,iHd)
   if iHd == 1: dmrg.fdopXfhop()
   #------------------------------------------
   return bvec

def BVecNQt(info,ndim0,prjmap,iHd,thresh=1.e-12):
   dmrg,isite,ncsite,flst,status,ifsym = info
   ldim,cdim,rdim,ndim = dmrg.dims
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
	 hvec = numpy.zeros((ldim,cdim,rdim),dtype=dmrg_dtype) 
         # Loop over operators
         for iop in range(nop):
	    porb,ipop = dmrg.opers[iop]
            cop = dmrg.fhop['site'+str(isite)+'/op'+str(iop)].value
            # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
            lop = fL['opers'+str(iop)].value
            rop = fR['opers'+str(iop)].value
            tmp = numpy.tensordot(lop,qvec,axes=([2],[0]))    # L[pia]*B[anb]=>T[pinb]
            tmp = numpy.tensordot(tmp,cop,axes=([0,2],[0,3])) # T[pinb]*W[pqmn]=>T[ibqm]
            tmp = numpy.tensordot(tmp,rop,axes=([1,2],[2,0])) # T[ibqm]*R[qjb]=>T[imj]
            hvec += dmrg.hpwts[iop]*tmp

	 if iHd == 0:
	    bvec += hvec.reshape(ndim)[prjmap]*dmrg.coef[iref]
	 elif iHd == 1:
	    bvec[iref] = hvec.reshape(ndim)[prjmap]

   elif ncsite == 2:

      jsite = isite+1
      assert cdim == 4*4
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
         qvec = numpy.tensordot(ksite1,ksite2,axes=([2],[0]))
	 hvec = numpy.zeros((ldim,4,4,rdim),dtype=dmrg_dtype) 
         # Loop over operators
         for iop in range(nop):
	    porb,ipop = dmrg.opers[iop]
	    cop = dmrg.fhop['site'+str(isite)+'/op'+str(iop)].value
            wop = dmrg.fhop['site'+str(jsite)+'/op'+str(iop)].value
            # lop[a,bdim1,kdim1]*cop[a,b,ndim,ndim']*rop[b,bdim2,kdim2]
            lop = fL['opers'+str(iop)].value
            rop = fR['opers'+str(iop)].value
	    tmp = numpy.tensordot(lop,qvec,axes=([2],[0]))    # L[pia]*B[anmb]=>T[pinmb] (a)
            tmp = numpy.tensordot(tmp,cop,axes=([0,2],[0,3])) # T[pinmb]*W[pqen]=>T[imbqe] (p,n)
            tmp = numpy.tensordot(tmp,wop,axes=([1,3],[3,0])) # T[imbqe]*W[qrfm]=>T[iberf] (m,q)
            tmp = numpy.tensordot(tmp,rop,axes=([1,3],[2,0])) # T[iberf]*R[rjb]=>T[iefj] (b,r)
            hvec += dmrg.hpwts[iop]*tmp

	 if iHd == 0:
	    bvec += hvec.reshape(ndim)[prjmap]*dmrg.coef[iref]
	 elif iHd == 1:
	    bvec[iref] = hvec.reshape(ndim)[prjmap]

   return bvec
