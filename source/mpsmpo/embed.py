#
# Localization based on UNO from UHF/UKS check files
#
import numpy
import scipy.linalg
import h5py
from pyscf import tools,gto,scf,dft
from pyscf.tools import molden
from pyscf.future.lo import pmloc,ulocal

def sqrtm(s):
   e, v = numpy.linalg.eigh(s)
   return numpy.dot(v*numpy.sqrt(e), v.T.conj())

def lowdin(s):
   e, v = numpy.linalg.eigh(s)
   return numpy.dot(v/numpy.sqrt(e), v.T.conj())

def loadMolMf(fname):
   chkfile = fname+'.chk'
   outfile = fname+'_cmo.molden'
   tools.molden.from_chkfile(outfile, chkfile)
   mol,mf = scf.chkfile.load_scf(chkfile)
   mo_coeff = mf["mo_coeff"] # (2,K,K)
   return mol,mo_coeff

# Embedding basis
def genEmbedBasis(mol,mo_coeff,selectionRule,thresh=0.001,lao='meta_lowdin',debug=False,ifplot=True):
   print '\n[embed.genEmbedBasis] for unrestricted determinant'
   ova = mol.intor_symmetric("cint1e_ovlp_sph")
   nb = mo_coeff.shape[1]
   # Check overlap
   diff = reduce(numpy.dot,(mo_coeff[0].T,ova,mo_coeff[0])) - numpy.identity(nb)
   print ' (CtSC-I)[a]',numpy.linalg.norm(diff)
   diff = reduce(numpy.dot,(mo_coeff[1].T,ova,mo_coeff[1])) - numpy.identity(nb)
   print ' (CtSC-I)[b]',numpy.linalg.norm(diff)
   # UHF-alpha/beta
   ma = mo_coeff[0]
   mb = mo_coeff[1]
   nalpha = (mol.nelectron+mol.spin)/2
   nbeta  = (mol.nelectron-mol.spin)/2
   print ' nalpha/nbeta = ',(nalpha,nbeta)
   # Spin-averaged DM
   ma_occ = ma[:,:nalpha]
   mb_occ = mb[:,:nbeta]
   pTa = numpy.dot(ma_occ,ma_occ.T)
   pTb = numpy.dot(mb_occ,mb_occ.T)
   pT = pTa+pTb

   #------------------------------------ 
   # OAO basis
   #------------------------------------ 
   # Due to optimization by segmentation, 
   # the lowdin here do not correspond to
   # the idea lowdin OAO.
   if lao == 'bad_lowdin': 
      s12 = sqrtm(ova)
      s12inv = lowdin(ova)
   # Better choice: Pbas*|chiANO>
   elif lao == 'meta_lowdin':
      from pyscf import lo
      meta = lo.orth_ao(mol,method='meta_lowdin')
      diff = reduce(numpy.dot,(meta.T,ova,meta))-numpy.identity(nb)
      s12inv = meta.copy()
      s12 = numpy.linalg.inv(s12inv)
   #
   # Psi = chiAO*C
   #     = (chiAO*Y)*(Yinv*C)
   # DM in ortho basis = Yinv*C*n*C^T*Yinv^T
   # Only in lowdin basis Y^T=Y.
   #
   pTOAO = reduce(numpy.dot,(s12,pT,s12.T))
   #------------------------------------ 

   # Define impurity
   labels = mol.spheric_labels()
   fragBasis = []
   fragLabels = []
   for idx,item in enumerate(labels):
      ifselect = False
      if selectionRule(item):
         ifselect = True
      if ifselect:
         fragBasis.append(idx)
         fragLabels.append(item)
   print ' Define central fragment:'
   print ' No. of totalBasis:',nb
   print ' No. of fragBasis :',len(fragBasis)
   print ' Indices of fragBasis:',fragBasis
   print ' fragLabels:'
   for idx,item in enumerate(fragLabels):
      print '  idx = ',idx,' fragBas=',item
   compBasis=list(set(range(nb))-set(fragBasis))
   nfrag = len(fragBasis)
   ncomp = len(compBasis)
   # Fragment
   pTf = pTOAO[numpy.ix_(fragBasis,fragBasis)]
   ef, u = scipy.linalg.eigh(-pTf)
   ef = -ef
   ne_f = sum(ef)
   print ' Diag_values of pTf:\n',numpy.diag(pTf)
   print ' Eigenvalues of pTf:\n',ef
   uf = numpy.zeros((nb,nfrag))
   # Retain the locality for impurity
   #uf[fragBasis,:] = u
   uf[fragBasis,:] = numpy.identity(nfrag)
   # Complementary
   if ncomp > 0:
      pTc = pTOAO[numpy.ix_(compBasis,compBasis)]
      ec, v = scipy.linalg.eigh(-pTc)
      ec = -ec
      ne_c = sum(ec)
      print ' Eigenvalues of pTc:\n',ec
      cindx = []
      aindx = []
      vindx = []
      for i in range(ncomp):
         if abs(ec[i]-2.0)<thresh:
            cindx.append(i)
         elif abs(ec[i])<thresh:
            vindx.append(i)
         else:
            aindx.append(i)
      ncStrict = len(numpy.argwhere(abs(ec-2.0)<1.e-6))
      nvStrict = len(numpy.argwhere(abs(ec)<1.e-6))
      naStrict = ncomp - ncStrict - nvStrict
      nc = len(cindx)
      na = len(aindx)
      nv = len(vindx)
      vc = numpy.zeros((nb,nc))
      va = numpy.zeros((nb,na))
      vv = numpy.zeros((nb,nv))
      vc[compBasis,:] = v[:,cindx]
      va[compBasis,:] = v[:,aindx]
      vv[compBasis,:] = v[:,vindx]
      # Set up the proper ordering
      ucoeff = numpy.hstack((uf,va,vc,vv))
      print '-'*70
      print ' Final results for classification of basis with thresh=',thresh
      print '-'*70
      print ' (nf,na,nc,nv) = ',nfrag,na,nc,nv
      print ' (ncomp,ncStrict,nvStrict,naStrict) =',ncomp,ncStrict,nvStrict,naStrict
      print ' Eigen_na =\n',ec[aindx]
      if ifplot:
         import matplotlib.pyplot as plt
	 plt.plot(abs(ef),marker='o',linewidth=2.0)
	 plt.plot(abs(ec),marker='o',linewidth=2.0)
	 plt.show()
   else:
      ne_c = 0.0
      ucoeff = uf.copy()
   # Check
   pTu = reduce(numpy.dot,(ucoeff.T,pTOAO,ucoeff))
   print ' Nf =',ne_f,'Nc =',ne_c,'Nt =',ne_f+ne_c
   if debug: 
      print ' diagonal of pTu =' 
      print numpy.diag(pTu)
      print 'ucoeff\n',ucoeff
   # Back to AO basis
   basis = numpy.dot(s12inv,ucoeff)
   # Dump
   diff = reduce(numpy.dot,(basis.T,ova,basis)) - numpy.identity(nb)
   print ' CtSC-I=',numpy.linalg.norm(diff)
   with open('embas.molden','w') as thefile:
       molden.header(mol,thefile)
       molden.orbital_coeff(mol,thefile,basis)
   with open('cmoA.molden','w') as thefile:
       molden.header(mol,thefile)
       molden.orbital_coeff(mol,thefile,ma)
   with open('cmoB.molden','w') as thefile:
       molden.header(mol,thefile)
       molden.orbital_coeff(mol,thefile,mb)
   ua = reduce(numpy.dot,(basis.T,ova,ma_occ))
   ub = reduce(numpy.dot,(basis.T,ova,mb_occ))
   if debug: print ' ua\n',ua
   if debug: print ' ub\n',ub
   ia = abs(reduce(numpy.dot,(ua.T,ua)))
   ib = abs(reduce(numpy.dot,(ub.T,ub)))
   print ' diffIa=',numpy.linalg.norm(ia-numpy.identity(nalpha))
   print ' diffIb=',numpy.linalg.norm(ib-numpy.identity(nbeta))
   return basis,ua,ub


if __name__ == '__main__':
   ### EXAMPLES ###   
   fname = 'hs_bp86' 
   mol,mocoeff = loadMolMf(fname)
   def rule(item):
      #if item[1] == 'Fe' and (item[2] == '3d' or item[2] == '4d'):
      #if item[0] == 13 and item[1] == 'Fe' and item[2] == '3d':
      if item[0] == 13 and item[1] == 'Fe' and (item[2] == '3d' or item[2] == '4d'):
         return True
      else:
	 return False
   basis,ua,ub = genEmbedBasis(mol,mocoeff,rule,thresh=0.01)
   
   import detToMPS
   mps = detToMPS.unetworkSplit(ua,ub,threshVal=1.e-3,ifclass=True)
   exit()
   
   #CrudeMPS
   #mps = detToMPS.unetwork(ua,ub,threshVal=1.e-2,ifclass=True)
   mps = detToMPS.unetwork(ua,ub,threshVal=1.e-3,ifclass=True)
   from mpo_dmrg.source.tools import mpslib
   bdim = mpslib.mps_bdim(mps)
   print 'bdim=',bdim
   ## nc,nf,na,nv= 62 20 20 278
   #print 'bdim1=',len(bdim[62:82] ),bdim[62:82]
   #print 'bdim2=',len(bdim[82:102]),bdim[82:102]
