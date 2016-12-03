import h5py
import numpy
import scipy.linalg
from qcmpodmrg.source.tools import fielder

#
# Provide the basic interface
#
class iface:
   def __init__(self,mol,mf):
      self.iflowdin = False
      self.iflocal = False
      self.ifreorder = False
      # Interface
      self.mol  = mol
      self.mf   = mf
      self.nelec= mol.nelectron
      self.spin = mol.spin
      self.nalpha = (mol.nelectron+mol.spin)/2
      self.nbeta  = (mol.nelectron-mol.spin)/2
      try: 
	 self.nbas = mf.mo_coeff[0].shape[0]
      except:
	 self.nbas = 0     
      self.mo_coeff = mf.mo_coeff
      self.lmo_coeff = None
      # frozen core
      self.nfrozen = 0
      self.ncut = 0

   def fci(self,nelecs=None):
      print '\n[iface.fci]'	   
      from pyscf import mcscf,fci
      ehf = self.mf.energy_tot(self.mf.make_rdm1())
      if nelecs is None:
         na = (self.mol.nelectron + self.mol.spin)/2
         nb = (self.mol.nelectron - self.mol.spin)/2
      else:
         na,nb = nelecs	 
      if na == nb:
         fcisol = fci.FCI(self.mol,self.mo_coeff)
      else:
	 fcisol = fci.FCI(self.mol,self.mo_coeff,singlet=False)
      #fcisol.nroots = 10
      fcisol.max_cycle = 300
      fcisol.max_space = 300
      #fcisol.conv_tol = 1.e-16
      fci.addons.fix_spin_(fcisol,0.1)
      efci,civec = fcisol.kernel(nelec=[na,nb])
      ecor = efci - ehf
      enuc = self.mol.energy_nuc()
      print "\nSummary of FCI:"
      print "nalpha,nbeta   =",na,nb
      print "E_scf(wo/wNUC) =",ehf - enuc,ehf
      print "E_fci(wo/wNUC) =",efci,efci+enuc
      print "E_cor =",ecor
      #print "Coeff:",mc.ci
      #print "Weights:",mc.ci**2
      return efci

   def ccsd(self):
      print '\n[iface.ccsd]'	   
      from pyscf import cc
      pcc=cc.ccsd.CC(self.mf)
      pcc.ccsd()
      return 0
       
   def lowdin(self):    
      print '\n[iface.lowdin]'	   
      s1e = self.mf.get_ovlp()
      self.lmo_coeff = lowdin(s1e)
      return 0

   def local(self):
      print '\n[iface.local]'	   
      if self.iflowdin:
	 self.lowdin() 
      else:
	 print 'Assuming closed-shell case, do split localization using PMloc'
 	 from pyscf.future.lo import pmloc
	 if self.nalpha == self.nbeta:
            print 'Closed-Virtual Separation: RHF orbitals'
	    cOrbs = self.mo_coeff[:,self.nfrozen:self.nelec/2]
            vOrbs = self.mo_coeff[:,self.nelec/2:]
            ierr,uc = pmloc.loc(self.mol,cOrbs)
            ierr,uv = pmloc.loc(self.mol,vOrbs)
	    clmo = numpy.dot(cOrbs,uc)
	    vlmo = numpy.dot(vOrbs,uv)
            lcoeff = numpy.hstack((clmo,vlmo))
         else:
            print 'Closed-Open-Virtual Separation: ROHF orbitals'
	    cOrbs = self.mo_coeff[:,self.nfrozen:self.nbeta]
	    oOrbs = self.mo_coeff[:,self.nbeta:self.nalpha]
            vOrbs = self.mo_coeff[:,self.nalpha:]
            ierr,uc = pmloc.loc(self.mol,cOrbs)
            ierr,uo = pmloc.loc(self.mol,oOrbs)
            ierr,uv = pmloc.loc(self.mol,vOrbs)
	    clmo = numpy.dot(cOrbs,uc)
	    olmo = numpy.dot(oOrbs,uo)
	    vlmo = numpy.dot(vOrbs,uv)
            lcoeff = numpy.hstack((clmo,olmo,vlmo))
	 # Check
         nbas = lcoeff.shape[1]
         ova = self.mol.intor_symmetric("cint1e_ovlp_sph")
	 diff = reduce(numpy.dot,(lcoeff.T,ova,lcoeff))\
	      - numpy.identity(nbas)
	 print '\nCheck orthnormality: diff(CtSC-I)=',numpy.linalg.norm(diff)
	 self.lmo_coeff = lcoeff.copy()
      return 0

   def molden(self,mo_coeff,fname='mocoeff'):
      print '\n[iface.molden] dump MOcoeff into file = '+fname+'.molden'   
      from pyscf.tools import molden
      with open(fname+'.molden','w') as thefile:
         molden.header(self.mol,thefile)
         molden.orbital_coeff(self.mol,thefile,mo_coeff,symm=['A']*mo_coeff.shape[1])
      return 0 	 

   def reorder(self,mo_coeff):
      print '\n[iface.reorder]'	   
      from pyscf import ao2mo
      c = mo_coeff
      k = c.shape[1]
      eritmp = ao2mo.outcore.general_iofree(self.mol,(c,c,c,c),compact=0)
      eritmp = eritmp.reshape(k,k,k,k)
      order = fielder.orbitalOrdering(eritmp,'kij')
      print ' order=',order
      return order

   # 
   # This is the central part
   # 
   def dump(self,fname='mole.h5'):
      # Effective
      nbas = self.nbas-self.nfrozen 
      sbas = nbas*2
      print '\n[iface.dump] (self.nbas,nbas)=',(self.nbas,nbas) 
      #
      # Basic information
      #
      f = h5py.File(fname, "w")
      cal = f.create_dataset("cal",(1,),dtype='i')
      enuc = self.mol.energy_nuc()
      nelec = self.nelec - self.nfrozen*2
      cal.attrs["nelec"] = nelec 
      cal.attrs["sbas"]  = sbas 
      cal.attrs["enuc"]  = enuc
      cal.attrs["escf"]  = 0. # Not useful at all: self.mf.energy_elec(self.mf.make_rdm1())[0]
      #
      # Intergrals
      #
      flter = 'lzf'
      # Local or CMO 
      if self.iflocal:
	 self.local()
	 mo_coeff = self.lmo_coeff.copy()
      else:
	 print 'canonical or user defined mo'
	 mo_coeff = self.mo_coeff[:,self.nfrozen:].copy()
      # Reorder
      if self.ifreorder:
         order = list(self.reorder(mo_coeff))
      else:
	 order = range(mo_coeff.shape[1])	    
      mo_coeff = mo_coeff[:,numpy.array(order)].copy()
      self.molden(mo_coeff,fname='mocoeff')
      # Dump MO coefficients
      f.create_dataset("mo_coeff_spatial", data=mo_coeff)
      # Occupation
      occun = numpy.zeros(sbas)
      for i in range(self.nalpha-self.nfrozen):
	 occun[2*i] = 1.0
      for i in range(self.nbeta-self.nfrozen):
	 occun[2*i+1] = 1.0
      print
      print 'initial occun for',len(occun),' spin orbitals:\n',occun
      sorder = numpy.array([[2*i,2*i+1] for i in order]).flatten()
      occun = occun[sorder].copy()
      assert abs(numpy.sum(occun)-nelec)<1.e-10
      print "sorder:",sorder
      print "occun :",occun
      # Symmetry
      if self.mol.symmetry and not self.iflocal:
         import pyscf.symm
         irrep_name = self.mol.irrep_id
         orbsym0 = pyscf.symm.label_orb_symm(self.mol,irrep_name,self.mol.symm_orb,mo_coeff)
         print '\nSpatial orbsym:\n',orbsym0
         orbsym = numpy.array([[i,i] for i in orbsym0]).flatten()
      else:
         orbsym = numpy.array([0]*sbas) 
      spinsym = numpy.array([[0,1] for i in range(nbas)]).flatten()
      print "orbsym :",orbsym
      print "spinsym:",spinsym
      f.create_dataset("occun",data=occun)
      f.create_dataset("orbsym",data=orbsym)
      f.create_dataset("spinsym",data=spinsym)
      #====================================================================
      # Spin orbital integrals
      #====================================================================
      # Integral transformation
      # SPECIAL FORM of MO for TRANS
      gmo_coeff = numpy.hstack((self.mo_coeff[:,:self.nfrozen],mo_coeff))
      print 'gmo_coeff.shape=',gmo_coeff.shape
      f.create_dataset("mo_coeff_spatialAll", data=gmo_coeff)
      kbas = gmo_coeff.shape[0]
      abas = gmo_coeff.shape[1]*2
      b = numpy.zeros((kbas,abas))
      b[:, ::2] = gmo_coeff.copy() 
      b[:,1::2] = gmo_coeff.copy()
      # INT1e:
      h = self.mf.get_hcore()
      hmo=reduce(numpy.dot,(b.T,h,b))
      hmo[::2,1::2]=hmo[1::2,::2]=0.
      # INT2e:
      from pyscf import ao2mo
      nb = abas 
      eri = ao2mo.general(self.mol,(b,b,b,b),compact=0).reshape(nb,nb,nb,nb)
      #eri = ao2mo.incore.general(self.mf._eri,(b,b,b,b),compact=0).reshape(nb,nb,nb,nb)
      eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
      # <ij|kl>=[ik|jl]
      eri = eri.transpose(0,2,1,3)
      # Antisymmetrize V[pqrs]=-1/2*<pq||rs> - In MPO construnction, only r<s part is used. 
      eri = -0.5*(eri-eri.transpose(0,1,3,2))
      #------------------ Frozen Core ------------------
      # fpq = hpq + <pr||qr>nr
      sfrozen = 2*self.nfrozen
      nr = numpy.zeros(abas)
      nr[:sfrozen] = 1.0 
      # Ecore = hcc + 1/2*<pq||pq>npnq
      ecore = numpy.einsum('pp,p',hmo,nr)\
	    - numpy.einsum('pqpq,p,q',eri,nr,nr)
      print 'E[core]=',ecore
      cal.attrs["ecor"]  = ecore
      fpq = hmo - 2.0*numpy.einsum('prqr,r->pq',eri,nr)
      orbrg = range(sfrozen,abas) 
      hmo = fpq[numpy.ix_(orbrg,orbrg)].copy()
      eri = eri[numpy.ix_(orbrg,orbrg,orbrg,orbrg)].copy()
      #------------------ Frozen Core ------------------
      # DUMP modified Integrals
      e1 = cal.attrs["escf"]
      ioccun = map(lambda x:x[0],numpy.argwhere(occun>1.e-4))
      #print 'ioccun=',ioccun
      e2 = numpy.einsum('ii',hmo[numpy.ix_(ioccun,ioccun)])\
         - numpy.einsum('ijij',eri[numpy.ix_(ioccun,ioccun,ioccun,ioccun)])
      ediff = abs(e1-e2-ecore)
      print 'E[pyscf]=',e1
      print 'E[elec] =',e2,' E[elec]+E[core]=',e2+ecore
      print 'E[diff] =',ediff
      #assert ediff<1.e-10 # can be different for casscf case
      int1e = f.create_dataset("int1e", data=hmo, compression=flter)
      int2e = f.create_dataset("int2e", data=eri, compression=flter)
      #====================================================================
      # Spatial integrals
      #====================================================================
      b = gmo_coeff
      # INT1e:
      h = self.mf.get_hcore()
      hmo = reduce(numpy.dot,(b.T,h,b))
      # INT2e:
      from pyscf import ao2mo
      nb = abas/2 
      eri = ao2mo.general(self.mol,(b,b,b,b),compact=0).reshape(nb,nb,nb,nb)
      # Frozen core case:
      # fpq = hpq + [pq|rr]*nr - 0.5*[pr|rq]*nr
      nr = numpy.zeros(nb)
      nr[:self.nfrozen] = 2.0 
      Fpq = hmo + numpy.einsum('pqrr,r->pq',eri,nr)\
		- 0.5*numpy.einsum('prrq,r->pq',eri,nr)
      #print numpy.linalg.norm(fpq[::2,::2] - Fpq)
      #print numpy.linalg.norm(fpq[1::2,1::2] - Fpq)
      #print numpy.linalg.norm(fpq[::2,1::2])
      #print numpy.linalg.norm(fpq[1::2,::2])
      orbrg = range(self.nfrozen,self.nbas) 
      hmo = Fpq[numpy.ix_(orbrg,orbrg)].copy()
      eri = eri[numpy.ix_(orbrg,orbrg,orbrg,orbrg)].copy()
      f.create_dataset("int1e_spatial", data=hmo, compression=flter)
      f.create_dataset("int2e_spatial", data=eri, compression=flter)
      #====================================================================
      f.close()
      print 'Successfully dump information for HS-DMRG calculations! fname=',fname
      self.check(fname)
      return 0

   def check(self,fname='mole.h5'):
      print '\n[iface.check]'	   
      f2 = h5py.File(fname, "r")
      print "nelec=",f2['cal'].attrs['nelec']
      print "sbas =",f2['cal'].attrs['sbas']
      print 'enuc =',f2['cal'].attrs['enuc']
      print 'ecor =',f2['cal'].attrs["ecor"]
      print 'escf =',f2['cal'].attrs['escf']
      print f2['int1e']
      print f2['int2e']
      print f2['int1e_spatial']
      print f2['int2e_spatial']
      f2.close()
      print "FINISH DUMPINFO into file =",fname
      return 0

#==================================================================
# Auxilliary functions 
#==================================================================
def lowdin(s):
   ''' new basis is |mu> c^{lowdin}_{mu i} '''
   e, v = numpy.linalg.eigh(s)
   return numpy.dot(v/numpy.sqrt(e), v.T.conj())
