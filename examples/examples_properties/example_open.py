import numpy
from pyscf import gto,scf

#==================================================================
# MOLECULE
#==================================================================
mol = gto.Mole()
mol.verbose = 5 #6

#==================================================================
# Coordinates and basis
#==================================================================
molname = 'h3' #o2' #be' #h2cluster'#h2o3' #c2'

if molname == 'h3':
   r = 0.7
   #a1 = numpy.array([math.sqrt(3)*r/2.0,0.,0.])
   a1 = numpy.array([0.,0,r*1.5])
   a2 = numpy.array([0,r,0.])
   a3 = numpy.array([0,-r,0.])
   mol.atom = [['H',a1],['H',a2],['H',a3]]
   mol.basis = '6-31g'

elif molname == 'cr':
   R = 1.0
   natoms = 2 
   mol.atom = [['Cr',(0.0,0.0,i*R)] for i in range(natoms)]
   mol.basis = '6-31g'

elif molname == 'h':
   R = 3.0
   natoms = 7 #40 #,14,20,50
   mol.atom = [['H',(0.0,0.0,i*R)] for i in range(natoms)]
   mol.basis = 'sto-3g' #cc-pvdz' #sto-3g' #6-31g' 

elif molname == 'o2':
   R=1.2
   mol.atom = [['O',(0,0,0)],
   	       ['O',(0,0,R)]]		
   mol.basis = 'sto-3g'

#==================================================================
mol.symmetry = False #True
mol.charge = -1
mol.spin = 0 #2
#==================================================================
mol.build()

#==================================================================
# SCF
#==================================================================
mf = scf.ROHF(mol)
mf.init_guess = 'atom'
mf.level_shift = 0.0
mf.max_cycle = 100
mf.conv_tol=1.e-14
#mf.irrep_nelec = {'B2':2}
print(mf.scf())

#==================================================================
# Dump integrals
#==================================================================
from mpodmrg.source.itools import ipyscf_real
iface = ipyscf_real.iface(mol,mf)
#iface.fci()
#iface.fci(nelecs=[8,8])
iface.fci(nelecs=[2,1])
#iface.fci(nelecs=[4,3])
#iface.ccsd()
#iface.molden(iface.mo_coeff,'CMO')
#print iface.reorder(iface.mo_coeff)
#iface.local()
#iface.molden(iface.lmo_coeff,'LMO')
#print iface.reorder(iface.lmo_coeff)

iface.iflocal = True #False
iface.iflowdin = True #False
iface.ifreorder = False
iface.ifdual = False
iface.param1 = 0.0 #1.0
iface.param2 = 0.0 #1.0
iface.spin   = 0.0 #0.0

iface.dump(fname='mole.h5')
