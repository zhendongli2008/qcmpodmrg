import h5py
from mpi4py import MPI
from qcmpodmrg.source.itools.molinfo import class_molinfo
from qcmpodmrg.source import mpo_dmrg_class
from qcmpodmrg.source import mpo_dmrg_schedule
from qcmpodmrg.source.qtensor import qtensor_api

#==================================
# Main program
#==================================
comm = MPI.COMM_WORLD
size = comm.size
rank = comm.rank
# MPI init
if size > 0 and rank ==0: print '\n[MPI init]'
comm.Barrier()
print ' Rank= %s of %s processes'%(rank,size)

mol=class_molinfo()
mol.comm=comm
fname = "mole.h5"
mol.loadHam(fname)
# Tempory file will be put to this dir
mol.tmpdir = './'
mol.build()

##################
# Global settings
##################
sval = 0.0
sz = 0.0

#############################
# 0. Initialize an MPS(N,Sz) 
#############################
dmrg = mpo_dmrg_class.mpo_dmrg()
dmrg.occun = mol.orboccun
dmrg.comm = mol.comm
dmrg.path = mol.path
dmrg.nsite = mol.sbas/2
dmrg.sbas = mol.sbas
dmrg.isym = 2
dmrg.build()
dmrg.qsectors = {str([mol.nelec,sz]):1} 
sc = mpo_dmrg_schedule.schedule()
sc.fixed(maxM=5,maxiter=0)
sc.prt()
dmrg.ifplot=False
dmrg.partition()
dmrg.loadInts(mol)
dmrg.dumpMPO()
dmrg.default(sc)
dmrg.checkMPS()

#-------------------------------------------------------
# The last site only carries information from one state
flmps0 = h5py.File(dmrg.path+'/lmps','r')
flmps1 = h5py.File(dmrg.path+'/lmpsQt','w')
qtensor_api.fmpsQt(flmps0,flmps1,'L')
flmps0.close()

if sc.maxiter > 0:
   frmps0 = h5py.File(dmrg.path+'/rmps','r')
   frmps1 = h5py.File(dmrg.path+'/rmpsQt','w')
   qtensor_api.fmpsQt(frmps0,frmps1,'R')
   frmps0.close()
#-------------------------------------------------------

###############################
### 1. Using an MPS in Qt form
###############################
dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.comm = mol.comm
dmrg2.nsite = mol.sbas/2
dmrg2.sbas  = mol.sbas
dmrg2.isym = 2
dmrg2.build()
dmrg2.qsectors = {str([mol.nelec,sz]):1} 
sc = mpo_dmrg_schedule.schedule()
sc.maxM = 5
sc.maxiter = 1 
sc.normal()
sc.prt()
mol.build()
dmrg2.path = mol.path
dmrg2.partition()
dmrg2.ifQt = True
dmrg2.ifguess = True 
dmrg2.ifprecond = True
dmrg2.maxslc = 1
dmrg2.loadInts(mol)
dmrg2.dumpMPO()
dmrg2.default(sc,flmps1)#oneSite=True)
dmrg2.checkMPS()
flmps1.close()

flmps1 = h5py.File(dmrg2.path+'/lmps','r')
dmrg2.checkMPS(flmps1,status='L')

#####################
# 2. Spin Projection
#####################
dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.nsite = mol.sbas/2
dmrg2.sbas  = mol.sbas
dmrg2.isym = 2
dmrg2.build()
dmrg2.comm = mol.comm
dmrg2.qsectors = {str([mol.nelec,sz]):1} 
sc = mpo_dmrg_schedule.schedule()
sc.startM = 10
sc.maxiter = 1 
sc.normal()
sc.prt()
#---------------------------
dmrg2.ifs2proj = True
dmrg2.npts = 3
dmrg2.s2quad(sval,sz)
#---------------------------
mol.build()
dmrg2.path = mol.path
dmrg2.ifQt = True
dmrg2.ifguess = True #False
dmrg2.ifprecond = True #False
dmrg2.partition()
dmrg2.loadInts(mol)
dmrg2.dumpMPO()
dmrg2.default(sc,flmps1)
dmrg2.checkMPS()
flmps1.close()
