import h5py
from mpi4py import MPI
from qcmpodmrg.source.itools.molinfo import class_molinfo

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

mol.nelec = mol.sbas/2 

from qcmpodmrg.source import mpo_dmrg_class
from qcmpodmrg.source import mpo_dmrg_schedule

################################
# 0. Initialize an MPS(N,Sz) 
################################
dmrg = mpo_dmrg_class.mpo_dmrg()
#dmrg.occun = mol.orboccun
dmrg.path = mol.path
dmrg.nsite = mol.sbas/2
dmrg.sbas  = mol.sbas
dmrg.isym = 1
dmrg.build()
dmrg.comm = mol.comm
dmrg.qsectors = {str([mol.nelec]):1} 
sc = mpo_dmrg_schedule.schedule()
sc.fixed(maxM=10,maxiter=1)
sc.prt()
exit()
dmrg.partition()
dmrg.loadInts(mol)
dmrg.dumpMPO()
dmrg.default(sc)
dmrg.checkMPS()

#-------------------------------------------------------
flmps = dmrg.flmps #h5py.File(dmrg.path+'/lmps','r')
dmrg.checkMPS(flmps,status='L')

if sc.maxiter != 0:
   frmps = dmrg.frmps #h5py.File(dmrg.path+'/rmps','r')
   dmrg.checkMPS(frmps,status='R')
#-------------------------------------------------------

################################
# 1. Using an MPS in Qt form
################################
dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.nsite = mol.sbas/2
dmrg2.sbas  = mol.sbas
dmrg2.isym = 1
dmrg2.build()
dmrg2.comm = mol.comm
dmrg2.qsectors = {str([mol.nelec]):3} 
sc2 = mpo_dmrg_schedule.schedule()
sc2.maxM = 10
#sc2.maxiter = 2
sc2.normal()
sc2.prt()
mol.build()
dmrg2.path = mol.path
dmrg2.partition()
dmrg2.loadInts(mol)
dmrg2.dumpMPO()
dmrg2.default(sc2,flmps)#,oneSite=True)
dmrg2.checkMPS()

flmps.close()
if sc.maxiter != 0: 
   frmps.close()

flmps = h5py.File(dmrg2.path+'/lmps','r')
dmrg2.checkMPS(flmps,status='L')
flmps.close()
