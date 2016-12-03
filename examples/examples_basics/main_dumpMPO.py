import numpy
import math
import h5py
from mpi4py import MPI
from mpodmrg.source.itools.molinfo import class_molinfo

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
mol.isym =0 #2 #WhetherUseSym
mol.symSz=0 #1 #TargetSpin-2*Sz
mol.symS2=0.0 #Total Spin
# Tempory file will be put to this dir
mol.tmpdir = './'
mol.build()

from mpodmrg.source import mpo_dmrg_class
from mpodmrg.source import mpo_dmrg_schedule

sval = 1.0
sz = 1.0

icase = 0
is2proj = 0
maxM = 10

################################
# 0. Initialize an MPS(N,Sz) 
################################
dmrg = mpo_dmrg_class.mpo_dmrg()
dmrg.occun = mol.orboccun
dmrg.path = mol.path
dmrg.nsite = mol.sbas/2
dmrg.sbas  = mol.sbas
dmrg.isym = 2
dmrg.build()
dmrg.comm = mol.comm
dmrg.qsectors = {str([mol.nelec,sz]):1} 
dmrg.ifIO = True
dmrg.partition()
dmrg.loadInts(mol)
dmrg.dumpMPO()
#sc = mpo_dmrg_schedule.schedule()
#sc.fixed(maxM=maxM,maxiter=1)
#sc.prt()
#dmrg.default(sc)
#dmrg.checkMPS()
