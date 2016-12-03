#
# Analysis for integral dependent properties - energy component
#
import numpy
import h5py
from mpi4py import MPI
from qcmpodmrg.source.itools.molinfo import class_molinfo
from qcmpodmrg.source import mpo_dmrg_init
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
sval = 0.5
sz = 0.5
ifQt = True #False
ifs2proj = False
   
dmrg = mpo_dmrg_class.mpo_dmrg()
dmrg.nsite = mol.sbas/2
dmrg.sbas  = mol.sbas
dmrg.isym = 2
dmrg.build()
dmrg.comm = mol.comm
dmrg.qsectors = {str([mol.nelec,sz]):1} 
#---------------------------
if ifs2proj:
   dmrg.ifs2proj = True
   dmrg.npts = 3
   dmrg.s2quad(sval,sz)
#---------------------------
dmrg.partition()
dmrg.loadInts(mol)
mol.build()
dmrg.path = mol.path
dmrg.ifQt = ifQt
dmrg.dumpMPO()

###########
# Example #
###########
prefix = 'lmps_data_for_h3_631g/'
fnames = ['lmps_NQt_NProj','lmps_NQt_Proj','lmps_Qt_NProj','lmps_Qt_Proj']
fname = prefix + fnames[2]
flmps = h5py.File(fname,'r')
dmrg.checkMPS(flmps,ifep=True)
print dmrg.ecomp

dmrg.final()
flmps.close()
