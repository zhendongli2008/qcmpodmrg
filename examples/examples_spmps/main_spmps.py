#
# (H2)5
#
#    29    98    98     1   T       -9.5146270734      0.107E-13      0.444E-08
#    29    98    98     2   T       -9.1593501796      0.355E-14      0.687E-08
#    29    98    98     3   T       -9.1593501557      0.160E-13      0.789E-08
#    29    98    98     4   T       -9.1593493623      0.178E-13      0.566E-08
#    29    98    98     5   T       -8.7666537613      0.178E-14      0.419E-08
#    29    98    98     6   T       -8.7656959315      0.160E-13      0.858E-08
#
#  State =    1     0.000  Proj  =    1.000
#  State =    2     2.000  Proj  =    1.000
#  State =    3     2.000  Proj  =    1.000
#  State =    4     2.000  Proj  =    1.000
#  State =    5     0.000  Proj  =    1.000
#  State =    6     0.000  Proj  =    1.000
#
import h5py
import shutil
import numpy
from mpi4py import MPI
from qcmpodmrg.source.itools.molinfo import class_molinfo
from qcmpodmrg.source import mpo_dmrg_class
from qcmpodmrg.source import mpo_dmrg_schedule

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

sval = 1.0
sz = 0.0

################################
# 0. Initialize an MPS(N,Sz) 
################################
dmrg = mpo_dmrg_class.mpo_dmrg()
dmrg.occun = numpy.array([1.,0.,0.,1.]*5)
dmrg.path = mol.path
dmrg.nsite = mol.sbas/2
dmrg.sbas  = mol.sbas
dmrg.isym = 2
dmrg.build()
dmrg.comm = mol.comm
dmrg.qsectors = {str([mol.nelec,sz]):1} 
sc = mpo_dmrg_schedule.schedule()
sc.fixed(maxM=1,maxiter=0)
sc.prt()
dmrg.ifIO = True
dmrg.ifQt = False
dmrg.partition()
dmrg.loadInts(mol)
dmrg.dumpMPO()
dmrg.default(sc)
dmrg.checkMPS()
dmrg.final()

################################
# 1. Using an MPS in Qt form
################################
flmps1 = h5py.File(dmrg.path+'/lmps','r')
dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.nsite = mol.sbas/2
dmrg2.sbas  = mol.sbas
dmrg2.isym = 2
dmrg2.build()
dmrg2.comm = mol.comm
dmrg2.qsectors = {str([mol.nelec,sz]):1} 
sc2 = mpo_dmrg_schedule.schedule(tol=1.e-8)
#sc2.fixed(maxM=30,maxiter=6,noise=0.0,ncsite=2)
sc2.maxM = 30
sc2.normal()
sc2.maxiter = 6
sc2.prt()
#---------------------------
sc2.Tols = [10.*tol for tol in sc2.Tols] 
dmrg2.ifs2proj = True
dmrg2.npts = 3
dmrg2.s2quad(sval,sz)
#---------------------------
mol.build()
dmrg2.path = mol.path
dmrg2.ifIO = True
dmrg2.ifQt = False # KEY
dmrg2.ifplot = True
dmrg2.partition()
dmrg2.loadInts(mol)
dmrg2.dumpMPO()
dmrg2.default(sc2,flmps1)
dmrg2.checkMPS()
# New L-MPS
dmrg2.final()
flmps1.close()

if rank == 0:
   shutil.copy(dmrg2.path+'/lmps','./lmps'+str(int(sz)))
