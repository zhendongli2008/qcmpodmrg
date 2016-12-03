from mpi4py import MPI
from mpodmrg.source.molinfo import class_molinfo

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
from mpodmrg.source.mpsmpo import mps_io

#---------------------------------
# COPY MPS.h5 to each directory
#---------------------------------
import os
cmd = 'cp mps.h5 '+mol.path
os.system(cmd)
#---------------------------------

# LOAD MPS
fname = mol.path+'/mps.h5'
mps,qnum = mps_io.loadMPS(fname)
lmps = [mps,qnum]
bdim = map(lambda x:len(x),qnum)
if mol.comm.rank == 0: print ' bdim = ',bdim

# One site algorithm
sval = 0.0
sz   = 0.0
Dmax = max(bdim)

dmrg = mpo_dmrg_class.mpo_dmrg()
dmrg.nsite = mol.sbas/2
dmrg.nhops = mol.sbas
dmrg.isym = 2
dmrg.build()
dmrg.comm = mol.comm
dmrg.qsectors = {str([mol.nelec,sz]):1} 
dmrg.Dmax = Dmax
dmrg.enuc = mol.enuc

sc = mpo_dmrg_schedule.schedule(maxM=Dmax,tol=1.e-7)
sc.single(noise=0.0,tol=1.e-2)
#sc.mixed(noise=0.0,tol=1.e-2)
sc.maxiter = 10
sc.prt()

dmrg.ifs2proj = True
dmrg.npts = 5
dmrg.s2quad(sval,sz=sz)

dmrg.comm.Barrier()
dmrg.partition()
dmrg.comm.Barrier()
dmrg.Dcut = [4, 16, 48, 12, 3, 4, 1, 1, 1]
dmrg.default(mol,sc,lmps,oneSite=True)
dmrg.checkMPS(mol)
