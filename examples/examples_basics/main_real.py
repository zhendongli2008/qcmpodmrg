from mpi4py import MPI
from qcmpodmrg.source.molinfo import class_molinfo

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

from qcmpodmrg.source import mpo_dmrg_class
from qcmpodmrg.source import mpo_dmrg_schedule

sz = 1.0

dmrg = mpo_dmrg_class.mpo_dmrg()
dmrg.nsite = mol.sbas/2
dmrg.nhops = mol.sbas
dmrg.isym = 2
dmrg.build()
dmrg.comm = mol.comm
dmrg.qsectors = {str([mol.nelec,sz]):1} 
Dmax = 10
dmrg.Dmax = Dmax
#dmrg.test(mol)
sc = mpo_dmrg_schedule.schedule(maxM=Dmax,tol=1.e-7)
#sc.startM = 1000
sc.maxiter = 3
sc.mixed()
dmrg.ifplot=False
dmrg.partition()
dmrg.default(mol,sc)
dmrg.checkMPS(mol)
exit()





#lmps = [dmrg.lmps,dmrg.qnuml]
#dmrg2 = mpo_dmrg_class.mpo_dmrg()
#dmrg2.nsite = mol.sbas/2
#dmrg2.isym = 2
#dmrg2.build()
#dmrg2.comm = mol.comm
#dmrg2.qsectors = {str([mol.nelec,0.0]):1} 
#dmrg2.Dmax = Dmax
#sc = mpo_dmrg_schedule.schedule(maxM=Dmax,tol=1.e-7)
#sc.single(tol=1.e-7)
#sc.maxiter = 3
#dmrg2.default(mol,sc,lmps)

lmps = [dmrg.lmps,dmrg.qnuml]
dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.nsite = mol.sbas/2
dmrg2.nhops = mol.sbas
dmrg2.isym = 2
dmrg2.build()
dmrg2.comm = mol.comm
dmrg2.qsectors = {str([mol.nelec,sz]):1} 
#dmrg2.qsectors = {str([mol.nelec,0.0]):1} 
Dmax = 10
dmrg2.Dmax = Dmax
sc = mpo_dmrg_schedule.schedule(maxM=Dmax,tol=1.e-7)

#sc.single(tol=1.e-7)
sc.mixed()


dmrg2.ifs2proj = True
dmrg2.npts = 5
#dmrg2.quad = 'Simpson' 
dmrg2.s2quad(sval=1.,sz=sz)

mol.build()
dmrg2.partition()
dmrg2.default(mol,sc,lmps)
dmrg2.checkMPS(mol)
