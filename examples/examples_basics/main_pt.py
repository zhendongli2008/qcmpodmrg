#
# Test on diagonal approximation (error < 1% for E2):
#
# D0=10 new -9.39416809113
#
#
# D0=15 PT  = -0.160206223947
#	diag= -0.159170325112
# D0=40 PT  = -0.092036862721 
#	diag= -0.091888237440
#	diag= -0.091886879156 [projected vector]
#
# D0=40 new -9.65017167
#	PT  = -0.075680305931 [3] 
#
# H10 / R=2A:
# 
# sc2.fixed(maxM=30,maxiter=3,noise=1.e-4,ncsite=2)
# 
# --------------------------------------------------
# ifH0 = True: Hdiag -> Bad due to intruder states!
# --------------------------------------------------
# 
# Summary of MPS-CI:
#  HIJ=
# [[ -9.3883811   -0.42520284]
#  [ -0.42520284 -57.06078743]]
#  SIJ=
# [[  1.00000000e+00  -5.72579578e-08]
#  [ -5.72579578e-08   6.52678585e+00]]
#  CI energies= [-9.42874999 -8.7021866 ]
#  Coeff[0] = [-0.97182231 -0.092265  ]
# Li-Zhendongs-MacBook-Pro-2:examples zhendongli2008$ 
# 
# 
# -------------------------------
# ifH0 = False: Full Hamiltonian
# -------------------------------
# 
# Summary of MPS-CI:
#  HIJ=
# [[-9.3883811  -0.28846978]
#  [-0.28846978 -5.82620193]]
#  SIJ=
# [[  1.00000000e+00   2.90780677e-07]
#  [  2.90780677e-07   6.51301458e-01]]
#  CI energies= [-9.58741142 -8.74644598]
#  Coeff[0] = [-0.87368807 -0.60280872]
# 
# --------------------------------------------------------------------------------------------
#  DMRG sweep energy:
#   isweep =   0  nmvp =  133  eav[i] =     -0.228079971118  dwt[i] = 3.47e-03  de = -2.3e-01
#   isweep =   1  nmvp =  158  eav[i] =     -0.297262264204  dwt[i] = 4.98e-02  de = -6.9e-02
#   isweep =   2  nmvp =  188  eav[i] =     -0.320729631148  dwt[i] = 5.78e-02  de = -2.3e-02
# --------------------------------------------------------------------------------------------
# 
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

sval = 0.0
sz = 0.0

icase = 1
is2proj = 0
maxM = 10
if icase == 0:

   ################################
   # 0. Initialize an MPS(N,Sz) 
   ################################
   dmrg = mpo_dmrg_class.mpo_dmrg()
   dmrg.occun = mol.orboccun
   dmrg.path = mol.path
   dmrg.nsite = mol.sbas/2
   dmrg.sbas  = mol.sbas
   dmrg.isym = 2
   dmrg.initNsweep = 2 
   dmrg.build()
   dmrg.comm = mol.comm
   dmrg.qsectors = {str([mol.nelec,sz]):1} 
   sc = mpo_dmrg_schedule.schedule()
   sc.fixed(maxM=30,maxiter=0)
   sc.prt()
   dmrg.ifIO = True
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
   dmrg2.isym = 2
   dmrg2.build()
   dmrg2.comm = mol.comm
   dmrg2.qsectors = {str([mol.nelec,sz]):1} 
   sc2 = mpo_dmrg_schedule.schedule(tol=1.e-5)
   sc2.maxM = maxM
   sc2.coff = 2
   sc2.maxiter = 15 
   sc2.normal()
   sc2.prt()
   #---------------------------
   if is2proj != 0:
      sc2.Tols = [10.*tol for tol in sc2.Tols] 
      dmrg2.ifs2proj = True
      dmrg2.npts = 3
      dmrg2.s2quad(sval,sz)
   #--------------------------
   mol.build()
   dmrg2.path = mol.path
   dmrg2.ifIO = True
   dmrg2.partition()
   dmrg2.loadInts(mol)
   dmrg2.dumpMPO()
   dmrg2.default(sc2,flmps)
   dmrg2.checkMPS()
   dmrg2.final()

   flmps.close()
   if sc.maxiter != 0: 
      frmps.close()
 
   if rank == 0: 
      import shutil
      srcfile = dmrg2.path+'/lmps'
      dstdir = './lmps0'
      shutil.copy(srcfile, dstdir)
      print 'Energy',dmrg2.Energy

else:
  
   # It is fine to read lmps here,
   # since the mps will be copied.
   flmps = h5py.File('lmps0','r')
   dmrg2 = mpo_dmrg_class.mpo_dmrg()
   dmrg2.nsite = mol.sbas/2
   dmrg2.sbas  = mol.sbas
   dmrg2.isym = 2
   dmrg2.build()
   dmrg2.comm = mol.comm
   dmrg2.qsectors = {str([mol.nelec,sz]):1} 
   #---------------------------
   if is2proj != 0:
      dmrg2.ifs2proj = True
      dmrg2.npts = 3
      dmrg2.s2quad(sval,sz)
   #---------------------------
   mol.build()
   dmrg2.path = mol.path
   dmrg2.partition()
   dmrg2.loadInts(mol)
   dmrg2.ifpt = True
   
   f = h5py.File(mol.fname, "r")
   sint1e = f['int1e_spatial'].value
   sint2e = f['int2e_spatial'].value
   dmrg2.sint1e = sint1e.copy()
   dmrg2.sint2e = sint2e.copy()
   f.close()

   dmrg2.ifH0 = 4
   dmrg2.iffq = False #True
   dmrg2.ifdiag = False #True
   #dmrg2.ptlst = [6,7] #range(4,8) #None #range(16,20)
   sc2 = mpo_dmrg_schedule.schedule(tol=1.e-7)
   #sc2.startNoise = 0.
   #sc2.startM = 10
   #sc2.maxM = 10
   #sc2.maxiter = 10
   #sc2.normal()
   sc2.fixed(maxM=30,maxiter=8,ncsite=2,noise=0.)
   sc2.prt()

   dmrg2.wfex = [flmps]
   dmrg2.coef = [1.0]
   dmrg2.dumpMPO()
   dmrg2.default(sc2,flmps)
   dmrg2.checkMPS()
   
   dmrg2.wfex = [flmps,dmrg2.flmps]
   e,v =  dmrg2.ci()
   
   flmps.close()
