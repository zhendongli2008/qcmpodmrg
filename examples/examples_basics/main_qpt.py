import h5py
from mpi4py import MPI
from mpodmrg.source.itools.molinfo import class_molinfo
from mpodmrg.source.qtensor import qtensor_api

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

is2proj = 1
icase = 1
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
   dmrg.build()
   dmrg.comm = mol.comm
   dmrg.qsectors = {str([mol.nelec,sz]):1} 
   sc = mpo_dmrg_schedule.schedule()
   sc.fixed(maxM=maxM,maxiter=1)
   sc.prt()
   dmrg.ifIO = True
   dmrg.partition()
   dmrg.loadInts(mol)
   dmrg.dumpMPO()
   dmrg.default(sc)
   dmrg.checkMPS()
   
   #-------------------------------------------------------
   flmps0 = dmrg.flmps
   flmps1 = h5py.File(dmrg.path+'/lmpsQt','w')
   qtensor_api.fmpsQt(flmps0,flmps1,'L')
   flmps0.close()
   
   if sc.maxiter > 0:
      frmps0 = dmrg.frmps
      frmps1 = h5py.File(dmrg.path+'/rmpsQt','w')
      qtensor_api.fmpsQt(frmps0,frmps1,'R')
      frmps0.close()
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
   sc2 = mpo_dmrg_schedule.schedule()
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
   #---------------------------
   mol.build()
   dmrg2.path = mol.path
   dmrg2.ifQt = True # KEY
   dmrg2.ifIO = True
   dmrg2.partition()
   dmrg2.loadInts(mol)
   dmrg2.dumpMPO()
   dmrg2.default(sc2,flmps1)
   dmrg2.checkMPS()
   
   flmps1.close()
   if sc.maxiter != 0: 
      frmps1.close()
  
   # New L-MPS
   dmrg2.checkMPS()
   dmrg2.final()
  
   if rank == 0: 
      import shutil
      srcfile = dmrg2.path+'/lmps'
      dstdir = './lmpsQ'
      shutil.copy(srcfile, dstdir)
      print 'Energy',dmrg2.Energy

else:
   
   flmps = h5py.File('lmpsQ','r')
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

   dmrg2.ifH0 = 3
   dmrg2.iffq = False #True
   dmrg2.ifdiag = False #True
   sc2 = mpo_dmrg_schedule.schedule()
   sc2 = mpo_dmrg_schedule.schedule(tol=1.e-4)
   sc2.fixed(maxM=80,maxiter=6,ncsite=2,noise=0.)
   sc2.prt()
   
   dmrg2.ifQt = True
   dmrg2.maxslc = 1 #3
   dmrg2.wfex = [flmps]
   dmrg2.coef = [1.0]
   dmrg2.dumpMPO()
   dmrg2.default(sc2,flmps)
   dmrg2.checkMPS()
      
   dmrg2.wfex = [flmps,dmrg2.flmps]
   e,v = dmrg2.ci()
   
   flmps.close()
