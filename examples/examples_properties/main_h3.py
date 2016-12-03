import h5py
from mpi4py import MPI
from mpodmrg.source.itools.molinfo import class_molinfo
from mpodmrg.source import mpo_dmrg_class
from mpodmrg.source import mpo_dmrg_schedule
from mpodmrg.source.qtensor import qtensor_api
import shutil

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

##################
# Global settings
##################
#doublet states
sval = 0. #0.5
sz = 0. #0.5
ifQt = True #False
ifs2proj = False #True
prefix = './lmps_data_for_h3-_631g'

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
Dmax = 20
thresh = 1.e-10
sc = mpo_dmrg_schedule.schedule()
sc.fixed(maxM=Dmax,maxiter=0)
sc.prt()
dmrg.ifplot=False
dmrg.ifIO=True
dmrg.partition()
dmrg.loadInts(mol)
dmrg.dumpMPO()
dmrg.default(sc)
dmrg.checkMPS()
dmrg.final()

if ifQt:
   flmps0 = h5py.File(dmrg.path+'/lmps','r')
   flmps1 = h5py.File(dmrg.path+'/lmpsQt','w')
   qtensor_api.fmpsQt(flmps0,flmps1,'L')
   flmps0.close()
   if sc.maxiter > 0:
      frmps0 = h5py.File(dmrg.path+'/rmps','r')
      frmps1 = h5py.File(dmrg.path+'/rmpsQt','w')
      qtensor_api.fmpsQt(frmps0,frmps1,'R')
      frmps0.close()
else:
   flmps1 = h5py.File(dmrg.path+'/lmps','r')

# Spin Projection
dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.nsite = mol.sbas/2
dmrg2.sbas  = mol.sbas
dmrg2.isym = 2
dmrg2.build()
dmrg2.comm = mol.comm
dmrg2.qsectors = {str([mol.nelec,sz]):1} 
Dmax = 30
thresh = 1.e-6
sc = mpo_dmrg_schedule.schedule(maxM=Dmax,tol=thresh)
sc.startM = Dmax
sc.maxiter = 10 
sc.normal()
sc.prt()
#---------------------------
if ifs2proj:
   dmrg2.ifs2proj = True
   dmrg2.npts = 4
   dmrg2.s2quad(sval,sz)
#---------------------------
mol.build()
dmrg2.path = mol.path
dmrg2.ifQt = ifQt
dmrg2.partition()
dmrg2.loadInts(mol)
dmrg2.dumpMPO()
dmrg2.default(sc,flmps1)
dmrg2.checkMPS()
dmrg2.final()
flmps1.close()

if rank == 0: 
   if not ifQt:
      if not ifs2proj:
         fname = prefix+'/lmps_NQt_NProj'
      else:
         fname = prefix+'lmps_NQt_Proj'
   else:
      if not ifs2proj:
         fname = prefix+'/lmps_Qt_NProj'
      else:
         fname = prefix+'/lmps_Qt_Proj'
   shutil.copy(dmrg2.path+'/lmps', fname)
   print '\nRank =',rank,' Energy=',dmrg2.Energy,' fname=',fname
