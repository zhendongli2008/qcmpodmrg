import os
import numpy
import math
import h5py
import shutil
from mpi4py import MPI
from qcmpodmrg.source.itools.molinfo import class_molinfo

try: 
  shutil.rmtree('./cal0')
  shutil.rmtree('./cal1')
except:
  pass

os.mkdir('./cal0')
os.mkdir('./cal1')

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

from qcmpodmrg.source import mpo_dmrg_class
from qcmpodmrg.source import mpo_dmrg_schedule

t = 1
u = 0.1
maxM = 10
nsite = 10
nelec = 10
sval = 0.0
sz = 0.0
nstate = 1

################################
# 0. Initialize an MPS(N,Sz) 
################################
dmrg = mpo_dmrg_class.mpo_dmrg()
dmrg.occun = None
dmrg.path = './cal0'
dmrg.nsite = nsite
dmrg.sbas  = 1 # for model =1 rather than 2*nsite
dmrg.isym = 2
dmrg.build()
dmrg.comm = comm
dmrg.qsectors = {str([nelec,sz]):nstate} 
sc = mpo_dmrg_schedule.schedule()
sc.fixed(maxM=maxM,maxiter=1)
sc.prt()
dmrg.ifIO = True
dmrg.partition()
dmrg.model_t = t
dmrg.model_u = u
dmrg.dumpMPO_Model('Hubbard')
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
dmrg2.path = './cal1'
dmrg2.nsite = dmrg.nsite
dmrg2.sbas  = 1 # for model =1 rather than 2*nsite
dmrg2.isym = 2
dmrg2.build()
dmrg2.comm = comm
dmrg2.qsectors = {str([nelec,sz]):nstate} 
sc2 = mpo_dmrg_schedule.schedule(tol=1.e-7)
sc2.maxM = maxM
sc2.coff = 2
sc2.maxiter = 15 
sc2.normal()
sc2.prt()
dmrg2.ifIO = True
dmrg2.partition()
dmrg2.model_t = t
dmrg2.model_u = u
dmrg2.dumpMPO_Model('Hubbard')
dmrg2.default(sc2,flmps)
dmrg2.checkMPS()
dmrg2.final()

flmps.close()
if sc.maxiter != 0: 
   frmps.close()
