import numpy
import h5py
from mpi4py import MPI
from qcmpodmrg.source.itools.molinfo import class_molinfo
from qcmpodmrg.source.qtensor import qtensor_api
from qcmpodmrg.source import mpo_dmrg_class
from qcmpodmrg.source import mpo_dmrg_schedule
import shutil
import os

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
#fname = "mole.h5"
#mol.loadHam(fname)

# Tempory file will be put to this dir
mol.tmpdir = './' 
mol.tmpdir = os.environ['SCRATCH']
#print 'tmpdir=',mol.tmpdir
mol.build()

import thubbard
t = 1.0
u = 2.
n = 4
nsite,tmatrix = thubbard.t2d(n,t)
us = numpy.ones(nsite)*u
nelec = 16
sval = 0.0
sz = 0.0

# initial occ
arow = [1.,0.,0.,1.]*2
brow = [0.,1.,1.,0.]*2

ifQt = False 
isym = 1
state_key = str([nelec])

icase = 1 # =0 initial guess; =1 run with larger D
if icase == 0:

   ################################
   # 0. Initialize an MPS(N,Sz) 
   ################################
   dmrg = mpo_dmrg_class.mpo_dmrg()
   dmrg.occun = numpy.array(arow+brow+arow+brow)
   dmrg.path = mol.path
   dmrg.nsite = nsite
   dmrg.sbas  = nsite*2
   dmrg.isym = 1
   dmrg.build()
   dmrg.comm = mol.comm
   dmrg.qsectors = {state_key:1}
   sc = mpo_dmrg_schedule.schedule()
   sc.fixed(maxM=1,maxiter=0)
   sc.prt()
   dmrg.ifQt = False
   dmrg.partition()
   #====================================
   dmrg.h1e = tmatrix
   dmrg.model_Usite = us
   dmrg.dumpMPO_Model("HubbardGeneral")
   #====================================
   dmrg.default(sc)
   dmrg.checkMPS()

   if rank == 0:
      #-------------------------------------------------------
      flmps0 = dmrg.flmps
      flmps1 = h5py.File(dmrg.path+'/lmpsQt','w')
      qtensor_api.fmpsQt(flmps0,flmps1,'L',isym=isym)
      flmps0.close()
      flmps1.close()
      if ifQt:
 	 shutil.copy(dmrg.path+'/lmpsQt','./lmpsQ0')
      else:
 	 shutil.copy(dmrg.path+'/lmps','./lmpsQ0')
      #-------------------------------------------------------

else:

   # 1. Using an MPS in Qt form
   flmps1 = h5py.File('./lmpsQ0','r')
   dmrg2 = mpo_dmrg_class.mpo_dmrg()
   dmrg2.nsite = nsite
   dmrg2.sbas  = nsite*2
   dmrg2.isym = 1
   dmrg2.build()
   dmrg2.comm = mol.comm
   dmrg2.qsectors = {state_key:1}
   
   np = 2
   sc2 = mpo_dmrg_schedule.schedule()
   sc2.MaxMs  = [1]*np + [500]*np + [1000]*np + [1500]*np + [2000]*(2*np)
      #        + [1600]*np + [2000]*np + [2400]*np + [2800]*np \
      #        + [3200]*np + [3600]*np + [4000]*np + [4400]*np \
      #        + [4800]*np + [5200]*np + [5600]*np + [6000]*np
   ns = len(sc2.MaxMs)
   sc2.Sweeps = range(ns)
   sc2.Tols   = [1.e-4]*(2*np) + [1.e-6]*(2*np) + [1.e-8]*(2*np) 
	      #+ [1.e-4]*(2*np) + [1.e-5]*(2*np) \
              #+ [1.e-6]*(8*np)
   sc2.Noises = [1.e-4]*(2*np) + [1.e-5]*(2*np) + [0.0]*(2*np) 
	      #+ [1.e-6]*(2*np) + [1.e-7]*(2*np) \
              #+ [0.e-0]*(8*np)
   sc2.coff = 0
   sc2.Tag = 'Normal2'
   sc2.collect()
   sc2.maxiter = ns
   sc2.prt()
   
   ##---------------------------
   #dmrg2.ifs2proj = True
   #dmrg2.npts = 4
   #dmrg2.s2quad(sval,sz)
   ##---------------------------
   dmrg2.path = mol.path
   dmrg2.ifQt = ifQt
   dmrg2.partition()
   #====================================
   dmrg2.h1e = tmatrix
   dmrg2.model_Usite = us
   dmrg2.dumpMPO_Model("HubbardGeneral")
   #debug: dmrg2.fhop.close()
   #debug: exit()
   #====================================
   dmrg2.default(sc2,flmps1)
   # New L-MPS
   dmrg2.final()
   flmps1.close()
   
   if rank == 0:
      shutil.copy( dmrg2.path+'/lmps','./lmpsQs1')
      print 'Energy',dmrg2.Energy
