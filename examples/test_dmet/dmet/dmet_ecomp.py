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
#fname = "mole.h5"
#mol.loadHam(fname)
# Tempory file will be put to this dir
mol.tmpdir = './'
mol.build()

# Specific for Hubbard Model
mol.sbas = 16*2

##################
# Global settings
##################
#sval = 0.5
#sz = 0.5
ifQt = False
ifs2proj = False
   
dmrg = mpo_dmrg_class.mpo_dmrg()
dmrg.nsite = mol.sbas/2
dmrg.sbas  = mol.sbas
dmrg.isym = 1
dmrg.build()
dmrg.comm = mol.comm
dmrg.qsectors = {str([mol.nelec]):1} 
####---------------------------
###if ifs2proj:
###   dmrg.ifs2proj = True
###   dmrg.npts = 3
###   dmrg.s2quad(sval,sz)
####---------------------------
dmrg.partition()
#dmrg.loadInts(mol)
#mol.build()
dmrg.path = mol.path
dmrg.ifQt = ifQt

#====================================
# Generate MPO
#====================================
import thubbard
t = 1.0
u = 2.
n = 4
nsite,tmatrix = thubbard.t2d(n,t)
us = numpy.ones(nsite)*u
dmrg.h1e = tmatrix
dmrg.model_Usite = us
dmrg.dumpMPO_Model("HubbardGeneral")
#====================================

###########
# Example #
###########
#prefix = 'lmps_data_for_h3_631g/'
#fnames = ['lmps_NQt_NProj','lmps_NQt_Proj','lmps_Qt_NProj','lmps_Qt_Proj']
#fname = prefix + fnames[2]
fname = './lmpsQs1'
flmps = h5py.File(fname,'r')

#==========================
# Compute energy component
#==========================
dmrg.checkMPS(flmps,ifep=True)
print
print len(dmrg.ecomp)
print dmrg.ecomp

#==========================================================
# Compute RDMs:
# For more general RDMs: 
# see generation of mpo for mpo_dmrg_propsMPO.py line 129.
#==========================================================
from qcmpodmrg.source.properties import mpo_dmrg_propsMPO
from qcmpodmrg.source.properties import mpo_dmrg_props
nsite = dmrg.nsite
rdm1 = numpy.zeros((nsite,nsite,2))
for p in range(nsite):
   for q in range(p+1):
      fname = 'top'
      fop = mpo_dmrg_propsMPO.genMPO_Epq(nsite,p,q,fname,dmrg.ifQt)
      exphop = mpo_dmrg_props.evalProps(dmrg,flmps,flmps,fop,status='L')
      fop.close()
      rdm1[p,q] = exphop
      rdm1[q,p] = exphop
      print 'p,q=',(p,q),'rdm1pq=',exphop
rdm1 = rdm1.transpose(2,0,1)
rdm1t = rdm1[0]+rdm1[1]
print 'tr<P>=',numpy.trace(rdm1t)
print rdm1t

dmrg.final()
flmps.close()
