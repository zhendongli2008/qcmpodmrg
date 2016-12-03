#
# Expt: http://webbook.nist.gov/cgi/cbook.cgi?Formula=o2&NoIon=on&Units=SI&cUV=on&cGC=on&cES=on&cDI=on
# 1. X3Sigma_g- (triplet ground state) [| |]
# 2. 1Delta_g: doubly degenerate (spatial) [|| 0]
# 3. b1Sigma_g+
#
# O2/STO-3G/R=1.2
#
#    15    67    67     1   T     -175.9643814399      0.313E-12      0.145E-08
#    15    67    67     2   T     -175.9256676059      0.853E-13      0.530E-08
#    15    67    67     3   T     -175.9256676059      0.284E-13      0.577E-08
#    15    67    67     4   T     -175.9048535909      0.227E-12      0.345E-08
#    15    67    67     5   T     -175.6559203682      0.227E-12      0.848E-08
#
# [FCI_s2val]
#
#  State =    1     2.000  Proj  =    1.000
#  State =    2    -0.000  Proj  =    1.000
#  State =    3    -0.000  Proj  =    1.000
#  State =    4     0.000  Proj  =    1.000
#  State =    5     2.000  Proj  =    1.000
#
#==================================================================
# Basically, SP-DMRG converges to A1 irrep.
#==================================================================
#
# =================================================================
#  Summary:   algorithm =  twoSite    status =  L
#  (Dmax,etol,noise) = (  100,  1.00e-05,  1.00e-04)
# -----------------------------------------------------------------
#   idx =    0  dwt= 0.00e+00  eigs= [-175.92488699 -175.90361683]
#   idx =    1  dwt= 1.11e-16  eigs= [-175.92493469 -175.90371136]
#   idx =    2  dwt= 0.00e+00  eigs= [-175.92517271 -175.90434645]
#   idx =    3  dwt= 7.53e-07  eigs= [-175.92517853 -175.90439331]
#   idx =    4  dwt= 1.54e-04  eigs= [-175.92543889 -175.9046202 ]
#------------------------------------------------------------------
#   idx =    5  dwt= 6.26e-04  eigs= [-175.92562799 -175.90480975]
#------------------------------------------------------------------
#   idx =    6  dwt= 1.63e-03  eigs= [-175.92553294 -175.90474546]
#   idx =    7  dwt= 4.03e-03  eigs= [-175.92489056 -175.90405612]
#   idx =    8  dwt= 1.51e-03  eigs= [-175.9248718  -175.90396539]
# -----------------------------------------------------------------
#  averaged energy [   5] =    -175.915218868604   dwts = 6.26e-04
#  time for sweep = 3.02e+02 s
#  settings: ifs2proj =  True
# =================================================================
# 
import h5py
from mpi4py import MPI
from mpodmrg.source.itools.molinfo import class_molinfo
from mpodmrg.source import mpo_dmrg_class
from mpodmrg.source import mpo_dmrg_schedule
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

##################
# Global settings
##################
#Triplet states
#sval = mol.nspin*1.0/2.0
#sz = mol.nspin*1.0/2.0
#Singlet states
sval = 0.
sz = 0.
nstate = 2

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
Dmax = 10
thresh = 1.e-10
sc = mpo_dmrg_schedule.schedule()
sc.fixed(maxM=Dmax,maxiter=0)
sc.prt()
dmrg.ifplot=False
dmrg.ifIO=True
dmrg.partition()
dmrg.loadInts(mol)
dmrg.default(sc)
dmrg.checkMPS()

#-------------------------------------------------------
# The last site only carries information from one state
flmps0 = h5py.File(dmrg.path+'/lmps','r')
flmps1 = h5py.File(dmrg.path+'/lmpsQt','w')
qtensor_api.fmpsQt(flmps0,flmps1,'L')
flmps0.close()

if sc.maxiter > 0:
   frmps0 = h5py.File(dmrg.path+'/rmps','r')
   frmps1 = h5py.File(dmrg.path+'/rmpsQt','w')
   qtensor_api.fmpsQt(frmps0,frmps1,'R')
   frmps0.close()

#dmrg.ifQt = True
#dmrg.ifIO = True
#dmrg.dumpMPO()
#dmrg.checkMPS(flmps1,status='L')
#dmrg.checkMPS(frmps1,status='R')
#-------------------------------------------------------

###############################
### 1. Using an MPS in Qt form
###############################
#dmrg2 = mpo_dmrg_class.mpo_dmrg()
#dmrg2.comm = mol.comm
#dmrg2.nsite = mol.sbas/2
#dmrg2.sbas  = mol.sbas
#dmrg2.isym = 2
#dmrg2.build()
#dmrg2.qsectors = {str([mol.nelec,sz]):1} 
#Dmax = 100
#thresh = 1.e-10
#sc = mpo_dmrg_schedule.schedule()
#sc.maxM = Dmax
#sc.maxiter = 1 
#sc.normal()
#sc.prt()
#mol.build()
#dmrg2.path = mol.path
#dmrg2.partition()
#dmrg2.ifQt = True
#dmrg2.ifIO = True
#dmrg2.ifguess = True 
#dmrg2.ifprecond = True
#dmrg2.ifplot=False
#dmrg2.maxslc = 1
#dmrg2.loadInts(mol)
#dmrg2.default(sc,flmps1)#oneSite=True)
#dmrg2.checkMPS()
#
#flmps1.close()
#flmps1 = h5py.File(dmrg2.path+'/lmps','r')
#dmrg2.checkMPS(flmps1,status='L')
#exit()

#####################
# 2. Spin Projection
#####################
dmrg2 = mpo_dmrg_class.mpo_dmrg()
dmrg2.nsite = mol.sbas/2
dmrg2.sbas  = mol.sbas
dmrg2.isym = 2
dmrg2.build()
dmrg2.comm = mol.comm
dmrg2.qsectors = {str([mol.nelec,sz]):nstate} 
Dmax = 100
thresh = 1.e-6
sc = mpo_dmrg_schedule.schedule(maxM=Dmax,tol=thresh)
sc.startM = Dmax
sc.maxiter = 1 
sc.normal()
sc.prt()
#---------------------------
dmrg2.ifs2proj = True
dmrg2.npts = 3
dmrg2.s2quad(sval,sz)
#---------------------------
dmrg2.partition()
dmrg2.loadInts(mol)
mol.build()
dmrg2.path = mol.path
dmrg2.ifQt = True
dmrg2.ifIO = True
dmrg2.ifguess = True #False
dmrg2.ifprecond = True #False
dmrg2.default(sc,flmps1)
dmrg2.checkMPS()
flmps1.close()
