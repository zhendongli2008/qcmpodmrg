import os 
import string
import numpy
import h5py
import time
from mpi4py import MPI
from qcmpodmrg.source import sysutil_io

#
# SCF & ERI interface
#
class class_molinfo:
   
   def __init__(self):	
      self.fname = None
      self.nelec = None 
      self.sbas  = None
      self.enuc  = None
      self.escf  = None
      self.ecor  = None
      # Array
      self.orboccun = None
      self.orbsymO3 = None
      self.orbsymSz = None
      # Settings
      self.comm  = None
      self.cutoff = 1.e-20
      # Storage path
      self.tmpdir = './'
      self.path   = None

   # Only rank0 loads Hamiltonian and information.
   def loadHam(self,fname="mole.h5"):
      status=0
      if self.comm.rank == 0:
	 try:     
            f = h5py.File(fname, "r")
	 except IOError:
            print "\nerror in molinfo.loadHam: No such file =",fname
	    status=1
      status = self.comm.bcast(status)
      if status == 1: exit(1)
      self.fname = fname
      if self.comm.rank == 0:
         self.nelec = f['cal'].attrs['nelec']
         self.sbas  = f['cal'].attrs['sbas']
         self.enuc  = f['cal'].attrs['enuc']
         self.escf  = f['cal'].attrs['escf']
         self.ecor  = f['cal'].attrs["ecor"]
      # Bcast
      self.nelec = self.comm.bcast(self.nelec) 
      self.sbas  = self.comm.bcast(self.sbas )
      self.enuc  = self.comm.bcast(self.enuc )
      self.escf  = self.comm.bcast(self.escf )
      self.ecor  = self.comm.bcast(self.ecor )
      # Empty	
      self.orboccun = numpy.empty(self.sbas,dtype=numpy.float_)
      self.orbsymO3 = numpy.empty(self.sbas,dtype=numpy.int)
      self.orbsymSz = numpy.empty(self.sbas,dtype=numpy.int)
      if self.comm.rank == 0:
         self.orboccun = f['occun'].value
	 self.orbsymO3 = f['orbsym'].value
         self.orbsymSz = f['spinsym'].value
      # Bcast with buffer array
      self.comm.Bcast( [self.orboccun,MPI.DOUBLE] ) 
      self.comm.Bcast( [self.orbsymO3,MPI.INT] ) 
      self.comm.Bcast( [self.orbsymSz,MPI.INT] )
      if self.comm.rank == 0: 
         f.close()	   
         print " loadHam sucessfully!"
      return 0

   # Define partition of operators
   def build(self):   
      self.comm.Barrier()
      print '[molinfo.build] rank=',self.comm.rank 
      #
      # tmpdir:
      # dirname = dateMar_29_13_07_20_2016_rank3_pid71559_mpo_dmrg
      #
      pid=str(os.getpid())
      dat=string.join(time.asctime().split(':'))
      dat=string.join(dat.split(' ')[1:],'_')
      suffix='date'+dat+'_rank'+str(self.comm.rank)+'_pid'+pid
      dirname = suffix+'_mpo_dmrg'
      self.path = self.tmpdir+dirname
      sysutil_io.createDIR(self.path)
      return 0

# TEST
if __name__ == '__main__':
   comm = MPI.COMM_WORLD
   mol = class_molinfo()
   mol.comm = comm
   mol.loadHam()
   print "\n === TEST of molinfo ==="
   print " mol.nelec = ",mol.nelec
   print " mol.sbas  = ",mol.sbas
   print " mol.enuc  = ",mol.enuc
   print " mol.escf  = ",mol.escf
   print " mol.ecor  = ",mol.ecor
