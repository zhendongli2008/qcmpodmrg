#!/usr/bin/env python
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def __init__(self,maxM=1000,tol=1.e-7,maxiter=40):
# def getParameters(self,dmrg,isweep):
# def checkConv(self,dmrg,isweep,deltaE):
# def fixed(self,ncsite=2,maxM=None,tol=None,maxiter=None,noise=1.e-4):
# def normal(self,iop=0):
# def collect(self):
# def prt(self):
#
import math

# A clean way to define schedule.
class schedule():
   def __init__(self,maxM=1000,tol=1.e-7,maxiter=40):
      self.maxM = maxM
      self.tol = tol
      self.maxiter= maxiter
      #--- internal ---#
      self.startM = None
      self.startTol = 1.e-4  
      self.startNoise = 1.e-4
      self.coff   = 2
      self.change = None 
      self.Tag    = None
      self.Sweeps = []
      self.MaxMs  = []
      self.Tols   = []
      self.Noises = []

   # Setup 5 parameters: (ncsite,status,Dmax,crit_e,noise)
   # 			  used in DMRG sweeps. 
   def getParameters(self,dmrg,isweep):
      taglst = ['Normal0','Normal1','Normal2','Fixed1','Fixed2']
      if self.Tag not in taglst:
         print 'error: no such Tag in taglst! Tag=',self.Tag
	 exit(1)
      if self.Tag == 'Normal0':
	 if isweep < self.change:
	    idx = min(len(self.MaxMs)-1,isweep)
	    dmrg.ncsite = 2
	    dmrg.status = 'twoSite'
	    dmrg.Dmax   = self.MaxMs[idx]
	    dmrg.crit_e = self.Tols[idx]
	    dmrg.noise  = self.Noises[idx]
	 else:
	    dmrg.ncsite = 1
	    dmrg.status = 'oneSite'
	    dmrg.Dmax   = self.MaxMs[-1]
	    dmrg.crit_e = self.Tols[-1]
	    dmrg.noise  = self.Noises[-1]
	    # Reset psi0
	    if isweep == self.change: dmrg.psi0 = None
      elif self.Tag == 'Normal1':
	 dmrg.ncsite = 1
	 dmrg.status = 'oneSite'
	 idx = min(len(self.MaxMs)-1,isweep)
	 dmrg.Dmax   = self.MaxMs[idx]
	 dmrg.crit_e = self.Tols[idx]
	 dmrg.noise  = self.Noises[idx]
      elif self.Tag == 'Normal2':
	 dmrg.ncsite = 2
	 dmrg.status = 'twoSite'
	 idx = min(len(self.MaxMs)-1,isweep)
	 dmrg.Dmax   = self.MaxMs[idx]
	 dmrg.crit_e = self.Tols[idx]
	 dmrg.noise  = self.Noises[idx]
      elif self.Tag == 'Fixed1':
	 dmrg.ncsite = 1
	 dmrg.status = 'oneSite'
	 dmrg.Dmax   = self.MaxMs[-1]
	 dmrg.crit_e = self.Tols[-1]
	 dmrg.noise  = self.Noises[-1]
      elif self.Tag == 'Fixed2':
	 dmrg.ncsite = 2
	 dmrg.status = 'twoSite'
	 dmrg.Dmax   = self.MaxMs[-1]
	 dmrg.crit_e = self.Tols[-1]
	 dmrg.noise  = self.Noises[-1]
      return 0

   # Check Convergence
   def checkConv(self,dmrg,isweep,deltaE):
      taglst = ['Normal0','Normal1','Normal2','Fixed1','Fixed2']
      if self.Tag not in taglst:
         print 'error: no such Tag in taglst! Tag=',self.Tag
	 exit(1)
      ifconv = False	 
      if self.Tag in ['Normal0','Normal1','Normal2']:
	 if isweep > self.change and abs(deltaE) < self.tol: ifconv = True
      elif self.Tag in ['Fixed1','Fixed2']:
	 if abs(deltaE) < self.tol: ifconv = True
      return ifconv

   # Fixed (ncsite,maxM,tol,noise), used for testing & initialization!
   def fixed(self,ncsite=2,maxM=None,tol=None,maxiter=None,noise=1.e-4):
      self.Sweeps = []
      self.MaxMs  = []
      self.Tols   = []
      self.Noises = []
      if ncsite == 1:
	 self.Tag = 'Fixed1'
      elif ncsite == 2:
	 self.Tag = 'Fixed2'
      self.Sweeps = [0]
      # M
      if maxM is not None: self.maxM = maxM 
      self.MaxMs = [self.maxM]
      # Tol
      if tol is not None: self.tol = tol
      self.Tols = [self.tol]
      # maxiter
      if maxiter is not None: self.maxiter = maxiter
      # noise 
      self.Noises = [noise]
      # it does not matter 
      self.change = 0 
      return 0

   # Normal DMRG: twoSite -> oneSite follows 
   # schedule used in the BLOCK code.
   def normal(self,iop=0):
      self.Sweeps = []
      self.MaxMs  = []
      self.Tols   = []
      self.Noises = []
      if iop == 0:
	 self.Tag = 'Normal0'
      elif iop == 1:
	 self.Tag = 'Normal1'
      elif iop == 2:
	 self.Tag = 'Normal2'
      else:
	 raise NotImplementedError
      # Determine startM
      if self.startM is None:
         self.startM = min(max(int(1.5*math.sqrt(self.maxM)),25),self.maxM)
      startM = min(self.startM,self.maxM)
      N_sweep = 0
      Tol = self.startTol
      Noise = self.startNoise
      # Gradually increase M to maxM by
      # double M after each two sweeps
      while startM < self.maxM:
         self.Sweeps += [N_sweep]
	 self.MaxMs  += [startM]
         self.Tols   += [Tol]
         self.Noises += [Noise]
         N_sweep += 1
         startM *= 2
      # Gradually tighten the convergence 
      # by reduce Tol after each two sweeps
      iNoise = 0
      while Tol >= self.tol:
         self.Sweeps += [N_sweep]
         self.MaxMs  += [self.maxM]
         self.Tols   += [Tol]
         self.Noises += [Noise]
         N_sweep +=1
         Tol /=10.0
	 iNoise += 1
	 if iNoise == 1: Noise = 0.0
      self.change = N_sweep + self.coff
      return 0
   
   # Setup convergence
   def collect(self):
      if self.Tag is None: 
         print 'error: Tag still needs to be set!'
	 exit(1)
      self.maxM = self.MaxMs[-1]
      self.tol  = self.Tols[-1]
      self.change = self.Sweeps[-1]+2
      self.maxiter = max(self.change,40)
      return 0
      
   def prt(self):
      n = len(self.Sweeps)
      print ' DMRG sweep schedule:',\
      	    ' Tag = ',self.Tag,\
	    ' maxM =',self.maxM,\
      	    ' etol =',self.tol
      for i in range(min(n,self.maxiter)):
         print '  isweep =%4d   maxM = %5d   etol = %6.2e   noise = %6.2e'%\
	       (self.Sweeps[i],self.MaxMs[i],self.Tols[i],self.Noises[i])
      print ' change =%4d'%self.change,'  maxiter =',self.maxiter
      return 0

if __name__ == '__main__':
   sc = schedule(maxM=20000)
   #sc.startNoise = 0.
   sc.normal()
   sc.prt()
   
   print
   sc.fixed()
   sc.prt() 
   
   print
   sc.maxM = 10
   sc.maxiter = 1
   sc.tol = 1.e-1
   sc.fixed()
   sc.prt()
   
   print
   sc.fixed()
   sc.prt()
   
   print
   sc.Sweeps = [0,2,4,6,8]
   sc.MaxMs  = [10,20,30,50,100]
   sc.Tols   = [1.e-3,1.e-4,1.e-5,1.e-6,1.e-7]
   sc.Noises = [1.e-4,0.0,0.0,0.0,0.0]
   sc.Tag = 'Normal2'
   sc.collect()
   sc.prt()
