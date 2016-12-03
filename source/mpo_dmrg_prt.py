#!/usr/bin/env python
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
# 
# def title(dmrg):
# def block(nsite,isite,ncsite,status):
# def parameters(Dmax,crit_e,noise,ncsite,sitelst,actlst,status):
# def singleSweep(dmrg,his,status,dt):
# def finalSweep(dmrg,dt):
# def eplot(dmrg,esweeps,suffix="png",emin=None,emax=None):
# def flogWrite(self,flog,isweep,result1,result2):
# 
import time
import numpy

def title(dmrg):
   n = 70
   print '='*n 
   print ' FS-DMRG calculations: '
   print ' Data =',time.asctime()
   print ' Path =',dmrg.path
   print ' Settings = {ifex:%i, s2proj:%i, Qt:%i, AllInts:%i, guess:%i, precond:%i}'%\
           (dmrg.ifex,dmrg.ifs2proj,dmrg.ifQt,\
            dmrg.ifAllInts,dmrg.ifguess,dmrg.ifprecond)
   print '-'*n
   dmrg.schedule.prt()
   print '='*n     
   return 0

# Print sweep configuration
def block(nsite,isite,ncsite,status):
   l=isite
   r=nsite-isite-ncsite
   if l<0 or r<0: 
      print 'error! l/r=',l,r
      exit(1)
   if status=='L':
      symbol1='='
      symbol2='-'
   elif status=='R':
      symbol1='-'
      symbol2='='
   elif status=='S':
      symbol1='='
      symbol2='S'
   # Banner   
   string=[symbol1]*l+['*']*ncsite+[symbol2]*r
   string=''.join(string)
   print
   print ' Site[i]:',isite,'/',nsite,\
	 ' status =',status,': [',string,']  active dot:',range(isite,isite+ncsite)
   return 0

def parameters(Dmax,crit_e,noise,ncsite,sitelst,actlst,status):
   n = 65
   print '='*n
   print ' Sweep parameters:   status  =',status,'   ncsite =',ncsite
   print '-'*n
   print ' sitelst =',sitelst
   print ' actlst  =',actlst
   print '-'*n
   print ' (Dmax,etol,noise) = (%5d,  %7.2e,  %7.2e)'%(Dmax,crit_e,noise)
   print '='*n
   return 0

def singleSweep(dmrg,his,status,dt):
   n = 65
   m = len(his[0])
   eigs = numpy.array(his[0])
   dwts = numpy.array(his[1])
   eavs = eigs.dot(dmrg.wts)
   indx = numpy.argmin(eavs)
   eav  = eavs[indx]
   dwt  = dwts[indx]
   if dmrg.comm.rank == 0:
      print 
      print '='*n
      print ' Summary:   algorithm = ',dmrg.status,'   status = ',status
      print ' (Dmax,etol,noise) = (%5d,  %7.2e,  %7.2e)'%(dmrg.Dmax,dmrg.crit_e,dmrg.noise)
      print '-'*n
      for i in range(m):
         eigi = eigs[i]
         dwti = dwts[i]
	 print '  idx = %4d'%i,' dwt= %6.2e'%dwti,' eigs=',eigi
      # Final
      print '-'*n
      print ' averaged energy [%4d] = %20.12f   dwts = %6.2e'%\
           (indx,eav,dwt)
      print ' time for sweep = %7.2e s'%dt
      print ' settings: ifs2proj = ',dmrg.ifs2proj
      print '='*n
      print
   return indx,eav,dwt

def finalSweep(dmrg,dt):
   n = 92
   print
   print '='*n
   print ' Finalize Sweep: ',dmrg.nsweep,\
         '   nsites = ',dmrg.nsite,\
         '   nMVPs = ',numpy.sum(dmrg.nmvp),\
         '   time = %7.2e s'%dt,\
	 '   comm.size = %3d'%dmrg.comm.size
   print '-'*n
   dmrg.schedule.prt()
   print '-'*n
   assert len(dmrg.eav) == dmrg.nsweep
   print ' DMRG sweep energy:'
   for isweep in range(dmrg.nsweep):
      if isweep == 0: 
         de = dmrg.eav[0]
      else:
         de = dmrg.eav[isweep]-dmrg.eav[isweep-1]
      # Possible change of schedule
      if dmrg.schedule.Tag == 'Normal0' and isweep == dmrg.schedule.change: 
	 print ' twoSite to oneSite:'
      print '  isweep =%4d  nmvp =%5d  eav[i] =%20.12f  dwt[i] = %6.2e  de = %7.1e'%\
              (isweep,dmrg.nmvp[isweep],dmrg.eav[isweep],dmrg.dwt[isweep],de)
   print '-'*n
   print ' SweepInfo: ifconv = ',dmrg.ifconv,' maxiter = ',dmrg.schedule.maxiter
   print '='*n
   if dmrg.ifplot:
      esweeps = numpy.array(dmrg.esweeps).T
      eplot(dmrg,esweeps)
   return 0

def eplot(dmrg,esweeps,suffix="png",emin=None,emax=None):
   neig,n = esweeps.shape
   import matplotlib.pyplot as plt
   for ieig in range(neig):
      plt.plot(range(n),esweeps[ieig],'-',marker='o',linewidth=2.0)
   if emin is None: emin = numpy.amin(esweeps)
   if emax is None: emax = numpy.amax(esweeps)
   #emax = emin+0.1
   shift = 0.01
   plt.xlim(-0.5,n)
   plt.ylim(emin-shift,emax+shift)
   fsize = 20
   plt.xlabel("step",fontsize=fsize)
   plt.ylabel("energy",fontsize=fsize)
   # Verticle lines
   # twoSite
   nopt2 = 2*(dmrg.nsite-1)
   for isweep in range(1,dmrg.schedule.change+1): 
      plt.axvline(x=isweep*nopt2,color='b')
   # oneSite
   ioff = nopt2*dmrg.schedule.change
   nopt1 = 2*dmrg.nsite
   for isweep in range(1,dmrg.nsweep-dmrg.schedule.change+1):
      plt.axvline(x=ioff+isweep*nopt1,color='r')
   plt.savefig("convergence."+suffix)
   plt.show()
   return 0

def flogWrite(dmrg,flog,isweep,result1,result2):
   nmvp1,indx1,eav1,dwt1,elst1,dlst1 = result1
   nmvp2,indx2,eav2,dwt2,elst2,dlst2 = result2
   flog.write('\nisweep='+str(isweep)+' ncsite='+str(dmrg.ncsite)+'\n')
   flog.write('***** status=R *****\n')
   flog.write('indx1='+str(indx1)+'\n')
   flog.write('eav1 ='+str(eav1)+'\n')
   flog.write('dwt1 ='+str(dwt1)+'\n')
   flog.write('elst1='+str(elst1)+'\n')
   flog.write('dlst1='+str(dlst1)+'\n')
   flog.write('***** sigsr *****\n')
   idx = dmrg.nsite - indx1 - 1
   flog.write('idxbd='+str(idx)+'\n')
   flog.write('sigsr='+str(dmrg.sigsr[idx])+'\n')
   # Right sweep:
   #        nc=2     nc=1
   #       ***|***  ***|***
   # indx  432 10   543 210
   # isite 012 345  012 345
   #       idx=2    idx=2
   #       6-2-1=3  6-2-1=3
   flog.write('idx='+str(idx)+'\n')
   for idx,sigsr in enumerate(dmrg.sigsr):
      flog.write('dwts='+str(1.-numpy.sum(sigsr))+'\n')
      flog.write('sigsr['+str(idx)+']='+str(sigsr)+'\n')
   flog.write('***** status=L *****\n')
   flog.write('indx2='+str(indx2)+'\n')
   flog.write('eav2 ='+str(eav2)+'\n')
   flog.write('dwt2 ='+str(dwt2)+'\n')
   flog.write('elst2='+str(elst2)+'\n')
   flog.write('dlst2='+str(dlst2)+'\n')
   flog.write('***** sigsl *****\n')
   idx = indx1 + 1
   flog.write('idx='+str(idx)+'\n')
   for idx,sigsl in enumerate(dmrg.sigsl):
      flog.write('dwts='+str(1.-numpy.sum(sigsl))+'\n')
      flog.write('sigsl['+str(idx)+']='+str(sigsl)+'\n')
   return 0
