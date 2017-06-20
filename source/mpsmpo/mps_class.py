import copy
import math
import cmath
import numpy
import scipy.linalg
from tools import misc
from tools import jwtrans
from tools import itools
from tools import detlib
from tools import tensorRep
from tools import tensorDecomp
from tools import qparser
from tools import smalld
from tools import mpslib
import mpo_class
import mpo_qphys

# Inverse S*(S+1)=val
def seff(expval):
   return (math.sqrt(1.+4.*expval)-1.)/2.

# A wrapper for mpslib
class class_mps:
   #
   # MPS: {site[l,nu,r]} - rank3
   #
   def __init__(self,k,isym=0,sites=None,iop=0,qphys=None,qnums=None):
      self.nsite=k
      if sites is None:
         self.sites=[0]*k
      elif iop == 0:
	 # rk2
	 self.sites = copy.deepcopy(sites)
	 mpslib.mps_mps2rank3(0,self.sites)
      elif iop == 1:
	 # rk3
	 self.sites = copy.deepcopy(sites)
      # Quantum numbers on physical dimensions
      self.qphys = None
      if qphys is not None: 
	 self.qphys = copy.deepcopy(qphys)
      elif isym != 0:
         self.qphys = mpo_qphys.init(k,isym) 
      #
      # ZL2016.02.05: Quantum numbers on bond dimensions
      #
      self.qnums = None
      if qnums is not None: self.qnums = copy.deepcopy(qnums)

   #--------------------------------
   # Hartree-Fock reference
   #--------------------------------
   def hfstate(self,n,occ=None):
      print '\n[mps_class.hfstate]'
      sites = [0]*self.nsite
      if occ==None:
         pocc = [0]*self.nsite 
         for i in range(n):
            pocc[i] = 1		 
      else:
	 pocc = copy.deepcopy(occ)
      # assign occupation	 
      for i in range(self.nsite):
         tmp = numpy.zeros((1,2,1))
	 # occupied
	 if pocc[i] == 1:
	    tmp[0,1,0] = 1.0
         else:
	    tmp[0,0,0] = 1.0
	 sites[i] = tmp.copy()
      # rank-3
      self.sites = copy.deepcopy(sites)
      #--------------------------------------------
      # ZL20160205: set up quantum numbers ?
      #		    all the bond dimensions are 1
      #		    for simplicity, only particle 
      #		    number is implemented.
      # Directed graphs: (0)->|->|->|->(N)
      #--------------------------------------------
      if self.qphys is not None:
	 self.qnums = [0]*self.nsite
         qnumber = numpy.array(self.qphys[0][0],dtype=numpy.float_)
         for ibond in range(self.nsite):
            iphys = pocc[ibond]
            qphys = self.qphys[ibond][iphys]
            qnumber += numpy.array(qphys)
            # Each qbond is a list of numpy arrays!
            self.qnums[ibond] = [qnumber.copy()]
      return 0

   def analyze(self,refdet):
      print '\n[mps_class.analyze]'
      print ' Analyze particle-holes contributions by projection, spin-orbital is assumed.'
      print ' refdet = ',refdet
      print ' bdims  = ',self.bdim() 
      #
      # The population of states with t[ph] 'quantum' number:  
      #   nt = <t|t> where |t>=P|0>
      #      = <0|P2|0> = <0|P|0> = (numerical integrations)
      # Possible values for no. of quasi-particles = [0,K]
      # Since the projector itself is Hermitian, nt is real.
      #
      # U(1) group projector: Pn = 1/2pi*int_{0,2pi} exp(i*(N-n)x) dx
      # discretized version:     = 1/2pi*sum_{k} exp(-in*xk)*<MPS|MPO(xk)|MPS>
      #
      global icounter
      icounter = 0
      # Complex algebra is mandatory in order to use the separability of exp(iNx)
      def fx(x,mps,refdet,n):
	 global icounter
	 icounter += 1
         nsite = len(refdet)
         sites1 = mps.torank2()
         sites2 = []
	 for isite in range(nsite):
	    if refdet[isite] == 0:
	       ntmp = numpy.array([[1.,0.],[0.,cmath.exp(1.j*x)]])
	    else:
	       ntmp = numpy.array([[cmath.exp(1.j*x),0.],[0.,1.]])
	    if isite == 0:
	       tmp = numpy.einsum('ij,ja->ia',ntmp,sites1[isite])
	    elif isite == nsite-1:
	       tmp = numpy.einsum('ij,aj->ai',ntmp,sites1[isite])
	    else:
	       tmp = numpy.einsum('ij,ajb->aib',ntmp,sites1[isite])
	    sites2.append(tmp)
         val = mpslib.mps_dot(sites1,sites2)
	 val = val*cmath.exp(-1.j*n*x)/(2.0*numpy.pi)
	 val = val.real
         return val
      # First normalize and compress to accelerate the analysis
      norm = self.norm()
      mps = self.copy()
      #mps.icompress()
      print ' norm0  = ',norm
      print ' bdims  = ',mps.bdim() 
      mps.mul(1.0/norm)
      norm2 = mps.norm()
      from scipy.integrate import quad
      psum = 0.0
      k = self.nsite
      nelec = sum(refdet)
      maxNoQuasiParticles = 2*(min(nelec,k-nelec)+1)
      for n in range(maxNoQuasiParticles):
        # for particle number conserving Hamiltonian,
	# only pair of p-h can apper
        if n%2 == 1: continue 
        # Roughly 100 points per each integral
        # https://en.wikipedia.org/wiki/QUADPACK
	# http://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.integrate.quad.html
	y,err = quad(fx,0,2*numpy.pi,args=(mps,refdet,n))#epsrel=1.e-1)
        print ' n = %3d'%n,' erank=',n/2,' y=',y,' err=',err,' icounter=',icounter
	psum += y
      print ' Total population =',psum
      return 0 

   def analyze_fast(self,refdet):
      print '\n[mps_class.analyze_fast]'
      print ' Analyze particle-holes contributions by projection, spin-orbital is assumed.'
      print ' refdet = ',refdet
      print ' bdims  = ',self.bdim() 
      #
      # The population of states with t[ph] 'quantum' number:  
      #   nt = <t|t> where |t>=P|0>
      #      = <0|P2|0> = <0|P|0> = (numerical integrations)
      # Possible values for no. of quasi-particles = [0,K]
      # Since the projector itself is Hermitian, nt is real.
      #
      # U(1) group projector: Pn = 1/2pi*int_{0,2pi} exp(i*(N-n)x) dx
      # discretized version:     = 1/2pi*sum_{k} exp(-in*xk)*<MPS|MPO(xk)|MPS>
      #
      global icounter
      icounter = 0
      # Complex algebra is mandatory in order to use the separability of exp(iNx)
      def fx(xdata,mps,refdet):
	 global icounter
	 icounter += 1
         nsite = len(refdet)
         sites1 = mps.torank2()
	 ydata = numpy.zeros(xdata.shape,dtype=numpy.complex_)
	 for i,x in enumerate(xdata):
            sites2 = []
	    for isite in range(nsite):
	       if refdet[isite] == 0:
	          ntmp = numpy.array([[1.,0.],[0.,cmath.exp(1.j*x)]])
	       else:
	          ntmp = numpy.array([[cmath.exp(1.j*x),0.],[0.,1.]])
	       if isite == 0:
	          tmp = numpy.einsum('ij,ja->ia',ntmp,sites1[isite])
	       elif isite == nsite-1:
	          tmp = numpy.einsum('ij,aj->ai',ntmp,sites1[isite])
	       else:
	          tmp = numpy.einsum('ij,ajb->aib',ntmp,sites1[isite])
	       sites2.append(tmp)
            ydata[i] = mpslib.mps_dot(sites1,sites2)/(2.0*numpy.pi)
         return ydata
      # First normalize and compress to accelerate the analysis
      norm = self.norm()
      mps = self.copy()
      #mps.icompress()
      print ' norm0  = ',norm
      print ' bdims  = ',mps.bdim() 
      mps.mul(1.0/norm)
      norm2 = mps.norm()
      k = self.nsite
      nelec = sum(refdet)
      maxNoQuasiParticles = 2*(min(nelec,k-nelec)+1)
      from scipy.integrate import trapz,simps
      # Trapezoidal rule
      npoints = 1000
      xdata = numpy.linspace(0,2*numpy.pi,num=npoints)
      ydata = fx(xdata,mps,refdet)
      psum = 0.0
      for n in range(maxNoQuasiParticles):
         if n%2 == 1: continue
	 ydata2 = map(lambda x:cmath.exp(-1.j*n*x),xdata)
	 ydata2 = (ydata2*ydata).real
	 y = simps(ydata2,xdata)
         print ' n = %3d'%n,' erank=',n/2,' y=',y,' icounter=',icounter
	 psum += y
      print ' Total population =',psum
      return 0 

   #--------------------------------
   # Two-orbital site
   #--------------------------------
   def quadfun(self,case,refdet=None):
      dic = {'n':0,\
	     'omega':1,\
	     'ph':2,\
	     'sz':3,\
	     's2':4}
      icase = dic[case]
      # particle number 
      if icase == 0:
         def rfun(phi):
	    expm = numpy.zeros((4,4),dtype=numpy.complex_)
	    expm[0,0] = 1.0
	    expm[1,1] = cmath.exp(1.j*phi)
	    expm[2,2] = cmath.exp(1.j*phi)
	    expm[3,3] = cmath.exp(2.j*phi)
	    return expm
         def wfun(phi,qnum):
	    wt = cmath.exp(-1.j*qnum*phi)/(2.*numpy.pi)
 	    return wt
      # singly occupancy
      elif icase == 1:
         def rfun(phi):
	    expm = numpy.zeros((4,4),dtype=numpy.complex_)
	    expm[0,0] = 1.0
	    expm[1,1] = cmath.exp(1.j*phi)
	    expm[2,2] = cmath.exp(1.j*phi)
	    expm[3,3] = 1.0
	    return expm
         def wfun(phi,qnum):
	    wt = cmath.exp(-1.j*qnum*phi)/(2.*numpy.pi)
 	    return wt
      # particle-hole excitations
      elif icase == 2:
         def rfun(phi,iocc):
	    # Exp(x*(n1+n2))=Exp(x*n1)\otimesExp(x*n2)
 	    occ = numpy.zeros((2,2,2),dtype=numpy.complex_)
	    occ[0] = numpy.array([[1.,0.],[0.,cmath.exp(1.j*phi)]])
	    occ[1] = numpy.array([[cmath.exp(1.j*phi),0.],[0.,1.]])
	    expm = numpy.kron(occ[iocc[0]],occ[iocc[1]])
	    return expm
         def wfun(phi,qnum):
	    wt = cmath.exp(-1.j*qnum*phi)/(2.*numpy.pi)
 	    return wt
      # [i*alpha_k*Sz] 
      elif icase == 3:
         def rfun(phi):
	    expm = numpy.zeros((4,4),dtype=numpy.complex_)
	    expm[0,0] = 1.0
	    expm[1,1] = cmath.exp(-0.5j*phi)
	    expm[2,2] = cmath.exp(0.5j*phi)
	    expm[3,3] = 1.0
	    return expm
         def wfun(phi,qnum):
	    wt = cmath.exp(-1.j*qnum*phi)/(2.*numpy.pi)
 	    return wt
      # exp(-i*beta_k*Sy) -> Percus-Rotenberg (1962) 
      elif icase == 4:
         def rfun(phi):
	    expm = numpy.zeros((4,4),dtype=numpy.float_)
	    expm[0,0] = 1.0
	    #
	    # [-i]*[sigma_y] = [-i]*[[0,-i],  = [[0,-1],
	    #		  	     [i, 0]]     [1, 0]]
	    #
	    # MPO site:
	    # 0
	    # b [ cos(b/2), sin(b/2)]
	    # a [-sin(b/2), cos(b/2)]
	    # 2 
	    c = math.cos(0.5*phi)
	    s = math.sin(0.5*phi)
	    expm[1,1] = c 
	    expm[1,2] = s 
	    expm[2,1] = -s 
	    expm[2,2] = c 
	    expm[3,3] = 1.0
	    return expm
         def wfun(phi,qnum,sz):
	    wt = (2.*qnum+1.)/2.0*math.sin(phi)*smalld.value(qnum,sz,sz,phi)
 	    return wt
      return rfun,wfun

   def analyzer(self,case,refdet=None,npoints=1000,args=None):
      print '\n[mps_class.analyzer] case =',case
      
      # Complex algebra is mandatory in order to use the separability of exp(iNx)
      def fx(xdata,mps,rfun,args=None,dtype=numpy.complex_):
         nsite = mps.nsite
	 sites1 = mps.torank2()
         ydata = numpy.zeros(xdata.shape,dtype=dtype)
         for i,x in enumerate(xdata):
            sites2 = []
            for isite in range(nsite):
	       if args is None:	    
                  ntmp = rfun(x)
	       else:
		  occ = [args[2*isite],args[2*isite+1]] 
                  ntmp = rfun(x,occ)
               if isite == 0:
                  #tmp = numpy.einsum('ij,ja->ia',ntmp,sites1[isite])
		  tmp = numpy.dot(ntmp,sites1[isite])
               elif isite == nsite-1:
                  #tmp = numpy.einsum('ij,aj->ai',ntmp,sites1[isite])
		  tmp = numpy.dot(sites1[isite],ntmp.T)
               else:
                  #tmp = numpy.einsum('ij,ajb->aib',ntmp,sites1[isite])
		  # ajb,ij->abi
		  tmp = numpy.tensordot(sites1[isite],ntmp,axes=([1],[1]))
		  # abi->aib
		  tmp = tmp.transpose(0,2,1).copy() 
               sites2.append(tmp)
            ydata[i] = mpslib.mps_dot(sites1,sites2)
         return ydata

      # First normalize and compress to accelerate the analysis
      k = self.nsite
      norm = self.norm()
      mps = self.copy()
      print ' ksite =',k
      print ' norm0 =',norm
      print ' bdims =',mps.bdim() 
      mps.mul(1.0/norm)
      rfun,wfun = self.quadfun(case,refdet)
      from scipy.integrate import trapz,simps
      psum = 0.0
      expval = 0.0
      # Partilce number
      if case == 'n':
         b = 2*numpy.pi
	 xdata = numpy.linspace(0,b,num=npoints)
         ydata = fx(xdata,mps,rfun)
	 for n in range(2*k+1):
            ydata2 = map(lambda x:wfun(x,n),xdata)
            ydata2 = (ydata2*ydata).real
            y = simps(ydata2,xdata)
            psum += y
	    expval += y*n
	    print ' n =%3d'%n,' p[n] =%10.5f'%y
 
      elif case == 'sz':
         b = 2*numpy.pi
	 xdata = numpy.linspace(0,b,num=npoints)
         ydata = fx(xdata,mps,rfun)
	 for sz in numpy.arange(-0.5*k,0.5*k+0.1,0.5):
            ydata2 = map(lambda x:wfun(x,sz),xdata)
            ydata2 = (ydata2*ydata).real
            y = simps(ydata2,xdata)
            psum += y
	    expval += y*sz
	    print ' sz =%5.1f'%sz,' p[n] =%10.5f'%y
      
      elif case == 'omega': 
         b = 2*numpy.pi
	 xdata = numpy.linspace(0,b,num=npoints)
         ydata = fx(xdata,mps,rfun)
	 # No. of unpair electrons
	 for omega in range(0,k+1):
            ydata2 = map(lambda x:wfun(x,omega),xdata)
            ydata2 = (ydata2*ydata).real
            y = simps(ydata2,xdata)
            psum += y
	    expval += y*omega
	    # omega is even if n is even.
	    # omega is odd if n is odd.
	    print ' omega =%3d'%omega,' p[n] =%10.5f'%y
 
      elif case == 'ph':
	 if refdet is None:
	    print 'error: there must be a reference determinant'
	    exit(1)
         nelec = sum(refdet)
	 print ' refdet =',refdet 
	 print ' nelec of refdet =',nelec
         maxNoQuasiParticles = 2*(min(nelec,2*k-nelec))
         b = 2*numpy.pi
	 xdata = numpy.linspace(0,b,num=npoints)
         ydata = fx(xdata,mps,rfun,args=refdet)
	 # No. of quasiparticles
	 for nQ in range(maxNoQuasiParticles+1):
            ydata2 = map(lambda x:wfun(x,nQ),xdata)
            ydata2 = (ydata2*ydata).real
            y = simps(ydata2,xdata)
            psum += y
	    expval += y*nQ
	    print ' nQ =%3d'%nQ,' p[n] =%10.5f'%y
 
      elif case == 's2':
         if args is None:
 	    print 'error: a Sz value of the reference must be provide!'
	    exit(1)
         b = numpy.pi
	 xdata = numpy.linspace(0,b,num=npoints)
         ydata = fx(xdata,mps,rfun)
	 for stot in numpy.arange(0,0.5*k+0.1,0.5):
            ydata2 = map(lambda x:wfun(x,stot,args),xdata)
            ydata2 = (ydata2*ydata).real
            y = simps(ydata2,xdata)
            psum += y
	    expval += y*stot*(stot+1)
	    print ' stot =%5.1f'%stot,' p[n] =',y
         print ' S[eff] =',seff(expval)
      
      # Final print
      print ' Total population =',psum
      print ' <O> population =',expval
      return psum,expval
   
   #--------------------------------
   # Basic subroutines 
   #--------------------------------
   def copy(self):
      tmps = class_mps(self.nsite)
      tmps.sites = copy.deepcopy(self.sites)
      if self.qnums is not None:
	 tmps.qnums = copy.deepcopy(self.qnums)
      if self.qphys is not None:
	 tmps.qphys = copy.deepcopy(self.qphys)
      return tmps

   def torank2(self):
      tmp = copy.deepcopy(self.sites)
      mpslib.mps_mps2rank3(1,tmp)
      return tmp

   def bdim(self):
      tmp = self.torank2()
      bdim = mpslib.mps_bdim(tmp)
      return bdim

   def pdim(self):
      tmp = self.torank2()
      pdim = mpslib.mps_pdim(tmp)
      return pdim

   def prt(self,ifqnums=False):
      print "\nMPSinfo:"
      print " nsite = ",self.nsite	  
      for i in range(self.nsite):
         print " Site : ",i," Shape : ",self.sites[i].shape,\
	       " Val = ",numpy.linalg.norm(self.sites[i])
      if self.qphys is not None:
	 mpo_qphys.prt(self.qphys)
      if self.qnums is not None and ifqnums:
	 self.prtQnums()
      print "End of MPSinfo\n"
      return 0
   
   def prtQnums(self):
      print "[mps_class.prtQnums]" 
      print '-'*60
      print " quantum numbers on bonds:"
      for i in range(self.nsite):
	 print " Bond : ",i," bdim[i] =",len(self.qnums[i])
	 print " Qnums[i] =",self.qnums[i]
      print '-'*60
      return 0

   def prtBdim(self):
      print 'Bdim =',self.bdim()
      return 0

   def toMat(self):
      N = self.nsite
      for i in range(N):
	 if i == 0:
	    t1 = self.sites[0]
         else:
            t2 = self.sites[i]
	    # t1[l,r,nu,nd]*t2[l,r,nu',nd']
	    tmp = numpy.einsum('aib,bkc->aikc',t1,t2)
	    s = tmp.shape
	    t1 = tmp.reshape((s[0],s[1]*s[2],s[3]))
      s = tmp.shape
      assert s[0]==1 and s[2]==1
      t1 = t1.reshape(s[1])
      return t1

   #--------------------------------
   # Compression
   #--------------------------------
   def compress(self,thresh=1.e-12,iprt=2):
      t1 = self.torank2()
      t2 = copy.deepcopy(t1)
      mpslib.mps_compress(t2,thresh=thresh,iprt=iprt)
      rr = mpslib.mps_diff(t1,t2,iprt=0)
      if rr > 1.e-5:
	 print 'error: compression error is too large! rr=',rr
	 exit()
      mpslib.mps_mps2rank3(0,t2)
      self.sites = copy.deepcopy(t2)
      return 0

   def icompress(self,thresh=1.e-12,iprt=2):
      t1 = self.torank2()
      mpslib.mps_icompress(t1,thresh=thresh,iprt=iprt)
      mpslib.mps_mps2rank3(0,t1)
      self.sites = copy.deepcopy(t1)
      return 0

   def checkOverlap(self,mps):
      print '\n[mps_class.checkOverlap]'
      print ' norm0',self.norm()
      print ' norm1',mps.norm()
      print ' <0|1>',self.dot(mps)
      return 0

   def qcompress(self,thresh=1.e-12,Dcut=-1,debug=False,qphys=None):
      "Compress with Quantum Numbers: similar to compressions in HS-MPS,\
       performed using qleftCanon and qrightCanon without discarding."
      print '\n[mps_class.qcompress]'
      # Normalize such that sum of Schmidt values = 1.
      self.normalize()
      # Preparation step: 
      mps = self.copy()
      if qphys is not None:
	 mps.qphys = copy.deepcopy(qphys)
      if mps.qphys is None:
	 print 'error: qphys cannot be NONE!'
	 exit(1)
      # Check physical dimensions
      dphys1 = mps.pdim()
      dphys2 = map(lambda x:len(x),mps.qphys)
      if dphys1 != dphys2:
	 print 'error: physical dimensions must be consistent!'
	 print 'mps.pdim() =',dphys1
	 print 'mps.qphys  =',dphys2
	 exit(1)
      # M*M*M => L*L*L such that the latter Schmidt values = 1.
      mps.qleftCanon(debug=debug)
      # Compression step:
      # L*L*L => R*R*R
      mps.qrightCanon(thresh,Dcut,debug)
      # R*R*R => L*L*L
      mps.qleftCanon(thresh,Dcut,debug)
      self.checkOverlap(mps)
      return mps

   def dcompress(self,thresh=1.e-12,Dcut=-1,debug=False):
      "SVD compression"
      print '\n[mps_class.dcompress]'
      self.normalize()
      mps = self.copy()
      mps.leftCanon(debug=debug)
      mps.rightCanon(thresh,Dcut,debug)
      mps.leftCanon(thresh,Dcut,debug)
      self.checkOverlap(mps)
      return mps

   #--------------------------------
   # Canonical forms
   #--------------------------------
   # Left Canonical MPS
   def leftCanon(self,thresh=1.e-12,Dcut=-1,debug=False):
      if debug:
         print '[mps_class.leftCanon]'
	 print ' Bdim before: ',self.bdim()
      try:
	 assert abs(self.norm()-1.0)<1.e-8
      except AssertionError:
	 print ' norm = ',self.norm()
	 exit()
      sites = self.sites
      nsite = len(sites)
      tsite = sites[0].copy()
      link = [0]*(nsite-1)
      sval = [0]*(nsite-1)
      lmps = [0]*nsite
      # rk3
      for isite in range(nsite-1):
         s = tsite.shape
	 d1 = s[0]*s[1]
	 d2 = s[2]
         mat = tsite.reshape((d1,d2)).copy() 
	 u,sigs,vt = mpslib.mps_svd_cut(mat,thresh,Dcut)
	 bdim = len(sigs)
	 if debug:
	    print '-'*80	 
	    print ' Results[i]:',isite
	    print '-'*80	 
	    print ' dimension:',(d1,d2),'->',bdim
	    sum2 = numpy.sum(numpy.array(sigs)**2)
	    dwts = 1.0-sum2
	    print ' sum of sigs2:',sum2,' dwts:',dwts
	    print ' sigs:\n',sigs
	 sval[isite] = sigs.copy()
	 lmps[isite] = u.reshape((s[0],s[1],bdim)).copy()
	 tmp = numpy.diag(sigs).dot(vt)
	 link[isite] = tmp.copy()
	 #tsite = numpy.einsum('sl,lur->sur',tmp,sites[isite+1])
	 tsite = numpy.tensordot(tmp,sites[isite+1],axes=([1],[0]))
      norm = numpy.linalg.norm(tsite)
      lmps[nsite-1] = tsite/norm
      # Final
      self.sites = copy.deepcopy(lmps)
      if debug: self.prt()
      return link,sval

   # Right Canonical MPS
   def rightCanon(self,thresh=1.e-12,Dcut=-1,debug=False):
      if debug:
         print '[mps_class.rightCanon]'
	 print ' Bdim before: ',self.bdim()
      try:
	 assert abs(self.norm()-1.0)<1.e-8
      except AssertionError:
	 print ' norm = ',self.norm()
	 exit()
      sites = self.sites
      nsite = len(sites)
      tsite = sites[nsite-1].copy()
      link = [0]*(nsite-1)
      sval = [0]*(nsite-1)
      rmps = [0]*nsite
      # rk3
      for isite in range(nsite-1,0,-1):
         s = tsite.shape
	 d1 = s[0]
	 d2 = s[1]*s[2]
         mat = tsite.reshape((d1,d2)).copy()
	 # C=U s Vd => Ct=V* s Ut 
         u,sigs,vt = mpslib.mps_svd_cut(mat.T,thresh,Dcut)
	 bdim = len(sigs)
	 if debug:
	    print '-'*80	 
	    print ' Results[i]:',isite
	    print '-'*80	 
	    print ' dimension:',(d1,d2),'->',bdim
	    sum2 = numpy.sum(numpy.array(sigs)**2)
	    dwts = 1.0-sum2
	    print ' sum of sigs2:',sum2,' dwts:',dwts
	    print ' sigs:\n',sigs
	 sval[isite-1] = sigs.copy()
	 # u = V* ---> (Vd)[alpha,n*r]
	 rmps[isite] = u.T.reshape((bdim,s[1],s[2])).copy()
	 tmp = numpy.diag(sigs).dot(vt).T.copy()
	 link[isite-1] = tmp.copy()
	 #tsite = numpy.einsum('lur,rs->lus',sites[isite-1],tmp)
	 tsite = numpy.tensordot(sites[isite-1],tmp,axes=([2],[0]))
      norm = numpy.linalg.norm(tsite)
      rmps[0] = tsite/norm
      # Final
      self.sites = copy.deepcopy(rmps)
      if debug: self.prt()
      return link,sval

   #----------------------------------------------------
   # Left Canonical MPS compressed with quantum numbers
   #----------------------------------------------------
   def qleftCanon(self,thresh=1.e-12,Dcut=-1,debug=False):
      print '\n[mps_class.qleftCanon]'
      print ' Bdim before: ',self.bdim()
      assert abs(self.norm()-1.0)<1.e-8
      sites = self.sites
      nsite = len(sites)
      link = [0]*(nsite-1)
      sval = [0]*(nsite-1)
      lmps = [0]*nsite
      qnums = [0]*nsite
      # Current object
      tsite = sites[0].copy()
      # Now the symmetry is purely determined from qphys
      qnuml = [copy.deepcopy(self.qphys[0][0])]
      qphys = copy.deepcopy(self.qphys)
      # rk3
      for isite in range(nsite): 
         s = tsite.shape
	 d1 = s[0]*s[1]
	 d2 = s[2]
	 # Combine quantum numbers
	 qphys_isite = qphys[isite]
	 qnum1 = mpo_qphys.dpt(qnuml,qphys_isite)
         mat = tsite.reshape((d1,d2)).copy() 
         classes = qnum1 
         if isite != nsite-1:
	    dwts,qred,u,sigs,vt = qparser.row_svd(mat,classes,thresh,Dcut)
	    bdim = len(sigs)
	    print ' isite/bdim=',isite,bdim
	    lmps[isite] = u.reshape((s[0],s[1],bdim)).copy()
	    qnuml = numpy.array(copy.deepcopy(qred))
	    qnums[isite] = qnuml.copy()
	    # Update next tsite
	    tmp = numpy.diag(sigs).dot(vt)
	    link[isite] = tmp.copy()
	    #tsite = numpy.einsum('sl,lur->sur',tmp,sites[isite+1])
	    tsite = numpy.tensordot(tmp,sites[isite+1],axes=([1],[0]))
	    sval[isite] = sigs.copy()
         else:
	    # Otherwise the MPS is zero, or have multiple qnumbers 
	    # (particle number breaking!)
	    dwts,qred,u,sigs,vt = qparser.row_svd(mat,classes,thresh,1)
	    bdim = len(sigs)
	    print ' isite/bdim=',isite,bdim
	    assert bdim == 1
	    lmps[isite] = u.reshape((s[0],s[1],bdim)).copy()
	    qnuml = numpy.array(copy.deepcopy(qred))
	    qnums[isite] = qnuml.copy()
	 if debug:
	    print '-'*80	 
	    print ' Results[i]:',isite
	    print '-'*80	 
	    print ' dimension:',(d1,d2),'->',bdim
	    print ' qred:',qred
	    print ' sum of sigs2:',numpy.sum(numpy.array(sigs)**2),' dwts:',dwts
	    print ' sigs:\n',sigs
      # Final
      self.sites = copy.deepcopy(lmps)
      self.qnums = copy.deepcopy(qnums)
      if debug: self.prt()
      print ' Bdim after: ',self.bdim()
      return link,sval

   def qrightCanon(self,thresh=1.e-12,Dcut=-1,debug=False):
      print '\n[mps_class.qrightCanon]'
      print ' Bdim before: ',self.bdim()
      assert abs(self.norm()-1.0)<1.e-8
      sites = self.sites
      nsite = len(sites)
      link = [0]*(nsite-1)
      sval = [0]*(nsite-1)
      rmps = [0]*nsite
      qnums = [0]*nsite
      # Current object
      tsite = sites[nsite-1].copy()
      # Now the symmetry is purely determined from qphys
      qnumr = [copy.deepcopy(self.qphys[0][0])]
      qnums[nsite-1] = copy.deepcopy(qnumr)
      qphys = copy.deepcopy(self.qphys)
      # rk3
      for isite in range(nsite-1,-1,-1):
         s = tsite.shape
	 d1 = s[0]
	 d2 = s[1]*s[2]
	 # Combine quantum numbers
	 qphys_isite = qphys[isite]
	 qnum1 = mpo_qphys.dpt(qphys_isite,qnumr)
         mat = tsite.reshape((d1,d2)).copy() 
         classes = qnum1
	 if isite != 0:
	    dwts,qred,u,sigs,vt = qparser.row_svd(mat.T.copy(),classes,thresh,Dcut)
	    bdim = len(sigs)
	    print ' isite/bdim=',isite,bdim
	    rmps[isite] = u.T.reshape((bdim,s[1],s[2])).copy()
	    qnumr = numpy.array(copy.deepcopy(qred))
	    qnums[isite-1] = qnumr.copy()
	    # Update next tsite
	    tmp = numpy.diag(sigs).dot(vt).T.copy()
	    link[isite-1] = tmp.copy()
	    #tsite = numpy.einsum('lur,rs->lus',sites[isite-1],tmp)
	    tsite = numpy.tensordot(sites[isite-1],tmp,axes=([2],[0]))
	    sval[isite-1] = sigs.copy()
	 else:
	    # Always cut the last one
	    dwts,qred,u,sigs,vt = qparser.row_svd(mat.T.copy(),classes,thresh,1)
	    bdim = len(sigs)
	    print ' isite/bdim=',isite,bdim
	    assert bdim == 1
	    rmps[isite] = u.T.reshape((bdim,s[1],s[2])).copy()
	 if debug:
	    print '-'*80	 
	    print ' Results[i]:',isite
	    print '-'*80	 
	    print ' dimension:',(d1,d2),'->',bdim
	    print ' qred:',qred
	    print ' sum of sigs2:',numpy.sum(numpy.array(sigs)**2),' dwts:',dwts
	    print ' sigs:\n',sigs
      # Final
      self.sites = copy.deepcopy(rmps)
      self.qnums = copy.deepcopy(qnums)
      if debug: self.prt()
      print ' Bdim after: ',self.bdim()
      return link,sval

   #--------------------------------
   # Algebra: 
   #  1. MPS1*MPS2
   #--------------------------------
   def dot(self,other):
      mps1 = self.torank2()
      mps2 = other.torank2()
      return mpslib.mps_dot(mps1,mps2)

   def norm(self):
      tmp = self.torank2()
      norm = mpslib.mps_norm(tmp)
      return norm

   def mul(self,fac):
      sfac = math.pow(abs(fac),1/float(self.nsite))
      sign = 1.0
      if fac<0.: sign = -1.0
      for k in range(self.nsite):
         self.sites[k] = sfac*self.sites[k]
      self.sites[0] = sign*self.sites[0]
      return 0

   def normalize(self):
      norm = self.norm()
      if norm > 1.e-10:
         self.mul(1.0/norm)
      else:
	 # Set to zero     
         for k in range(self.nsite):
	    shp = self.sites[k].shape
	    self.sites[k] = numpy.zeros((1,shp[1],1)) 
      return norm

   # Addition of two MPS without quantum numbers
   def add(self,other):
      N = self.nsite
      mps = class_mps(N)
      # A[1,n,a12] = {A[1,n,a1],A[1,n,a2]}
      i = 0
      shape1 = self.sites[i].shape
      shape2 = other.sites[i].shape
      dim1 = 1
      dim2 = shape1[1]
      dim3 = shape1[2]+shape2[2]
      tmp = numpy.zeros((dim1,dim2,dim3),dtype=self.sites[i].dtype)
      tmp[0:,:,:shape1[2]] = self.sites[i]
      tmp[0:,:,shape1[2]:] = other.sites[i]
      mps.sites[i] = tmp.copy()
      # Middle
      for i in range(1,N-1):
	 shape1 = self.sites[i].shape
         shape2 = other.sites[i].shape
	 dim1 = shape1[0]+shape2[0]
	 dim2 = shape1[1]
	 dim3 = shape1[2]+shape2[2]
	 tmp = numpy.zeros((dim1,dim2,dim3),dtype=self.sites[i].dtype)
	 tmp[:shape1[0],:,:shape1[2]] = self.sites[i]
	 tmp[shape1[0]:,:,shape1[2]:] = other.sites[i]
	 mps.sites[i] = tmp.copy()
      # Last site: A[a12,n,1] = {A[a1,n],A[a2,n]}
      i = N-1
      shape1 = self.sites[i].shape
      shape2 = other.sites[i].shape
      dim1 = shape1[0]+shape2[0]
      dim2 = shape1[1]
      dim3 = 1
      tmp = numpy.zeros((dim1,dim2,dim3),dtype=self.sites[i].dtype)
      tmp[:shape1[0],:,0:] = self.sites[i]
      tmp[shape1[0]:,:,0:] = other.sites[i]
      mps.sites[i] = tmp.copy()
      return mps

   def fromDet(self,vmat1):
      nb,nelec = vmat1.shape
      assert nb == self.nsite
      civec2=numpy.zeros(misc.binomial(nb,nelec))
      for strAB in itools.combinations(range(nb),nelec):
         addr=tensorRep.str2addr_o1(nb,nelec,tensorRep.string2bit(strAB))
         civec2[addr]=numpy.linalg.det(vmat1[list(strAB),:])
      # Fock-space representation
      citensor=tensorRep.toONtensor(nb,nelec,civec2)
      # ToMPS format
      mps=tensorDecomp.toMPS(citensor,[2]*nb,1.e-14,plot=False)
      #
      # It is important to realize that in toONtensor, the ordering of 
      # orbitals is actually [k,...,2,1] while in our case, we need
      # [1,2,..,k]. So a reverse procedure is need to reverse the MPS.
      #
      mps = mpslib.mps_reverse(mps)
      bdim0=mpslib.mps_bdim(mps)
      mpslib.mps_compress(mps)
      bdim1=mpslib.mps_bdim(mps)
      mpslib.mps_mps2rank3(0,mps)
      self.sites = copy.deepcopy(mps)
      return 0

   def diff(self,other,iprt=1):
      rr = mpslib.mps_diff(self.torank2(),other.torank2(),iprt)
      return rr

   # A brute-force version of 1RDM
   def makeRDM1(self):
      mps = self.torank2()
      nb = len(mps)
      rdm1 = numpy.zeros((nb,nb))
      for i in range(nb):
         for j in range(nb):
   	    print 'rdm[i,j] (i,j)=',i,j  
            ci = mpo_class.mpo_r1(nb,i,1)
            aj = mpo_class.mpo_r1(nb,j,0)
            Aij = mpo_class.mpo_r1mul(ci,aj)
            op = mpo_class.class_mpo(nb)
            op.fromRank1(Aij)
   	    rdm1[i,j] = mpslib.mps_dot(mps,op.dot(mps))
      print 'Hermicity=',numpy.linalg.norm(rdm1-rdm1.T)
      return rdm1

   # RDM1q
   def makeRDM1q(self):
      mps = self.torank2()
      nb = len(mps)
      rdm1 = numpy.zeros((4,nb,nb))
      types = [[1,0],[1,1],[0,1],[0,0]]
      for i in range(nb):
         for j in range(nb):
   	    print 'rdm[i,j] (i,j)=',i,j 
	    for ijtp in range(4):
	       itp,jtp = types[ijtp]  
	       ci = mpo_class.mpo_r1(nb,i,itp)
               cj = mpo_class.mpo_r1(nb,j,jtp)
               Aij = mpo_class.mpo_r1mul(ci,cj)
               op = mpo_class.class_mpo(nb)
               op.fromRank1(Aij)
   	       rdm1[ijtp,i,j] = mpslib.mps_dot(mps,op.dot(mps))
      rdm1q = numpy.zeros((2*nb,2*nb))
      rdm1q[:nb,:nb] = rdm1[0]
      rdm1q[:nb,nb:] = rdm1[1]
      rdm1q[nb:,:nb] = rdm1[2]
      rdm1q[nb:,nb:] = rdm1[3]
      print 'Hermicity=',numpy.linalg.norm(rdm1q-rdm1q.T)
      return rdm1q

   # many-body RDM
   def mbRDM(self,sites):
      # l,r,u,d	   
      ones = numpy.ones(1).reshape(1,1,1,1).copy()
      rdm = ones
      # Sweep from first site
      for i in range(self.nsite):
         tmp = self.sites[i].copy()
         if i in sites:
	    #   u  i
	    #   |  |
	    # 1-|==|==...
	    #   |  |
	    #   d  j
	    tmp = numpy.einsum('aib,cjd->acbdij',tmp,tmp)
	    s = tmp.shape
	    tmp = tmp.reshape((s[0]*s[1],s[2]*s[3],s[4],s[5]))
	 else:
	    #   u  1
	    #   |  |
	    # 1-|==|==...
	    #   |  |
	    #   d  1
	    tmp = numpy.einsum('aib,cid->acbd',tmp,tmp)
	    s = tmp.shape
	    tmp = tmp.reshape((s[0]*s[1],s[2]*s[3],1,1))
	 rdm = numpy.einsum('lrud,rtij->ltuidj',rdm,tmp)
	 s = rdm.shape
	 rdm = rdm.reshape((s[0],s[1],s[2]*s[3],s[4]*s[5])).copy()
      rdm = rdm[0,0]
      return rdm

   # merge physical indices
   def merge(self,partition):
      print '\n[mps_class.merge]'
      print ' Partition =',partition
      nsite = len(partition)
      tmps = [0]*nsite
      qnums = [0]*nsite
      for idx,item in enumerate(partition):
	 for jdx,isite in enumerate(item):
	    if jdx == 0:
	       cop = self.sites[isite].copy()
            else:
	       # Must be consecutive	    
	       #   |   |
	       # ---------
	       tmp = self.sites[isite].copy()
	       #tmp = numpy.einsum('lax,xbr->labr',cop,tmp)
	       tmp = numpy.tensordot(cop,tmp,axes=([2],[0]))
	       s = tmp.shape
	       cop = tmp.reshape((s[0],s[1]*s[2],s[3])).copy()
         tmps[idx] = cop.copy()
	 # Taken the quantum numbers of the last site in each group: o---
	 if self.qnums is not None: 
	    qnums[idx] = self.qnums[item[-1]]
      # Product new MPS
      if self.qphys is None:
	 if self.qnums is None:
            tmps = class_mps(nsite,sites=tmps,iop=1)
    	 else:
	    tmps = class_mps(nsite,sites=tmps,iop=1,qnums=qnums)
      else:
	 qphys = mpo_qphys.merge(self.qphys,partition)
	 print ' Merged qphys_new:'
	 for idx,iqnum in enumerate(qphys):
	    print ' idx=',idx,' iqnum=',iqnum
	 if self.qnums is None:
            tmps = class_mps(nsite,sites=tmps,iop=1,qphys=qphys)
	 else:
            tmps = class_mps(nsite,sites=tmps,iop=1,qphys=qphys,qnums=qnums)
      return tmps

#
# Another functionalities
#
def detmps(nb,nelec):
   dmps = mpslib.detmps(nb,nelec)
   return class_mps(nb,sites=dmps)

def testDet(nb=12,nelec=6):
   import jacobi
   print '\n[testDet]'
   # Init   
   numpy.random.seed(2)
   h=numpy.random.uniform(-1,1,nb*nb)
   h=h.reshape(nb,nb)
   h=0.5*(h+h.T)
   e1,v1,givens = jacobi.jacobi(h,ifswap=False)
   vmat1 = v1[:,:nelec].copy()
   # New class implementation
   mps = class_mps(nb)
   mps.fromDet(vmat1)
   bdim0 = mps.bdim()
   rdm1 = mps.makeRDM1()
   diff = numpy.linalg.norm(rdm1-vmat1.dot(vmat1.T))
   print 'RDMdiff=',diff
   if diff > 1.e-10: exit()
   #-----------------------------------------------------#
   # Try to convert MPS(in AO basis) to MPS(in MO basis) #
   # by applying the unitary transformation gates.       #
   #-----------------------------------------------------#
   ngivens = len(givens)
   mps1 = mps.copy()
   for idx,rot in enumerate(givens):
      k,l,arg = rot
      if abs(arg)<1.e-10: continue
      print '>>> idx=',idx,'/',ngivens,' arg=',arg,math.cos(arg),math.sin(arg)
      umpo = mpo_class.genU1mpo(nb,k,l,-arg)
      mps1 = copy.deepcopy(umpo.dotMPS(mps1))
      print 'mps.bdim = ',mps1.bdim()
      mps1.compress()
   # Print 
   bdim0 = mps.bdim()
   bdim1 = mps1.bdim()
   print 'bdim0',bdim0
   print 'bdim1',bdim1
   mps1.compress(thresh=1.e-8)
   print 'bdim2',mps1.bdim()
   # Comparison
   dmps = detmps(nb,nelec)
   rr = dmps.diff(mps1,iprt=1)
   print 'diff=',rr
   nmpo = mpo_class.occmpo(nb)
   nelec1 = dmps.dot(nmpo.dotMPS(dmps))
   nelec2 = mps1.dot(nmpo.dotMPS(mps1))
   print 'Check nelec:'
   print 'nelec =',nelec
   print 'nelec1=',nelec1
   print 'nelec2=',nelec2
   #
   # bdim0 [2, 4, 8, 16, 32, 64, 128, 256, 128, 64, 32, 16, 8, 4, 2]
   # bdim1 [2, 3, 3, 3, 3, 3, 4, 4, 4, 3, 2, 2, 2, 1, 1]
   # [mps_diff] 
   # rr=0.00000e+00 
   # pp=1.00000 
   # pq=1.00000 
   # qp=1.00000 
   # qq=1.00000 
   # diff= 0.0
   # Check nelec:
   # nelec = 8
   # nelec1= 8.0
   # nelec2= 8.00000000001
   # 
   return 0

def testQnum():
   k = 6
   n = 3
   sordering = [0,1,3,2,4,5]
   occ = [1]*n + [0]*(k-n)
   occ = numpy.array(occ)[sordering]
   print 'occ=',occ
   mps0 = class_mps(k)
   mps0.hfstate(n,occ)
   mps0.prt()
   return 0 

# TEST
if __name__ == '__main__':
   #testDet()
   testQnum()
