import h5py
import time
import numpy
import qtensor
import qtensor_util
import qtensor_opers
from mpodmrg.source.mpsmpo import mps_io
from mpodmrg.source import mpo_dmrg_qphys
from mpodmrg.source import mpo_dmrg_opers

#@profile
def test_tensordot():
   # LOAD MPS
   fname = './mps.h5'
   mps,qnum = mps_io.loadMPS(fname)
   lmps = [mps,qnum]
   bdim = map(lambda x:len(x),qnum)
   print ' bdim = ',bdim
   for item in mps:
      print item.shape
   for item in qnum:
      print item
   nsite = len(mps)
   print nsite
   qphys = mpo_dmrg_qphys.initSpatialOrb(nsite,2)
   print qphys
   print len(qnum)
   ta = 0.
   tb = 0.
   for isite in range(nsite):
      ql = qnum[isite]
      qn = qphys[isite]
      qr = qnum[isite+1]
      cl = qtensor_util.classification(ql)
      cn = qtensor_util.classification(qn)
      cr = qtensor_util.classification(qr)
      site = mps[isite]
      print
      print 'isite=',isite
      #print len(ql),len(qn),len(qr)
      #print 'cl',cl
      #print 'cn',cn
      #print 'cr',cr
      tmps = qtensor.qtensor([False,False,True])
      tmps.fromDenseTensor(site,[ql,qn,qr])     
      tsite = tmps.toDenseTensor()
      diffDense = numpy.linalg.norm(tsite-site)
      print ' diffDense=',diffDense
      assert diffDense<1.e-12
      ##
      ## test-1
      ##
      #print site.shape
      #t0 = time.time()
      #tmp = numpy.tensordot(site,site,axes=([0,1],[0,1]))
      ##tmp1 = numpy.einsum('ijk,lmn',site,site)
      ##print 'outer=',numpy.linalg.norm(tmp-tmp1)
      #t1 = time.time()
      #print 'norm=',numpy.linalg.norm(tmp),'t1-t0=',t1-t0
      #t1 = time.time()
      #tmp2 = qtensor.tensordot(tmps,tmps,axes=([0,1],[0,1]),debug=False)
      #t2 = time.time()
      #print 'norm=',numpy.linalg.norm(tmp2.value),'t2-t1=',t2-t1
      #print 'ratio=',(t2-t1)/(t1-t0)
      #ta += t1-t0
      #tb += t2-t1
      ## compare
      #tmp3 = tmp2.toDenseTensor()
      #diffDense2 = numpy.linalg.norm(tmp-tmp3)
      #print ' diffDense2=',diffDense2
      #assert diffDense2<1.e-12
      #
      # test-2: full contraction
      #
      print 'full contraction:',site.shape
      t0 = time.time()
      tmp = numpy.tensordot(site,site,axes=([0,1,2],[0,1,2]))
      t1 = time.time()
      print 'norm=',numpy.linalg.norm(tmp),'t1-t0=',t1-t0
      t1 = time.time()
      tmp2 = qtensor.tensordot(tmps,tmps,axes=([0,1,2],[0,1,2]),debug=False)
      t2 = time.time()
      print 'norm=',numpy.linalg.norm(tmp2),'t2-t1=',t2-t1
      print 'ratio=',(t2-t1)/(t1-t0)
      ta += t1-t0
      tb += t2-t1
      # compare
      diff = numpy.linalg.norm(tmp-tmp2)
      print ' diff=',diff
      assert diff<1.e-10
   print 
   print 'ta=',ta
   print 'tb=',tb
   print
   return 0

#@profile
def test_transpose_merge():
   # LOAD MPS
   fname = './mps.h5'
   mps,qnum = mps_io.loadMPS(fname)
   lmps = [mps,qnum]
   bdim = map(lambda x:len(x),qnum)
   print ' bdim = ',bdim
   nsite = len(mps)
   qphys = mpo_dmrg_qphys.initSpatialOrb(nsite,2)
   for isite in range(nsite):
      ql = qnum[isite]
      qn = qphys[isite]
      qr = qnum[isite+1]
      cl = qtensor_util.classification(ql)
      cn = qtensor_util.classification(qn)
      cr = qtensor_util.classification(qr)
      site = mps[isite]
      tsite0 = site.transpose(2,0,1)
      print 'isite=',isite
      tmps  = qtensor.qtensor([False,False,True])
      tmps.fromDenseTensor(site,[ql,qn,qr])     
      tmps  = tmps.transpose(2,0,1)
      tsite = tmps.toDenseTensor()
      diff1 = numpy.linalg.norm(tsite0-tsite)
      print ' diff1=',diff1
      shape = tsite0.shape
      tmp   = tsite0.reshape((shape[0],shape[1]*shape[2]))
      tmps  = tmps.merge([[0],[1,2]])
      tmat  = tmps.toDenseTensor()
      diff2 = numpy.linalg.norm(tmp-tmat) 
      print ' diff2=',diff2
   return 0

#@profile
def test_creann():
   # LOAD MPS
   fname = './mps.h5'
   mps,qnum = mps_io.loadMPS(fname)
   lmps = [mps,qnum]
   bdim = map(lambda x:len(x),qnum)
   print ' bdim = ',bdim
   nsite = len(mps)
   qphys = mpo_dmrg_qphys.initSpatialOrb(nsite,2)
   ta = 0.
   tb = 0.
   for isite in range(nsite):
      ql = qnum[isite]
      qn = qphys[isite]
      qr = qnum[isite+1]
      cl = qtensor_util.classification(ql)
      cn = qtensor_util.classification(qn)
      cr = qtensor_util.classification(qr)
      print 'isite/nsite=',isite,nsite
      site = mps[isite]
      tmps  = qtensor.qtensor([False,False,True])
      tmps.fromDenseTensor(site,[ql,qn,qr])    
      tmps.prt()
      for iop in [1,0]:
         for p in range(2*nsite):
	    t0 = time.time()
	    op = mpo_dmrg_opers.genElemSpatialMat(p,isite,iop)
	    #csite = numpy.einsum('ij,ajb->aib',op,site)
	    csite = numpy.tensordot(op,site,axes=([1],[1])) # iab
	    csite = csite.transpose(1,0,2)
	    t1 = time.time()
	    qop = qtensor_opers.genElemSpatialQt(p,isite,iop)
	    # ijab,xby-> ijaxy -> ix,a,jy
	    tmps2 = qtensor.tensordot(qop,tmps,axes=([3],[1]))
	    tmps2 = tmps2.transpose(0,3,2,1,4)
	    tmps2 = tmps2.merge([[0,1],[2],[3,4]])
	    tsite = tmps2.toDenseTensor()
	    t2 = time.time()
	    assert csite.shape == tsite.shape
	    diff = numpy.linalg.norm(tsite-csite)
	    print 'iop,p,diff=',iop,p,csite.shape,diff,' t0=',t1-t0,' t1=',t2-t1
	    assert diff < 1.e-10
	    ta += t1-t0
	    tb += t2-t1
   # In case of large bond dimension, e.g., 
   # D=2000, t0/t1~0.21/0.09 due to sparsity!
   print
   print 'ta=',ta # ta= 20.7766697407
   print 'tb=',tb # tb= 18.2862818241
   print
   return 0

def test_Wfac():
#
# Wop*Site = [256*30,4,30*1015] = [935424000] - 7G
#
# isite/nsite= 4 14
# Basic information:
#  rank = 3  shape= [ 256    4 1015]  nsyms= [25  4 36]
#  nblks_allowed = 100  nblks = 3600
#  size_allowed  = 62804  size = 1039360  savings= 0.0604256465517
#  wop= (30, 30, 4, 4)
#  t1= 13.3494501114
#  t2= 2.59876251221e-05
# 
# isite/nsite= 5 14	--- 50G for storage.
# Basic information:
#  rank = 3  shape= [1015    4 1822]  nsyms= [36  4 44]
#  nblks_allowed = 136  nblks = 6336
#  size_allowed  = 376882  size = 7397320  savings= 0.0509484516014
#  wop= (30, 30, 4, 4)
# 
#>>> Sparse op becomes better for D~500 for computational time.
#
#isite/nsite= 3 14
#Basic information:
# rank = 3  shape= [ 64   4 256]  nsyms= [16  4 25]
# nblks_allowed = 64  nblks = 1600
# size_allowed  = 4900  size = 65536  savings= 0.0747680664062
# wop= (30, 30, 4, 4)
# t1= 0.615185976028
# t2= 0.715934991837
#isite,diff= 3 (1920, 4, 7680) 0.0  t0= 0.615185976028  t1= 0.715934991837
#
#isite/nsite= 4 14
#Basic information:
# rank = 3  shape= [ 256    4 1015]  nsyms= [25  4 36]
# nblks_allowed = 100  nblks = 3600
# size_allowed  = 62804  size = 1039360  savings= 0.0604256465517
# wop= (30, 30, 4, 4)
# t1= 10.8923699856
# t2= 3.5177989006
#isite,diff= 4 (7680, 4, 30450) 0.0  t0= 10.8923699856  t1= 3.5177989006
#
#ta= 11.5469501019
#tb= 4.54184865952
#
   # LOAD MPS
   fname = './mps.h5'
   mps,qnum = mps_io.loadMPS(fname)
   lmps = [mps,qnum]
   bdim = map(lambda x:len(x),qnum)
   print ' bdim = ',bdim
   nsite = len(mps)
   qphys = mpo_dmrg_qphys.initSpatialOrb(nsite,2)
   ta = 0.
   tb = 0.
   isz  = 0
   nbas = 2*nsite
   # random
   hmo = numpy.random.uniform(-1,1,(nbas,nbas))
   hmo[::2,1::2]=hmo[1::2,::2]=0.
   # [ij|kl]
   eri = numpy.random.uniform(-1,1,(nbas,nbas,nbas,nbas))
   eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
   # The spin symmetry is essential.
   #   <ij|kl>=[ik|jl]
   eri = eri.transpose(0,2,1,3)
   hq = hmo[isz]
   vqrs = eri[isz]
   maxn = 3 
   for isite in range(min(nsite,maxn)):
      ql = qnum[isite]
      qn = qphys[isite]
      qr = qnum[isite+1]
      cl = qtensor_util.classification(ql)
      cn = qtensor_util.classification(qn)
      cr = qtensor_util.classification(qr)
      print
      print 'isite/nsite=',isite,nsite
      site = mps[isite]
      tmps  = qtensor.qtensor([False,False,True])
      tmps.fromDenseTensor(site,[ql,qn,qr])    
      tmps.prt()
	 
      t0 = time.time()
      op = mpo_dmrg_opers.genWfacSpatial(nbas,isite,hq,vqrs)
      print ' wop=',op.shape
      #csite = numpy.einsum('lrij,ajb->lairb',op,site)
      csite = numpy.tensordot(op,site,axes=([3],[1])) # lriab
      csite = csite.transpose(0,3,2,1,4) # lriab->lairb
      s = csite.shape
      csite = csite.reshape((s[0]*s[1],s[2],s[3]*s[4]))
      t1 = time.time()
      print ' t1=',t1-t0

      qop = qtensor_opers.genWfacSpatialQt(nbas,isite,hq,vqrs,isz)
      tmps2 = qtensor.tensordot(qop,tmps,axes=([3],[1]))
      tmps2 = tmps2.transpose(0,3,2,1,4)
      tmps2 = tmps2.merge([[0,1],[2],[3,4]])
      tsite = tmps2.toDenseTensor()
      t2 = time.time()
      print ' t2=',t2-t1
   
      assert csite.shape == tsite.shape
      diff = numpy.linalg.norm(tsite-csite)
      print 'isite,diff=',isite,csite.shape,diff,' t0=',t1-t0,' t1=',t2-t1
      assert diff < 1.e-10
      ta += t1-t0
      tb += t2-t1
   print
   print 'ta=',ta 
   print 'tb=',tb 
   print
   return 0

def test_Hfac():
#
#------------------------------------------------------------------------
# The large memory cost of Wop*|Psi> [O(K2D2)] requires, for large Dmps
# and Dwop, a sequential compression should be implemented as a sweep,
# such that the [Wsite*MPSsite] can be avoided partially.
#------------------------------------------------------------------------
#
# isite,diff= 0 (1, 4, 232) 0.0  t0= 0.0527150630951  t1= 0.0431280136108
# isite,diff= 1 (232, 4, 928) 0.0  t0= 0.0108029842377  t1= 0.050961971283
# isite,diff= 2 (928, 4, 3712) 0.0  t0= 0.146265029907  t1= 0.182390928268
# isite,diff= 3 (3712, 4, 14848) 0.0  t0= 2.27565908432  t1= 1.25626206398
# isite= 4 [14848     4 58870]  t1= 3.11030101776
# isite= 5 [ 58870      4 105676]  t1= 30.1744351387
# isite= 6 [105676      4 162806]  t1= 101.686741114
# isite= 7 [162806      4  76966]  t1= 75.6775298119
# isite= 8 [76966     4 29638]  t1= 8.44200801849
# isite= 9 [29638     4  9976]  t1= 1.69688081741
# isite= 10 [9976    4 3074]  t1= 0.343852996826
# isite= 11 [3074    4  870]  t1= 0.101953983307
# isite= 12 [870   4 232]  t1= 0.044182062149
# isite= 13 [232   4   1]  t1= 0.00591492652893
# 
   # LOAD MPS
   fname = './mps.h5'
   mps,qnum = mps_io.loadMPS(fname)
   lmps = [mps,qnum]
   bdim = map(lambda x:len(x),qnum)
   print ' bdim = ',bdim
   nsite = len(mps)
   qphys = mpo_dmrg_qphys.initSpatialOrb(nsite,2)
   ta = 0.
   tb = 0.
   nbas = 2*nsite
   # random
   hmo = numpy.random.uniform(-1,1,(nbas,nbas))
   hmo[::2,1::2]=hmo[1::2,::2]=0.
   # [ij|kl]
   eri = numpy.random.uniform(-1,1,(nbas,nbas,nbas,nbas))
   eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
   # The spin symmetry is essential.
   #   <ij|kl>=[ik|jl]
   eri = eri.transpose(0,2,1,3)
   maxn = 3 
   for p in [6,5]:
      isz  = p%2
      hq = hmo[isz]
      vqrs = eri[isz]
      for isite in range(min(nsite,5)):
         ql = qnum[isite]
         qn = qphys[isite]
         qr = qnum[isite+1]
         cl = qtensor_util.classification(ql)
         cn = qtensor_util.classification(qn)
         cr = qtensor_util.classification(qr)
         print
         print 'isite/nsite=',isite,nsite
         site = mps[isite]
         tmps  = qtensor.qtensor([False,False,True])
         tmps.fromDenseTensor(site,[ql,qn,qr])    
	 print 'mps site info:'
	 tmps.prt()
           
         if isite < maxn:
            t0 = time.time()
            
	    #op = mpo_dmrg_opers.genWfacSpatial(nbas,isite,hq,vqrs)
	    op = mpo_dmrg_opers.genHfacSpatial(p,nbas,isite,hq,vqrs)
	    print ' wop=',op.shape
            #csite = numpy.einsum('lrij,ajb->lairb',op,site)
            csite = numpy.tensordot(op,site,axes=([3],[1])) # lriab
            csite = csite.transpose(0,3,2,1,4) # lriab->lairb
            s = csite.shape
            csite = csite.reshape((s[0]*s[1],s[2],s[3]*s[4]))
            t1 = time.time()
            print ' t1=',t1-t0

	    #qop = qtensor_opers.genWfacSpatialQt(nbas,isite,hq,vqrs,isz)
            qop = qtensor_opers.genHfacSpatialQt(p,nbas,isite,hq,vqrs)
	    tmps2 = qtensor.tensordot(qop,tmps,axes=([3],[1]))
            tmps2 = tmps2.transpose(0,3,2,1,4)
            tmps2 = tmps2.merge([[0,1],[2],[3,4]])
            tsite = tmps2.toDenseTensor()
            t2 = time.time()
            print ' t2=',t2-t1
      
            assert csite.shape == tsite.shape
            diff = numpy.linalg.norm(tsite-csite)
            print 'isite,diff=',isite,csite.shape,diff,' t0=',t1-t0,' t1=',t2-t1
            assert diff < 1.e-10
            ta += t1-t0
            tb += t2-t1

         else:

	    if isite == maxn: print '>>> Check internal consistency <<<'
	    t0 = time.time()
            #qop = qtensor_opers.genWfacSpatialQt(nbas,isite,hq,vqrs,isz)
            qop = qtensor_opers.genHfacSpatialQt(p,nbas,isite,hq,vqrs)
	    tmps2 = qtensor.tensordot(qop,tmps,axes=([3],[1]))
            tmps2 = tmps2.transpose(0,3,2,1,4)
            tmps2 = tmps2.merge([[0,1],[2],[3,4]])
            tmps2.prt()
            t1 = time.time()
	    sum1 = numpy.sum(tmps2.value)
            print 'isite=',isite,tmps.shape,' t0=',t1-t0,' sum=',sum1
	    tmps2 = None

	    qop = qtensor_opers.genHfacSpatialQt0(p,nbas,isite,hq,vqrs)
	    tmps2 = qtensor.tensordot(qop,tmps,axes=([3],[1]))
            tmps2 = tmps2.transpose(0,3,2,1,4)
            tmps2 = tmps2.merge([[0,1],[2],[3,4]])
            tmps2.prt()
	    t2 = time.time()
	    sum2 = numpy.sum(tmps2.value)
            print 'isite=',isite,tmps2.shape,' t1=',t2-t1,' sum=',sum2
	    tmps2 = None

	    diff = abs(sum1-sum2)
	    print 'diff =',diff
	    assert diff<1.e-10
   print
   print 'ta=',ta 
   print 'tb=',tb 
   print
   return 0


def test_HRfac():
   # LOAD MPS
   fname = './mps.h5'
   mps,qnum = mps_io.loadMPS(fname)
   lmps = [mps,qnum]
   bdim = map(lambda x:len(x),qnum)
   print ' bdim = ',bdim
   nsite = len(mps)
   qphys = mpo_dmrg_qphys.initSpatialOrb(nsite,2)
   ta = 0.
   tb = 0.
   nbas = 2*nsite
   # random
   hmo = numpy.random.uniform(-1,1,(nbas,nbas))
   hmo[::2,1::2]=hmo[1::2,::2]=0.
   # [ij|kl]
   eri = numpy.random.uniform(-1,1,(nbas,nbas,nbas,nbas))
   eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
   # The spin symmetry is essential.
   #   <ij|kl>=[ik|jl]
   eri = eri.transpose(0,2,1,3)
   maxn = 3 
   for p in [6,5]:
      isz  = p%2
      for isite in range(min(nsite,5)):
         ql = qnum[isite]
         qn = qphys[isite]
         qr = qnum[isite+1]
         cl = qtensor_util.classification(ql)
         cn = qtensor_util.classification(qn)
         cr = qtensor_util.classification(qr)
         print
         print 'isite/nsite=',isite,nsite
         site = mps[isite]
         tmps  = qtensor.qtensor([False,False,True])
         tmps.fromDenseTensor(site,[ql,qn,qr])    
	 print 'mps site info:'
	 tmps.prt()
           
         if isite < maxn:
	    # Reference value:
            t0 = time.time()
	    pindx = (p,0)
	    qpts = numpy.array([0.3])
	    op = mpo_dmrg_opers.genHRfacSpatial(pindx,nbas,isite,hmo,eri,qpts)
	    print ' wop=',op.shape
            #csite = numpy.einsum('lrij,ajb->lairb',op,site)
            csite = numpy.tensordot(op,site,axes=([3],[1])) # lriab
            csite = csite.transpose(0,3,2,1,4) # lriab->lairb
            s = csite.shape
            csite = csite.reshape((s[0]*s[1],s[2],s[3]*s[4]))
            t1 = time.time()
            print ' t1=',t1-t0

	    # Lowering the symmetry of MPS?
	    qop = qtensor_opers.genHRfacSpatialQt(pindx,nbas,isite,hmo,eri,qpts)
	    # We need to change qop construction allowing given qsyms !
	    tmps2 = tmps.reduceQsymsToN()
	    #tmps2 = tmps2.projectionNMs(tmps.qsyms)
	    #diff = numpy.linalg.norm(tmps2.value-tmps.value)
	    #print ' diff=',diff
	    tmps2 = qtensor.tensordot(qop,tmps2,axes=([3],[1]))
            tmps2 = tmps2.transpose(0,3,2,1,4)
            tmps2 = tmps2.merge([[0,1],[2],[3,4]])
            tsite = tmps2.toDenseTensor()
            
	    t2 = time.time()
            print ' t2=',t2-t1
            assert csite.shape == tsite.shape
            diff = numpy.linalg.norm(tsite-csite)
            print 'isite,diff=',isite,csite.shape,diff,' t0=',t1-t0,' t1=',t2-t1
            assert diff < 1.e-10
            ta += t1-t0
            tb += t2-t1

	 else:

	    # Reference value:
            t0 = time.time()
	    pindx = (p,0)
	    qpts = numpy.array([0.3])
            t1 = time.time()
	    qop = qtensor_opers.genHRfacSpatialQt(pindx,nbas,isite,hmo,eri,qpts)
	    tmps2 = tmps.reduceQsymsToN()
	    tmps2 = qtensor.tensordot(qop,tmps2,axes=([3],[1]))
	    print 'before transposing:'
	    tmps2.prt()
            tmps2 = tmps2.transpose(0,3,2,1,4)
	    print 'after transposing:'
	    tmps2.prt()
            tmps2 = tmps2.merge([[0,1],[2],[3,4]])
	    print 'after merging:'
	    tmps2.prt()
	    t2 = time.time()
            print 'isite=',isite,' t2=',t2-t1

   print
   print 'ta=',ta 
   print 'tb=',tb 
   print
   return 0

#@profile
def test_io():
   # LOAD MPS
   fname = './mps.h5'
   mps,qnum = mps_io.loadMPS(fname)
   lmps = [mps,qnum]
   bdim = map(lambda x:len(x),qnum)
   print ' bdim = ',bdim
   for item in mps:
      print item.shape
   for item in qnum:
      print item
   nsite = len(mps)
   print nsite
   qphys = mpo_dmrg_qphys.initSpatialOrb(nsite,2)
   print qphys
   print len(qnum)
   f1 = h5py.File("mpsQt.h5","w") 
   for isite in range(nsite):
      ql = qnum[isite]
      qn = qphys[isite]
      qr = qnum[isite+1]
      site = mps[isite]
      print
      print 'isite=',isite
      tmps = qtensor.qtensor([False,False,True])
      tmps.fromDenseTensor(site,[ql,qn,qr])    
      tmps.dump(f1,'site'+str(isite))

      tmps2 = qtensor.qtensor()
      tmps2.load(f1,'site'+str(isite))
      tmps2.prt()

   f1.close()
   return 0

if __name__ == '__main__':
    
   #test_transpose_merge()
   #test_creann()
   
   for i in range(30): test_io()
   
   #test_tensordot()
   #test_Wfac()
   #test_Hfac()
   #test_HRfac()
