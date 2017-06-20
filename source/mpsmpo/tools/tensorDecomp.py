#==========================================
# Various tensor decomposition schemes
# 1. MPS [tensor chain]
# 2. Tucker decomposition
# 3. CP decomposition via ALS algorithm
#==========================================
import math
import numpy
import scipy.linalg
import misc
import mpslib
import tensorSubs

#==========================================
# toMPS 
#==========================================
def toMPS(tensor,shape,thresh=1.e-10,**kwargs):
   print '\n[toMPS]'
   #
   # C[p1,p2,p3,p4]=U[p1,x]*s1V[d1,p2,p3,p4]
   # 		   	   =U[d1*p2,d2]*s2V[d2,p3,p4]
   #			   	       =U[d2,p3,d3]*s3V[d3,p4]
   #
   rank=len(shape)
   print 'shape=',shape,' size= ',tensor.shape,' rank=',rank
   if reduce(lambda x,y:x*y,shape) != tensor.size:
      print 'ERROR: inconsistent size!'
      exit(1)
   # BEGIN
   tensor2=tensor.copy()
   tensor2=tensor2.reshape(shape[0],-1)
   mps=[]
   isite=0
   isig1=[]
   for i in range(rank-1):
      print '\n--- site[i] =',i,'---'
      u1,sig1,v1=mpslib.mps_svd_cut(tensor2,thresh,-1)
      print 'sig1=',sig1.shape,'\n',sig1
      print 'stat=',len(numpy.argwhere(sig1>thresh))
      if len(sig1)>len(isig1): 
         isig1=sig1
	 isite=i
      mps.append(u1)
      tensor2=numpy.diag(sig1).dot(v1)
      if i < rank-2:
         tensor2=tensor2.reshape(u1.shape[1]*shape[i+1],-1)
      else:
	 mps.append(tensor2)   
   # Reshaping into lnr for latter compressing
   for i in range(1,rank-1):
      #print 'shape=',mps[i-1].shape[-1],shape[i],mps[i].shape[1]
      mps[i]=mps[i].reshape(mps[i-1].shape[-1],shape[i],mps[i].shape[1])
   # Final
   print '\nFinal MPS info:'
   print 'Phys dim:',shape
   print 'Bond dim:',mpslib.mps_bdim(mps)
   print 'Maximum bond dimension: site/rank =',isite,rank
   print 'Singular value =\n',isig1
   print 'Length = ',len(isig1),' Maximum=',shape[0]**(isite+1),\
	 ' Ratio = ',len(isig1)*1.0/shape[0]**(isite+1)
   mpslib.mps_check(tensor,shape,mps)
   if kwargs.get('plot')==True:
      import matplotlib.pyplot as plt
      plt.plot(range(len(isig1)),numpy.log10(isig1))
      plt.show()
   return mps

def toMPSc(tensor,shape,thresh=1.e-10): 
   # cast tensor with shape into mps 
   mps=toMPS(tensor,shape,thresh)
   D=-1
   mpslib.mps_leftSVD(mps,thresh,D)
   mpslib.mps_rightSVD(mps,thresh,D)
   #mpslib.mps_compress(mps,thresh)
   mpslib.mps_check(tensor,shape,mps)
   return mps

#==========================================
# toHSMPS (NOT CLEANED UP) 
#==========================================
import copy
import collections
def toHSMPS(tensor,shape,thresh=1.e-10,**kwargs):
   debug=False
   print '\n[toHSMPS]'
   #
   # C[p1,p2,p3,p4]=U[p1,x]*s1V[d1,p2,p3,p4]
   # 		   	   =U[d1*p2,d2]*s2V[d2,p3,p4]
   #			   	       =U[d2,p3,d3]*s3V[d3,p4]
   #
   rank=len(shape)
   print 'shape=',shape,' size= ',tensor.shape,' rank=',rank
   if reduce(lambda x,y:x*y,shape) != tensor.size:
      print 'ERROR: inconsistent size!'
      exit(1)
   # BEGIN
   tensor2=tensor.copy()
   mps=[]
   isite=0
   isig1=[]
   
   #--- Qsymmetry ---
   mps  =[0]*rank
   qbond=[0]*(rank-1)
   nphys=shape[0]

   ## first site
   #mps[0]=numpy.eye(nphys)
   #qbond[0]=[j for j in range(nphys)]
   #tensor2=tensor2.reshape(shape[0]*shape[1],-1)
   tensor2=tensor2.reshape(shape[0],-1)

   for i in range(0,rank-1):
      print '\n--- site[i] =',i,'---'
      qsym1=[i+j for j in range(nphys)]
      if debug: print 'qsym1=',qsym1
      if i==0:
	 qsyml = map(lambda x:[x,x],qsym1)
      else:	 
	 if debug: print 'qbond[i-1]',qbond[i-1]
         qsyml=hsmps_dpd(qbond[i-1],qsym1)
      if debug: print 'qsyml=',qsyml
      
      cnt=collections.Counter()
      for qnum,index in qsyml:
         cnt[qnum]+=1
      if debug:
         print 'items',cnt.items()   
         print 'keys:',list(cnt)
         print 'vals:',cnt.values()
      keys=list(cnt)
      vals=cnt.values()
      nsym=len(keys)

      qL = map(lambda x:x[0],qsyml)
      if debug: print "qL=",qL
      qstateL=[]
      qstateR=[]
      qinfo=[]
      sigval=[]
      for j in range(nsym):
         key = keys[j]
	 val = vals[j]
	 if debug: print '\n i/sym_i[L] = ',i,' key=',key,' val=',val
         rows = [k for k,x in enumerate(qL) if x==key]
	 if debug: print ' rows = ',rows 
         qrow=tensor2[rows,:]
         u, sig, v = mpslib.mps_svd_cut(qrow,thresh,-1)
	 v = numpy.diag(sig).dot(v)
         nres=len(sig)
	 if debug:
	    print ' SVD for CI shape of =',qrow.shape,' -> ',len(sig)
            print ' sigval=',sig
	 if sig[0]<thresh:
            print ' Too small sigs!'
	    continue
         sigval=sigval+list(sig)
	 qinfo.append((key,val,nres))
         qstateL.append(u)
         qstateR.append(v)
      tensor2=numpy.vstack(qstateR)
      mps.append(qstateL)
      qsym=[]
      for sym in qinfo:
	 qsym=qsym+[sym[0]]*sym[2]
      qbond[i]=copy.deepcopy(qsym)
      if debug:
	 print ' qbond[i]=',qbond[i]
      print ' sigvals =',sigval

      if len(isig1)<len(sigval): 
         isig1=sigval
	 isite=i
      if i < rank-2:
         tensor2=tensor2.reshape(tensor2.shape[0]*shape[i+1],-1)
      else:
	 mps.append(tensor2)

#   # Reshaping into lnr for latter compressing
#   for i in range(1,rank-1):
#      #print 'shape=',mps[i-1].shape[-1],shape[i],mps[i].shape[1]
#      mps[i]=mps[i].reshape(mps[i-1].shape[-1],shape[i],mps[i].shape[1])
#   # Final
   print '\nFinal MPS info:'
   print 'Phys dim:',shape
   if debug: print 'Q(uantum number)bond:',qbond
   print 'Bond dim:',map(lambda x:len(x),qbond)
   isig1=numpy.array(sorted(isig1,reverse=True))
   print 'Singular value =\n',isig1
   if kwargs.get('plot')==True:
      import matplotlib.pyplot as plt
      plt.plot(range(len(isig1)),numpy.log10(isig1))
      plt.show()
   return mps

def hsmps_dpd(qsym1,qsym2):
   qsym12=[]
   ic=0
   print qsym1
   print qsym2
   for i in range(len(qsym1)):
      sym1=qsym1[i]
      for j in range(len(qsym2)):
         sym2=qsym2[j]
         if sym1 < sym2:
	    qsym12.append([sym2,ic])
         else:
	    qsym12.append([99999,ic])
	 ic=ic+1
   return qsym12

#==========================================
# toTucker
#==========================================
import tuckerlib
def toTucker(tensor,shape,thresh=1.e-10):
   print '\n[toTucker]'
   #
   # C[p1,p2,p3,p4]=U[p1,d1]*s1V[d1,p2,p3,p4]
   # 		   	   =>s1V[p2,p3,p4,d1] (Transpose)
   #			   = U[p2,d2]*s3V[p3,p4,d1]
   #
   rank=len(shape)
   print 'shape=',shape,' size= ',tensor.shape,' rank=',rank
   if reduce(lambda x,y:x*y,shape) != tensor.size:
      print 'ERROR: inconsistent size!'
      exit(1)
   # BEGIN
   tensor2=tensor.copy()
   tensor2=tensor2.reshape(shape[0],-1)
   tucker_site=[]
   isite=0
   isig1=[]
   tshape=[]
   for i in range(rank):
      print '\n--- site[i] =',i,'---'
      u1,sig1,v1=mpslib.mps_svd_cut(tensor2,thresh,-1)
      print 'sig1=',sig1.shape,'\n',sig1
      print 'stat=',len(numpy.argwhere(sig1>thresh))
      if len(sig1)>len(isig1): 
         isig1=sig1
	 isite=i
      tucker_site.append(u1)
      tshape.append(u1.shape[1])
      tensor2=numpy.diag(sig1).dot(v1)
      # Tucker decomposition via permute
      # sV[a|p2p3p4]=>sV[p2p3p4|a]
      tensor2=tensor2.transpose(1,0)
      if i<rank-1:
         # =>sV[p2|p3p4a]
         tensor2=tensor2.reshape(shape[i+1],-1)
      else:
         # sV[a1a2a3a4]
         tucker_core=tensor2.reshape(tshape)
   # Final
   print '\nFinal Tucker info:'
   print 'Phys dim:',shape
   print 'Bond dim:',tucker_core.shape
   print 'Maximum bond dimension: site/rank =',isite,rank
   print 'Singular value =\n',isig1
   print 'Length = ',len(isig1),' Maximum=',shape[0]**(isite+1),\
	 ' Ratio = ',len(isig1)*1.0/shape[0]**(isite+1)
   tucker=(tucker_core,tucker_site)
   tuckerlib.tucker_check(tensor,shape,tucker)
   return tucker 

#==========================================
# toCP
#==========================================
import cplib
def toCP(tensor,shape,thresh=1.e-10,**kwargs):
   print '\n[toCP] alternating least square (ALS) algorithm'
   #
   # T[p1,p2,p3] => sum_r lambda[r]A1[r,p3]A2[r,p2]A3[r,p3]
   #
   rank=len(shape)
   print 'shape=',shape,' size= ',tensor.shape,' rank=',rank
   if reduce(lambda x,y:x*y,shape) != tensor.size:
      print 'ERROR: inconsistent size!'
      exit(1)
   # BEGIN
   maxiter=1000
   error  =1.0
   rank   =1
   cp     =(0,0)
   tensor2=tensor.copy()
   while error>thresh: 
      info,errorlst,cp=cp_als(tensor2,shape,maxiter,thresh,rank,cp)
      if info == 0: break
      rank=rank+1
   # Final
   print '\nFinal CP info:'
   print 'Phys dim:',shape
   print 'Rank_CP :',len(cp[0])
   print 'CP_vals :',cp[0]
   cplib.cp_check(tensor,shape,cp,1)
   # Plot
   if kwargs.get('plot')==True:
      import matplotlib.pyplot as plt
      plt.plot(range(len(errorlst)),numpy.log10(errorlst))
      plt.show()
   return cp

def cp_als(tensor,shape,maxiter,thresh,rank,seed=(0,0)):
   print '\n[cp_als] bdim=',rank,' maxiter=',maxiter 
   cp_core=[]    
   cp_site=[]
   N=len(shape)
   # Initialization
   cp_core=[0.01]*rank
   if isinstance(seed[0],numpy.ndarray):
      r0=len(seed[0])
      cp_core[:r0]=seed[0][:r0]
   for i in range(N):
      if isinstance(seed[0],numpy.ndarray):
	 u=seed[1][i]     
      else:
	 tmat=tensorSubs.tensor_matricization(tensor,shape,i)[1]
         u,sig,v=scipy.linalg.svd(tmat,full_matrices=False)
         u=u.transpose(1,0)
      dim=u.shape[0]
      if rank > dim:   	
         nres=rank-dim
         ures=numpy.random.uniform(-1,1,nres*shape[i])
         ures=ures.reshape(nres,shape[i])
         ures=map(lambda x: x*1.0/numpy.linalg.norm(x),ures)
         u=numpy.vstack((u,ures))
      cp_site.append(u[:rank])
   cp=(cp_core,cp_site)
   # Begin   
   info=0
   it  =0
   errorlst=[]
   while it <= maxiter:
      error=cplib.cp_check(tensor,shape,cp,0)
      print '--- it=',it,' error=',error
      errorlst.append(error)
      if error<=thresh: 
         print "ALS converged!"
	 info=0
	 break
      if it >=1 and abs(errorlst[-1]-errorlst[-2])<1.e-2*error:
	 print "No enough decrease!"
	 info=1
	 break
      # solve G*B=T (B=G^+*T) via least square problem
      for i in range(N):
         order=range(N)
	 order.remove(i)
 	 # G^+
	 complement=[cp_site[j] for j in order]
	 ctensor=reduce(tensorSubs.matrix_KRprod,complement)
	 complement=map(lambda x:x.dot(x.T),complement)
         # Harmard elementwise product
	 gram=1.0
	 for j in range(N-1):
	    gram=gram*complement[j]
	 # T
	 tmat=tensorSubs.tensor_matricization(tensor,shape,i)[1]
	 trp=numpy.einsum('rp,qp->rq',ctensor,tmat)
	 # >>> Least square
	 # gram=scipy.linalg.pinv(gram,rcond=1.e-10)
	 # ainew=gram.dot(trp)
	 ainew=numpy.linalg.lstsq(gram,trp)[0]
	 cp_core=map(lambda x: numpy.linalg.norm(x),ainew)
	 ainew=map(lambda x: x*1.0/numpy.linalg.norm(x),ainew)
	 cp_site[i]=numpy.array(ainew) 
      # update
      it=it+1
      cp=(cp_core,cp_site)
   # check
   if info !=1 and error>thresh:
      print "Maxiter is reached!"
      info=2
   return info,errorlst,cp

if __name__ == '__main__':
   tensor=numpy.zeros((2,2,2))
   tensor[0,0,0]=1.0
   tensor[0,1,0]=1.0
   tensor[1,0,0]=1.0
   tensor[1,1,1]=1.0
   tensor=tensor.reshape(8)
   #cp_als(tensor,[2,2,2],400,-1.e-10,2)
   toCP(tensor,[2,2,2],1.e-14,plot=True)

#
# !!! the ALS algorithm is really BAD for determinants. !!!
#
# #==========================================
# # toDet
# #==========================================
# def toDet(tensor,nsorb,nelec,thresh=1.e-10,**kwargs):
#    print '\n[toDet] ALS algorithm for Det form'
#    #
#    # T[p1,p2,p3] => sum_r lambda[r]A1[r,p3]A2[r,p2]A3[r,p3]
#    #
#    # BEGIN
#    maxiter=1000
#    error  =1.0
#    rank   =1
#    dets   =(0,0)
#    tensor2=tensor.copy()
#    info,errorlst,dets=det_als(tensor2,nsorb,nelec,maxiter,thresh,rank,dets)
#    #while error>thresh: 
#    #   info,errorlst,dets=det_als(tensor2,nsorb,nelec,maxiter,thresh,rank,dets)
#    #   if info == 0: break
#    #   rank=rank+1
#    # Final
#    print '\nFinal CP info:'
#    print 'Rank_Det:',len(dets[0])
#    print 'Det_coef:\n',dets[0]
#    print 'Det_vals:\n',dets[1]
#    detlib.det_check(tensor,nsorb,nelec,dets,1)
#    # Plot
#    if kwargs.get('plot')==True:
#       import matplotlib.pyplot as plt
#       plt.plot(range(len(errorlst)),numpy.log10(errorlst))
#       plt.show()
#    return dets
# 
# 
# import detlib
# import itools
# def det_als(tensor,nsorb,nelec,maxiter,thresh,rank,seed=(0,0)):
#    print '\n[det_als] bdim=',rank,' maxiter=',maxiter 
#    det_core=[]    
#    det_site=[]
#    # Initialize (CI-coeff,CI-configuration) 
#    det_core=numpy.array([0.01]*rank)
#    imat=numpy.identity(nsorb)
#    icounter=0   
#    # Stored as (N,K)
#    for i in itools.combinations(range(nsorb),nelec):
#       det_site.append(imat[list(i)])
#       icounter=icounter+1
#       if icounter==rank: break
#    if isinstance(seed[0],numpy.ndarray):
#       r0=len(seed[0])
#       det_core[:r0]=seed[0][:r0]
#       det_site[:r0]=seed[1][:r0]
#    norm=numpy.linalg.norm(det_core)
#    det_core=det_core/norm		   
#    dets=(det_core,det_site)
#    #
#    # Begin   
#    #
#    info=0
#    it  =0
#    errorlst=[]
#    while it <= maxiter:
#       error=detlib.det_check(tensor,nsorb,nelec,dets,0)
#       print '--- it=',it,' error=',error,' coeff=',det_core
#       errorlst.append(error)
#       if error<=thresh: 
#          print "ALS converged!"
# 	 info=0
# 	 break
#       if it >=1 and abs(errorlst[-1]-errorlst[-2])<1.e-2*error:
# 	 print "No enough decrease!"
# 	 info=1
# 	 break
#       iden=numpy.identity(nsorb)
#       for r in range(rank-1,rank):
# 	 # For r-th Determinant, search c[r]
#          print ' >>> r = ',r,' of R-dets = ',rank 
# 	 res=range(rank)
# 	 res.remove(r)
# 	 cr_old = 1.0
# 	 micro_iter  =0
# 	 micro_error =1.0
# 	 micro_thresh=1.e-6
# 	 site = det_site[r].copy() #(N,K)matrix
# 	 while micro_error>micro_thresh:
# 	    print '  micro_iter=',micro_iter,' diff=',micro_error
# 	    micro_iter+=1
# 	    # DMRG-sweep for sites	
# 	    for j in range(nelec):
# 	       vec=numpy.zeros(nsorb)
#       	       # <P|R> 
# 	       ic=0
#       	       for statep in itools.combinations(range(nsorb),nelec):
# 	          matrix = site[:,statep]
# 		  # cofactor[i]*x[i,mu]
# 	          cofactor = map(lambda x:detlib.det_cofactor(matrix,x,j),range(nelec))
# 		  vec += tensor[ic]*numpy.einsum('i,im->m',cofactor,iden[list(statep)])
# 		  ic = ic + 1
# 	       # <R'|R>
# 	       for rp in res:
# 	          matrix = det_site[rp].dot(site.T)
# 	          cofactor = map(lambda x:detlib.det_cofactor(matrix,x,j),range(nelec))
# 		  vec += det_core[rp]*numpy.einsum('i,im->m',cofactor,det_site[rp])
# 	       # Q|v>=(I-P')|v>
# 	       pindx=range(nelec)
# 	       pindx.remove(j)
# 	       Qv = vec - site[pindx].T.dot( site[pindx].dot(vec) )
# 	       Qv = Qv/numpy.linalg.norm(Qv)
# 	       site[j] = Qv.copy()
# 	       cr_new = numpy.dot(Qv,vec)
# 	       #print 'det',numpy.linalg.det(site.dot(site.T))
# 	       print '   orb_j: ',j,' cr_new: ',cr_new
# 	    # DIFF   
# 	    micro_error=abs(cr_new-cr_old)
# 	    cr_old=cr_new
#          det_core[r]=cr_new
# 	 det_site[r]=site.copy()
#       # update
#       it=it+1
#       dets=(det_core,det_site)
#    # check
#    if info !=1 and error>thresh:
#       print "Maxiter is reached!"
#       info=2
#    return info,errorlst,dets
# 
# #==========================================
# # toCPdet
# #==========================================
# import tensorRep
# from pyscf import fci
# def cpdet_check(tensor,shape,cp,iprt):
#    nsorb=shape[0]
#    nelec=len(shape)
#    cp_core=cp[0]
#    cp_site=cp[1]
#    rank =len(cp[0])
#    civec0=tensorRep.toCItensorReverse(nsorb,nelec,tensor,1)
#    civec1=numpy.zeros(civec0.size)
#    for r in range(rank):
#       stateR = numpy.array([cp_site[i][r] for i in range(nelec)])
#       for i in range(civec0.size):
#          bstring=bin(tensorRep.addr2str_o1(nsorb,nelec,i))
#          orblst=list(tensorRep.bit2string( int(bstring,2),nsorb ))
# 	 civec1[i]+=numpy.linalg.det(stateR[:,orblst])*cp_core[r]
#    norm1=numpy.linalg.norm(civec0)
#    norm2=numpy.linalg.norm(civec1)
#    diff =numpy.linalg.norm(civec1-civec0)
#    if iprt>0:
#       print 'norm1 :',norm1
#       print 'norm2 :',norm2
#       print 'cp_core\n',cp_core
#       print 'cp_site\n',cp_site
#       print 'diff  :',diff
#       print 'CIVEC0:',civec0
#       print 'CIVEC1:',civec1
#    return diff
# 
# def toCPdet(tensor,shape,thresh=1.e-10,**kwargs):
#    print '\n[toCPdet] alternating least square (ALS) algorithm'
#    #
#    # T[p1,p2,p3] => sum_r lambda[r]A1[r,p3]A2[r,p2]A3[r,p3]
#    #
#    rank=len(shape)
#    print 'shape=',shape,' size= ',tensor.shape,' rank=',rank
#    if reduce(lambda x,y:x*y,shape) != tensor.size:
#       print 'ERROR: inconsistent size!'
#       exit(1)
#    # BEGIN
#    maxiter=1000
#    error  =1.0
#    rank   =1
#    cp     =(0,0)
#    tensor2=tensor.copy()
#    K=shape[0]
#    N=len(shape)
#    while error>thresh and rank<=misc.binomial(K,N): 
#       info,errorlst,cp=cpdet_als(tensor2,shape,maxiter,thresh,rank,cp)
#       if info == 0: break
#       rank=rank+1
#    # Final
#    print '\nFinal CP info:'
#    print 'Phys dim:',shape
#    print 'Rank_CP :',len(cp[0])
#    print 'CP_vals :',cp[0]
#    cpdet_check(tensor,shape,cp,1)
#    # Plot
#    if kwargs.get('plot')==True:
#       import matplotlib.pyplot as plt
#       plt.plot(range(len(errorlst)),numpy.log10(errorlst))
#       plt.show()
#    return cp
# 
# def cpdet_als(tensor,shape,maxiter,thresh,rank,seed=(0,0)):
#    print '\n[cpdet_als] bdim=',rank,' maxiter=',maxiter 
#    cp_core=[]    
#    cp_site=[]
#    K=shape[0]
#    N=len(shape)
#    # Initialization
#    # Stored as (N,K)
#    det_site=[]
#    imat=numpy.identity(K)
#    icounter=0
#    for i in itools.combinations(range(K),N):
#       det_site.append(imat[list(i)])
#       icounter=icounter+1
#       if icounter==rank: break
#    # (N,R,K) from (R,N,K)
#    cp_site=numpy.array(det_site).transpose(1,0,2)
#    cp_core=numpy.array([1.e-3]*rank)
#    if isinstance(seed[0],numpy.ndarray):
#       r0=len(seed[0])
#       cp_core[:r0]=seed[0][:r0]
#    cp_core=cp_core/numpy.linalg.norm(cp_core)
#    cp=(cp_core,cp_site)
# #
# # WITHOUT ORTHOGONALITY, EASY TO GET INTO STUCK !!!
# # cp_core
# # [ 0.98353007  0.03505464  0.03505464  0.03505464  0.03505464  0.03505464]
# # cp_site
# # [[[  1.00000000e+00   9.16005289e-15  -7.24530774e-13   3.81422334e-13]
# #   [ -4.71560453e-12  -1.74166883e-11   7.82356633e-21   1.00000000e+00]
# #   [ -3.01496926e-13   4.87193399e-11  -1.00000000e+00  -9.17296854e-27]
# #   [  4.85172280e-12  -2.92917981e-12  -3.52484838e-24   1.00000000e+00]
# #   [  1.43213976e-13  -1.90034779e-11  -1.00000000e+00  -1.35261713e-26]
# #   [  1.45804419e-13  -1.90033408e-11  -1.00000000e+00   1.79710852e-26]]
# # 
# #  [[ -8.42646188e-16   1.00000000e+00   2.79821833e-12   6.70168718e-12]
# #   [  2.95304009e-10  -1.34209336e-10   1.00000000e+00  -8.81893348e-21]
# #   [ -1.29063851e-09  -8.33575386e-12  -1.53515202e-23   1.00000000e+00]
# #   [ -1.07081968e-10   1.34222173e-10   1.00000000e+00   9.25561719e-22]
# #   [  6.05973309e-10   4.14152656e-12   8.23925409e-24   1.00000000e+00]
# #   [  6.05944805e-10   4.21420678e-12   7.28977574e-24   1.00000000e+00]]]
# #
#    # Begin  
#    info=0
#    it  =0
#    errorlst=[]
#    iden=numpy.identity(K)
#    stride=[K**(N-i-1) for i in range(N)]
#    while it <= maxiter:
#       error=cpdet_check(tensor,shape,cp,0)
#       print '--- it=',it,' error=',error
#       #print ' cp_core=',cp_core
#       errorlst.append(error)
#       if error<=thresh: 
#          print "ALS converged!"
# 	 info=0
# 	 break
#       if it >=1 and abs(errorlst[-1]-errorlst[-2])<1.e-2*error:
# 	 print "No enough decrease!"
# 	 info=1
# 	 break
#       # solve least square problem
#       for j in range(N):
# 
#          # v_{rm}[i]
#          vec=numpy.zeros((rank,K))
# 	 for r in range(rank):
#             stateR = numpy.array([cp_site[i][r] for i in range(N)]).transpose(1,0)
#             for stateP in itools.combinations(range(K),N):
#  	       matrix=stateR[stateP,:]
#     	       # cofactor[i]*x[i,mu]
#                cofactor = map(lambda x:detlib.det_cofactor(matrix,x,j),range(N))
# 	       addr = numpy.dot(list(stateP),stride)    
#   	       vec[r] += tensor[addr]\
# 		       * numpy.einsum('i,im->m',cofactor,iden[list(stateP)])
# 	 vec=vec*math.sqrt(math.factorial(N))
# 	 trp=vec.reshape(rank*K)
# 
#          # GRAM
#          gram=numpy.zeros((rank,rank,K,K))
# 	 nres=range(N)
# 	 nres.remove(j)
# 	 for rp in range(rank):
#             stateRp = numpy.array([cp_site[i][rp] for i in range(N)])
# 	    for r in range(rank):
#                stateR = numpy.array([cp_site[i][r] for i in range(N)])
# 	       matrix = stateRp.dot(stateR.T)
# 	       # G(r',r)[jk,ij] 
# 	       gki =numpy.zeros((N-1,N-1))
# 	       for ic in range(N-1):
# 		  i=nres[ic]
# 		  rows=range(N)
# 		  rows.remove(i)
# 	          matij=matrix[numpy.array(rows)[:,numpy.newaxis],
# 	   	               numpy.array(nres)]
# 		  jdx=rows.index(j)
# 		  rows2=range(N-1)
# 		  rows2.remove(jdx)
# 		  for kc in range(N-1):
# 	 	     cols=range(N-1)
# 		     cols.remove(kc)
# 		     if rows2==[] and cols==[]:
# 			matjkij=numpy.identity(1)     
# 		     else:
# 		        matjkij=matij[numpy.array(rows2)[:,numpy.newaxis],
# 				   numpy.array(cols)]
# 		     # 		index in old mat, index in new matij 
# 		     gki[kc,ic]=(-1)**(i+j)*(-1)**(jdx+kc)*numpy.linalg.det(matjkij)
#                # product
# 	       vecR =[stateR[k] for k in nres]
# 	       vecRp=[stateRp[i] for i in nres]
# 	       gram[rp,r]+=numpy.einsum('km,ki,in->mn',vecR,gki,vecRp)
# 	       gram[rp,r]+=detlib.det_cofactor(matrix,j,j)*numpy.identity(K)
# 
# 	 # Solve least square problem
# 	 # G[r'm,rn]
#          gram=gram.transpose(0,2,1,3).reshape((rank*K,rank*K))
# 	 gram=gram+1.e-3*numpy.identity(rank*K)
# 	 #print numpy.linalg.norm(gram-gram.T)
# 	 xrm =numpy.linalg.lstsq(gram,trp)[0]
# 	 xrm =xrm.reshape((rank,K))
# 	 cp_core=numpy.array(map(lambda x: numpy.linalg.norm(x),xrm))
# 	 xrm =map(lambda x: x*1.0/numpy.linalg.norm(x),xrm)
# 	 cp_site[j]=numpy.array(xrm)
# 
#       # update
#       it=it+1
#       cp=(cp_core,cp_site)
#    # check
#    if info !=1 and error>thresh:
#       print "Maxiter is reached!"
#       info=2
#    return info,errorlst,cp
