import numpy
import scipy.linalg

#
# THIS IS ONE OF THE MOST IMPORTANT FUNCTIONS !!!
#
def row_svd(cimat,classes,thresh,Dcut):
   debug = False #True 
   if debug:
      print '\n[row_svd]'
      print ' CImat.shape = ',cimat.shape
   # 1. Symmetry counting using dictionary
   dic = {}
   for idx,val in enumerate(classes):
      dic.setdefault(str(val),[]).append(idx)
   if debug:
      print ' classes = ',classes
      for item in dic:
         print ' key = ',item,' idx = ',dic[item]
   # 2. Decomposition by symmetry sectors
   nrows,ncols = cimat.shape
   if debug: print ' nrows,ncols=',nrows,ncols
   nrowc = len(classes)
   if nrowc != nrows:
      print 'error: nrowc != nrows in row_svd:',nrowc,nrows
      exit(1)
   qsymL = []
   rotL = []
   rotR = []
   sigs = []
   # Defined for the eval functions
   inf = float('inf')
   for idx,item in enumerate(dic):
      rows = dic[item]
      qrow = cimat[rows,:]
      u, sig, v = scipy.linalg.svd(qrow, full_matrices=False)
      ndim  = len(sig) # min(dimL,dimR)
      if debug:
         print ' idx=%5d'%idx,' isym=',item,' ndiml/ndimr=',qrow.shape,' ndim=',ndim
         print '    sig=',sig
      key   = eval(item)
      qsymL = qsymL + [key]*ndim
      # Assemble block-u into the full matrix
      ucol  = numpy.zeros((nrows,ndim))
      ucol[rows,:] = u
      rotL.append(ucol)
      rotR.append(v)
      sigs.append(sig)
   # 3. Extract information
   rotL = numpy.hstack(rotL)
   rotR = numpy.vstack(rotR)
   sigs = numpy.hstack(sigs)
   if debug: print ' Sum of sig0^2 =',numpy.sum(sigs**2)
   # 
   # Truncation (decimation) according to the weights
   #
   indx = numpy.argsort(sigs)[-1::-1]
   # 
   # Determine the states to be retained
   #
   nres = len(sigs)
   # Adjusted bond dimension by the weights
   tsig = sigs[indx].copy()
   for i in range(nres):
      if(tsig[i]<thresh*1.01):
         nres=i
         break
   if nres==0: nres=1
   if Dcut>0: nres = min(nres,Dcut)
   indx = indx[:nres]
   sigs = sigs[indx].copy()
   dwts = 1.0-numpy.sum(sigs**2)
   qsymL= [qsymL[i] for i in indx]
   rotL = rotL[:,indx]
   rotR = rotR[indx,:]
   #
   # Resort qsymL
   #
   indx = sorted(range(nres),key=lambda x:qsymL[x])
   qsymL= [qsymL[i] for i in indx]
   sigs = sigs[indx].copy()
   rotL = rotL[:,indx].copy()
   rotR = rotR[indx,:].copy()
   #debug=True
   if debug:
      print ' Final of [row_svd] results:'
      print '  qsymL:',qsymL
      print '  truncated sigs :',sigs.shape,'\n',sigs
      print '  rotL :',rotL.shape
      print '  rotR :',rotR.shape
      print '  Sum of sig1^2 =',numpy.sum(sigs**2)
      print '  Discarded weights = ',dwts
      print
   return dwts,qsymL,rotL,sigs,rotR


# Blockwise diagonalization
def blk_diag(rho0,cimat,classes,thresh,Dcut,debug=False):
   if debug:
      print '\n[blk_diag]'
      print ' CImat.shape = ',cimat.shape
      print ' rho0.shape = ',rho0.shape
   # 1. Symmetry counting using dictionary
   dic = {}
   for idx,val in enumerate(classes):
      dic.setdefault(str(val),[]).append(idx)
   if debug:
      print ' classes = ',classes
      for item in dic:
         print ' key = ',item,' idx = ',dic[item]
   # 2. Decomposition by symmetry sectors
   neig,nrows,ncols = cimat.shape
   if debug: print ' nrows,ncols=',nrows,ncols
   qsymL = []
   rotL = []
   sigs = []
   # Defined for the eval functions
   inf = float('inf')
   for idx,item in enumerate(dic):
      rows = dic[item]
      qrow = rho0[numpy.ix_(rows,rows)]
      #sig2,u = scipy.linalg.eigh(qrow)
      sig2,u = numpy.linalg.eigh(qrow)
      # Renormalize
      sig2 = numpy.abs(sig2)
      if debug: 
	 print ' idx=%5d'%idx,' isym=',item,' ndiml/ndimr=',qrow.shape
	 print '    sig2=',sig2
      ndim  = len(sig2) # min(dimL,dimR)
      key   = eval(item)
      qsymL = qsymL + [key]*ndim
      # Assemble block-u into the full matrix
      ucol  = numpy.zeros((nrows,ndim),dtype=numpy.float_)
      ucol[rows,:] = u
      rotL.append(ucol)
      sigs.append(sig2)
   # 3. Extract information
   rotL = numpy.hstack(rotL)
   sigs = numpy.hstack(sigs)
   # Renormalize
   sigs = sigs/numpy.sum(sigs)
   if debug: print ' Sum of sig0^2 =',numpy.sum(sigs)
   # 
   # Truncation (decimation) according to the weights
   #
   indx = numpy.argsort(sigs)[-1::-1]
   # 
   # Determine the states to be retained
   #
   nres = len(sigs)
   # Adjusted bond dimension by the weights
   tsig = sigs[indx].copy()  
   for i in range(nres):
      if(tsig[i]<thresh*1.01):
	 nres=i
	 break
   if nres==0: nres=1 
   if Dcut>0: nres = min(nres,Dcut)
   indx = indx[:nres] 
   sigs = sigs[indx].copy()
   dwts = 1.0-numpy.sum(sigs)
   qsymL= [qsymL[i] for i in indx]
   rotL = rotL[:,indx]
   #
   # Resort by qsymL
   #
   indx = sorted(range(nres),key=lambda x:qsymL[x])
   qsymL= [qsymL[i] for i in indx]
   sigs = sigs[indx].copy()
   rotL = rotL[:,indx].copy()
   if debug:
      print ' Final of [blk_diag] results:'
      print '  len(qsymL):',len(qsymL)
      print '  qsymL:',qsymL
      print '  truncated sigs :',sigs.shape,'\n',sigs
      print '  rotL :',rotL.shape
      print '  Sum of sig1^2 =',numpy.sum(sigs)
      print '  Discarded weights = ',dwts
      print 
   #srotR = rotL.dagger.dot(cimat)
   srotR = numpy.einsum('ij,aix->ajx',rotL,cimat).copy()
   return dwts,qsymL,rotL,sigs,srotR
