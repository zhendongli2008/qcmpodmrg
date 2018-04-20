#=======================================================
#
# The efficiency depends on the primitive basis 
# adopted in the cut. The naive slater det basis may 
# not be good enough. Entanglement based decomposition
# is better, although the overlap between two determinants
# with nonorthogonal orbitals needs to be computed.
#
# Flow:
#   (1) UHF=>MPS=>P*MPS (A poor man's PUHF???)
#   (2) UHF=>H*MPS("MRCISD")=>P*MPS' (Size-extensive error)
#
#=======================================================
import numpy
import scipy

#----------------------------
# Mayer's orthogonalization!
#----------------------------
#def genU(vec):
#   n = vec.shape[0]
#   u = numpy.zeros((n,n))
#   u[:,0] = vec
#   u[0,1:] = -vec[1:]
#   fac = -1.0/(1.0+vec[0])
#   # row
#   for i in range(1,n):
#      u[i,i] = 1.0
#      # col
#      for j in range(1,n):
#         u[i,j] += fac*vec[i]*vec[j]
#   return u
#
#------------------------------
# Single orbital decomposition
#------------------------------
#def genDecomp(info,thresh=1.e-10):
#   idx,wfac,vmat = info
#   # Some simple cases:
#   empty = numpy.array([])
#   dims = vmat.shape
#   if len(dims)<2: return [[idx,0,1.0,wfac,empty]]
#   k,n = dims 
#   if n == 0: return [[idx,0,1.0,wfac,empty]]
#   if k == 1 and n == 1:
#      return [[idx,1,vmat[0,0],wfac,empty]]
#   if k == n and k > 1: 
#      sgn = numpy.linalg.det(vmat)
#      return [[idx,1,sgn,wfac,numpy.identity(k-1)]]
#   elif k > 1 and n == 1:
#      norm1 = abs(vmat[0,0])
#      vb = vmat[1:,0]
#      norm2 = numpy.linalg.norm(vb)
#      if norm2>thresh: vmat[1:,0] = vb/norm2
#      return [[idx,1,norm1,wfac,empty],\
#              [idx,0,norm2,wfac,vmat[1:,0]]]
#   elif k == 1 and n > 1:
#      print 'error: n>k=1'
#      print k,n,vmat
#      exit(1)
#   # General
#   norm1 = numpy.linalg.norm(vmat[0])
#   if norm1<thresh:
#      # In this case, u = I. 
#      vmat2 = vmat
#      sgn = 1.0
#   else:	   
#      v0 = vmat[0]/norm1
#      u = genU(v0)
#      # faster computation?
#      sgn = numpy.linalg.det(u)
#      vmat2 = vmat.dot(u)
#   vb = vmat2[1:,0]
#   norm2 = numpy.linalg.norm(vb)
#   if norm2>thresh: vmat2[1:,0] = vb/norm2
#   return [[idx,1,sgn*norm1,wfac,vmat2[1:,1:]],\
#	   [idx,0,sgn*norm2,wfac,vmat2[1:,:]]]
#
#------------------------------
# Binary Tree Construction
#------------------------------
#def tree(vmat,thresh=1.e-8,debug=True):
#   import itertools
#   print '\n[detToMPS.tree]'
#   k,n = vmat.shape
#   states = [[0,1.0,vmat]]
#   bdims = [1]
#   sites = []
#   for irow in range(k):
#      # Key step to perform the decomposition
#      nstates = map(lambda x: genDecomp(x),states)
#      lst = list(itertools.chain(*nstates))
#      wts = numpy.array(map(lambda x:abs(x[2]*x[3]),lst))
#      #Without pruning, the size of the tree increases exponentially (2^n).
#      indices = numpy.argwhere(wts>thresh)
#      nstate = len(indices)
#      print ' irow =',irow,' nstate =',nstate
#      if debug: print ' wts =\n',wts[indices]
#      assert nstate>0
#      bdims.append(nstate)
#      atmp = numpy.zeros((bdims[-2],2,bdims[-1])) 
#      states = []
#      for idx in range(nstate):
#	 info = lst[indices[idx]]
#	 #if debug: print info[0],info[1],idx,info[2],info[2]*info[3],\
#	 #       	 info[4].shape,info[4]
#	 atmp[info[0],info[1],idx] = info[2]
#	 prodWfac = info[2]*info[3]
#         states += [[idx,prodWfac,info[4]]]
#      if irow == 0:
#	 shape = atmp.shape     
#	 atmp = atmp.reshape(shape[1],shape[2]) 
#      elif irow == k-1: 
#	 atmp = numpy.einsum('ipj->ip',atmp)
#      sites.append(atmp)
#   print ' Bdims =',mpslib.mps_bdim(sites)
#   return sites

#------------------------------------------------------------
# Generator for Fock space states: 
#    the threshold is in fact for the Schidmt value, 
#    similar to the square root of discarded weight in DMRG.
#------------------------------------------------------------
def genFSstates(wts,state=None,thresh=1.e-10):
   n = len(wts)
   occ = 1
   vir = 0
   if state is None and n == 0: yield []
   if state is None:
      if abs(wts[0][0])>thresh:
         for newstate in genFSstates(wts[1:],[wts[0][0],[occ]],thresh):
            yield newstate
      if abs(wts[0][1])>thresh:
         for newstate in genFSstates(wts[1:],[wts[0][1],[vir]],thresh):
            yield newstate
   else:
      if n == 0:
         iwts,istate = state
	 if abs(iwts)>thresh: yield state
      else:
         iwts,istate = state
         if abs(iwts*wts[0][0])>thresh:
            for newstate in genFSstates(wts[1:],[iwts*wts[0][0],istate+[occ]],thresh):
               yield newstate
         if abs(iwts*wts[0][1])>thresh:
            for newstate in genFSstates(wts[1:],[iwts*wts[0][1],istate+[vir]],thresh):
               yield newstate

#------------------------------------------------------------
# Classfication of intermediates
#------------------------------------------------------------
def classification(iterator,debug=False):
   if debug: print '\n[detToMPS.classification]'
   dic = {}
   for info in iterator:
      wts,state = info
      n = numpy.sum(state)
      dic.setdefault(n,[]).append(info)
   dim = 0
   psum = 0.0
   dic2 = {}
   for key in dic.keys():
      dic2[key] = map(lambda x:x[1],dic[key])
      dim0 = len(dic[key])
      dim += dim0
      wts = numpy.sum(map(lambda x:x[0]**2,dic[key]))
      psum += wts
      if debug: print ' ne=',key,' dim=',dim0,' wts=',wts
   if debug: print ' total=',dim,' wts=',psum
   return dim,dic2

#------------------------------------------------------------
# Determinant network:
#   The width is theoretically controled by 2^min[N,(K-N)].
#   In pratice, by choosing an appropriate basis that minimizes
#   the entanglement in the given state, the width of the graph
#   can be controlled.
#
# The goal of such network is to facilitate the construction
# of MPS representation in a given basis, which provides a 
# compact Fock space representation of the wave function.
#------------------------------------------------------------
def network(vmat,thresh=1.e-8,threshVal=1.e-8,debug=False,ifclass=False,ifbdim=False):
   k,n = vmat.shape
   print '\n[detToMPS.network] threshVal = %7.1e'%threshVal,' (k,n) =',(k,n)
   if k == 1:
      print ' error: k = 1' 
      exit(1)
   #
   # Construct networks from DET[vmat].
   #
   print ' >>> 1. Generate impurity Fock space:'
   commonBasis = []
   intermediates = {}
   keff = k
   for irow in range(k):
      nf = irow+1
      nc = k-nf
      tmp = vmat[:irow+1].copy()
      try:
	 u,sigs,vt = scipy.linalg.svd(tmp)
         #print sigs
      except numpy.linalg.linalg.LinAlgError:
	 print '\nerror: numpy.linalg.linalg.LinAlgError'     
	 print tmp.shape
	 print tmp.dump('tmpMatrix')
         exit()
      #>>> The global sign is not important in screening! 
      #sgn = numpy.linalg.det(vt)
      sgn = 1.0
      #>>> Construct the new representation: Ut*A*V, Ut=blk(Ut,I)
      # Sigma
      rf = numpy.zeros((nf,n))
      nsigs = len(sigs)
      assert nsigs == min(nf,n)
      rf[range(nsigs),range(nsigs)] = sigs
      # C*V
      rc = vmat[irow+1:].dot(vt.T)
      rmat = numpy.vstack((rf,rc))
      #>>> Transform to a data structure [o1,o2,...,oN] 
      weights = []
      for i in range(nsigs):
	 cf = sgn*sigs[i]
	 vr = rc[:,i]
	 norm = numpy.linalg.norm(vr)
	 if norm>thresh: 
	    cr = sgn*norm
	 else:
	    cr = 0.0
	 # Combination coefficients   
 	 weights.append([cf,cr])
      # Generate Fock space states.
      ispace = genFSstates(weights,thresh=threshVal)
      dim,dic = classification(ispace,ifclass)
      if debug:
         print ' irow =',irow
         print '   > nsigs =',nsigs
         print '   > (cfrag,cbath) = ',weights
         print '   > ucoeff =',u[:,:nsigs]
         print '   > dim(Lspace) =',dim
         print '   > generated L-fock space =',dic
      else:
         print ' irow =',irow,' nsigs =',nsigs,' dim(Lspace) =',dim
      # Only ON states are stored.
      intermediates[irow] = [dim,dic]
      commonBasis.append(u[:,:nsigs])
      # Only for testing
      if ifbdim and dim == 1: 
	 keff = irow
	 break
   print
   print ' Summary: intermediate states on each layer (bdims) for keff/k =',(keff,k)
   print [intermediates[i][0] for i in range(keff)]
   print
   if ifbdim: return 0
   #
   # |Psi> = |U*Sig*Vt> = |Phi(U)>*det(Vt)
   # Sig = [1.0]*nelec in the last case (sig>0).
   #
   sgn = numpy.linalg.det(vt)
   #
   # Compute the possible overlaps
   #
   print ' >>> 2. Compute site tensor:'
   sites = []
   for irow in range(k):
      u1 = commonBasis[irow]
      dim1,space1 = intermediates[irow]
      #
      # <n[1]|a[1]>
      #
      if irow == 0:
	 site = numpy.zeros((2,dim1))
	 idx = 0
	 for isym in space1.keys():
	    ispace = space1[isym]
	    ndim = len(ispace)
	    assert ndim == 1
	    istate = ispace[0]
	    # <n1=0|a1>
	    if isym == 0:
	       site[0,idx] = 1.0
	    # <n1=1|a1>
	    elif isym == 1:
	       site[1,idx] = u1[0,0]
	    idx += 1
      #
      # <a[k-1]n[k]|a[k]>     
      #
      else:
	 u0 = commonBasis[irow-1]
         dim0,space0 = intermediates[irow-1]
	 site = numpy.zeros((dim0,2,dim1))
	 idx0 = 0
	 for isym0 in space0.keys():
	    ispace0 = space0[isym0]
	    ndim0 = len(ispace0)
	    idx1 = 0
	    for isym1 in space1.keys():
	       ispace1 = space1[isym1]
	       ndim1 = len(ispace1)
	       #
	       # Case1: <a[k-1]n[k](=0)|a[k]>
	       #
	       if isym1 == isym0:
		  for id0,istate0 in enumerate(ispace0):
		     orbs0 = numpy.argwhere(numpy.array(istate0)==1)
		     orbs0 = map(lambda x:x[0],orbs0)
		     umat0 = numpy.vstack((u0[:,orbs0],numpy.zeros((1,isym1))))
	             for id1,istate1 in enumerate(ispace1):
			orbs1 = numpy.argwhere(numpy.array(istate1)==1)
			orbs1 = map(lambda x:x[0],orbs1)
			umat1 = u1[:,orbs1]
			if isym0 == 0:
			   site[idx0+id0,0,idx1+id1] = 1.0
		 	else:
			   s01 = numpy.dot(umat0.T,umat1)
			   site[idx0+id0,0,idx1+id1] = numpy.linalg.det(s01)
	       #
	       # Case2: <a[k-1]n[k](=1)|a[k]>
	       #
       	       elif isym1 == isym0+1:
		  for id0,istate0 in enumerate(ispace0):
		     orbs0 = numpy.argwhere(numpy.array(istate0)==1)
		     orbs0 = map(lambda x:x[0],orbs0)
		     nelec0 = len(orbs0) 
		     umat0 = numpy.zeros((irow+1,nelec0+1))
		     umat0[:irow,:nelec0] = u0[:,orbs0]
		     umat0[ irow, nelec0] = 1.0
	             for id1,istate1 in enumerate(ispace1):
			orbs1 = numpy.argwhere(numpy.array(istate1)==1)
			orbs1 = map(lambda x:x[0],orbs1)
			umat1 = u1[:,orbs1]
			s01 = numpy.dot(umat0.T,umat1)
			site[idx0+id0,1,idx1+id1] = numpy.linalg.det(s01)
	       idx1 += ndim1
	    idx0 += ndim0
      # Last site
      if irow == k-1:
         s = site.shape
	 assert s[2] == 1
	 site = sgn*site.reshape(s[0],s[1])
      print ' irow =',irow,' isite.shape =',site.shape
      sites.append(site)
   return sites

#------------------------------------------------------------
# Merge
#------------------------------------------------------------
def detMerge(vmata,vmatb):
   k,na = vmata.shape
   k,nb = vmatb.shape
   assert na >= nb
   nt = na+nb
   vmat = numpy.zeros((2*k,nt))
   #vmat[1::2,1:2*nb:2] = vmatb[:,:nb]
   #vmat[::2,:2*nb:2] = vmata[:,:nb]
   #vmat[::2,2*nb:] = vmata[:,nb:]
   vmat[::2,:na] = vmata
   vmat[1::2,na:] = vmatb
   return vmat

#------------------------------------------------------------
# Unrestricted case
#------------------------------------------------------------
def unetwork(vmata,vmatb,thresh=1.e-8,threshVal=1.e-8,debug=False,ifclass=False):
   print '\n[detToMPS.unetwork]'
   # >>> a,b,a,b ordering
   vmat = detMerge(vmata,vmatb)
   sites = network(vmat,thresh,threshVal,debug,ifclass)
   k,na = vmata.shape
   k,nb = vmatb.shape
   msites = [0]*k
   # site[0]
   #tmp = numpy.einsum('pa,aqb->pqb',sites[0],sites[1])
   tmp = numpy.tensordot(sites[0],sites[1],axes=([1],[0]))
   s = tmp.shape
   msites[0] = tmp.reshape(s[0]*s[1],s[2])
   # site[k-1]
   #tmp = numpy.einsum('apb,bq->apq',sites[-2],sites[-1])
   tmp = numpy.tensordot(sites[-2],sites[-1],axes=([2],[0]))
   s = tmp.shape
   msites[-1] = tmp.reshape(s[0],s[1]*s[2])
   # sites[1]-sites[k-2]
   for k in range(1,k-1):
      #tmp = numpy.einsum('apb,bqc->apqc',sites[2*k],sites[2*k+1])
      tmp = numpy.tensordot(sites[2*k],sites[2*k+1],axes=([2],[0]))
      s = tmp.shape
      msites[k] = tmp.reshape(s[0],s[1]*s[2],s[3])
   return msites

#------------------------------------------------------------
# Unrestricted case: Merge MPSa & MPSb
#------------------------------------------------------------
def unetworkSplit(vmata,vmatb,thresh=1.e-8,threshVal=1.e-8,debug=False,ifclass=False):
   print '\n[detToMPS.unetworkSplit]'
   k,na = vmata.shape
   k,nb = vmatb.shape
   sitesA = network(vmata,thresh,threshVal,debug,ifclass,ifbdim=True)
   sitesB = network(vmatb,thresh,threshVal,debug,ifclass,ifbdim=True)
   print k,na
   print k,nb
   exit()
   msites = [0]*k
   # site[0]
   #tmp = numpy.einsum('pa,aqb->pqb',sites[0],sites[1])
   tmp = numpy.tensordot(sites[0],sites[1],axes=([1],[0]))
   s = tmp.shape
   msites[0] = tmp.reshape(s[0]*s[1],s[2])
   # site[k-1]
   #tmp = numpy.einsum('apb,bq->apq',sites[-2],sites[-1])
   tmp = numpy.tensordot(sites[-2],sites[-1],axes=([2],[0]))
   s = tmp.shape
   msites[-1] = tmp.reshape(s[0],s[1]*s[2])
   # sites[1]-sites[k-2]
   for k in range(1,k-1):
      #tmp = numpy.einsum('apb,bqc->apqc',sites[2*k],sites[2*k+1])
      tmp = numpy.tensordot(sites[2*k],sites[2*k+1],axes=([2],[0]))
      s = tmp.shape
      msites[k] = tmp.reshape(s[0],s[1]*s[2],s[3])
   return msites


if __name__ == '__main__':
   
   from mpo_dmrg.source import mpo_class
   from mpo_dmrg.source.tools import mpslib

   weights = [[1.0,0.0]]
   gen = genFSstates(weights)
   print '*** Fock space basis (idx) = [wts,state] ***' 
   for idx,i in enumerate(gen):
      print idx,i
   
   weights = [[0.0,1.0]]
   gen = genFSstates(weights)
   print '*** Fock space basis (idx) = [wts,state] ***' 
   for idx,i in enumerate(gen):
      print idx,i
   
   weights = [[1.0,0.0],[1.0,0.0]]
   gen = genFSstates(weights)
   print '*** Fock space basis (idx) = [wts,state] ***' 
   for idx,i in enumerate(gen):
      print idx,i
   
   weights = [[1.0,0.0],[0.0,1.0]]
   gen = genFSstates(weights)
   print '*** Fock space basis (idx) = [wts,state] ***' 
   for idx,i in enumerate(gen):
      print idx,i
   
   weights = [[0.3,0.4],[0.2,0.8]]
   gen = genFSstates(weights)
   print '*** Fock space basis (idx) = [wts,state] ***' 
   for idx,i in enumerate(gen):
      print idx,i

   weights = [[0.5,0.5],[0.5,0.5],[0.5,0.6],[0.5,1],[0.5,1],[0,1],[0.1,1],[0.1,0.2],[0.2,0.3],[0.1,0.2]]
   gen = genFSstates(weights)
   print '*** Fock space basis (idx) = [wts,state] ***' 
   for idx,i in enumerate(gen):
      print idx,i

   # Init: 
   # [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
   #nb = 20
   #ne = 10
   nb = 4
   ne = 3
   numpy.random.seed(2)
   h=numpy.random.uniform(-1,1,nb*nb)
   h=h.reshape(nb,nb)
   h=0.5*(h+h.T)
   e,v = scipy.linalg.eigh(h)
   vmat = v[:,:ne].copy()
   mps2 = network(vmat,threshVal=1.e-4)
   # Test
   mps = mpo_class.detToMPS(vmat)
   print '\n*** Comparison ***'
   bdim = mpslib.mps_bdim(mps)
   print 'bdim=',bdim
   bdim2 = mpslib.mps_bdim(mps2)
   print 'bdim2=',bdim2
   rr = mpslib.mps_diff(mps,mps2,iprt=1)
   print 'diff=',rr

   # SVD test:
   #vmat[1::2,::2] = 0.0
   #vmat[::2,1::2] = 0.0
   u,sig,vt = scipy.linalg.svd(vmat)
   print 'vmat\n',vmat
   print 'u\n',u
   print 'vt\n',vt
   print 'sig\n',sig
   #
   # SVD do not mix alpha and beta in U and V.
   #
   # vmat
   # [[ 0.56640333  0.         -0.36372841]
   #  [ 0.         -0.53847908  0.        ]
   #  [ 0.39211494  0.          0.79453743]
   #  [ 0.          0.53600291  0.        ]]
   # u
   # [[ 0.19041698  0.          0.9817033   0.        ]
   #  [ 0.          0.70873445  0.          0.70547536]
   #  [-0.9817033   0.          0.19041698  0.        ]
   #  [ 0.         -0.70547536  0.          0.70873445]]
   # vt
   # [[-0.31017751  0.         -0.95067866]
   #  [-0.         -1.          0.        ]
   #  [ 0.95067866  0.         -0.31017751]]
   # sig
   # [ 0.89331982  0.75977552  0.66342644]
   #
   vmat2 = detMerge(vmat[:,:3],vmat[:,:1])
   mps2 = network(vmat2,threshVal=1.e-2)
   # Test
   mps = mpo_class.detToMPS(vmat2)
   print '\n*** Comparison ***'
   bdim = mpslib.mps_bdim(mps)
   print 'bdim=',bdim
   bdim2 = mpslib.mps_bdim(mps2)
   print 'bdim2=',bdim2
   rr = mpslib.mps_diff(mps,mps2,iprt=1)
   print 'diff=',rr
   
   # Unrestricted case
   vmata = vmat[:,:2]
   vmatb = vmat[:,:1]
   mps2 = unetwork(vmata,vmatb,threshVal=1.e-2)
   # Test
   print '\n*** Comparison ***'
   print 'shape=',map(lambda x:x.shape,mps2)
