#
# 2016.04.20 -lzd- Boundary case? full contraction & zero contraction
# 2016.04.13 -lzd- Optimized version
#
import time
import copy
import numpy
import itertools
import qtensor_util
import qtensor_idlst

#
# Sliced objects
#
class Qt:
   def __init__(self,slcdim=None,maxslc=None):
      self.nslc = None
      if slcdim is not None:
         self.slcdim = slcdim
	 if maxslc is not None:
	    self.maxslc = [maxslc]*slcdim
   	    self.genDic()
	 else:
            self.maxslc = [0]*slcdim
	    self.size   = None
      else:
 	 self.slcdim = None
	 self.maxslc = None
         self.size = None

   def prt(self):  
      print 'slcdim=',self.slcdim
      print 'maxslc=',self.maxslc
      print 'nslc  =',self.nslc 
      return 0

   def ravel(self,multi):
      return numpy.ravel_multi_index(multi,self.maxslc)

   def unravel(self,indx):
      return numpy.unravel_index(indx,self.maxslc)

   def genNslc(self):
      if self.nslc is None: self.nslc = numpy.prod(self.maxslc) 
      return 0

   def genDic(self):
      self.genNslc()
      self.size = [0]*self.nslc
      self.dic = dict([(i,qtensor()) for i in range(self.nslc)])
      return 0

   def dumpInfo(self,f1,name):
      grp = f1.create_group(name)
      grp.attrs['slcdim'] = self.slcdim
      grp.attrs['maxslc'] = self.maxslc
      grp.attrs['nslc']   = self.nslc
      grp.attrs['size']   = self.size
      return 0

   def dumpSLC(self,f1,name,idx):
      self.dic[idx].dump(f1,name+'_slc'+str(idx))
      return 0

   def dump(self,f1,name):
      self.dumpInfo(f1,name)
      for idx in range(self.nslc):
	 if self.size[idx] > 0: 
            self.dumpSLC(f1,name,idx)
      return 0

   def loadSLC(self,f1,name,idx):
      qt = qtensor()
      qt.load(f1,name+'_slc'+str(idx))
      return qt

   def loadInfo(self,f1,name):
      grp = f1[name]
      self.slcdim = grp.attrs['slcdim']  
      self.maxslc = grp.attrs['maxslc']  
      self.nslc   = grp.attrs['nslc']  
      self.size   = grp.attrs['size']
      return 0

   def load(self,f1,name):
      self.loadInfo(f1,name)
      self.dic = dict([(i,0) for i in range(self.nslc)])
      for idx in range(self.nslc):
	 if self.size[idx] > 0:
	    self.dic[idx] = self.loadSLC(f1,name,idx)
      return 0

   def diagH(self):
      diag = Qt()
      diag.slcdim = self.slcdim # 2
      diag.maxslc = self.maxslc # [1,3]
      diag.nslc   = self.nslc   # 3
      diag.size   = self.size   # [21,15,16]
      diag.dic = dict([(i,qtensor()) for i in range(self.nslc)])
      for idx in range(self.nslc):
         if diag.size[idx] > 0:
	    diag.dic[idx] = self.dic[idx].diagH()
      return diag

#
# Sparse blocks
#
class qtensor:
   def __init__(self,status=None):
      #------------------
      self.rank  = None	 	# int64  1 
      self.qsyms = None 	# list of numpy.arrays (float64)
      self.ndims = None  	# list of array containing dimensions in each sym.
			        #  ndims= [array([4, 1, 2, 2, 1, 2, 1, 1, 2]), array(...]
      #------------------
      self.nqnum = None		# (nsyms[i],nqnum) = qsyms[i]
      self.nsyms = None		# nsyms = [ 9  4 16] 
      self.shape = None  	# shape = [16  4 64]
      self.size  = None 	# size  = 4096=16*4*64
      self.nblks = None         # nblks = 576 =9*4*16
      #------------------
      # KEY for BLOCKS
      #------------------
      self.nblks_allowed = None # int64  1
      self.iblks_allowed = None # int32  nsyms
      self.dims_allowed  = None # int32  cartesian products of ndims (nblks) 
      self.idx_allowed   = None # int64  nblks
      self.ndim_allowed  = None # int64  nblks
      self.ioff_allowed  = None # int64  nblks
      #------------------
      self.size_allowed  = None # int64  1
      self.value   = None  	# dtype  size_allowed
      self.savings = None       # float64
      if status is None:
	 self.status = None
      else:
	 self.status = numpy.array(status)
      #------------------
      # For contractions
      #------------------
      self.axes = None
      self.clst = None
      #------------------
      # Optional
      #------------------
      self.idlst = None # list of list
        	        #  idlst= [[[6, 7, 8, 9], [5], [11, 12], [3, 4], [0], ...]
      #------------------

   def copy(self,other):
      #------------------
      self.rank  = other.rank   
      self.nqnum = other.nqnum
      self.size  = other.size
      self.nblks = other.nblks
      self.nblks_allowed = other.nblks_allowed
      self.size_allowed  = other.size_allowed
      self.qsyms = [0]*self.rank
      self.ndims = [0]*self.rank
      self.nsyms = [0]*self.rank
      self.shape = [0]*self.rank
      for i in range(self.rank):
	 self.qsyms[i] = other.qsyms[i].copy()
	 self.ndims[i] = other.ndims[i].copy()
	 self.nsyms[i] = other.nsyms[i]
	 self.shape[i] = other.shape[i]
      self.iblks_allowed = other.iblks_allowed.copy()
      self.dims_allowed  = other.dims_allowed.copy() 
      self.idx_allowed   = other.idx_allowed.copy()  
      self.ndim_allowed  = other.ndim_allowed.copy() 
      self.ioff_allowed  = other.ioff_allowed.copy() 
      if other.savings is not None:
         self.savings = other.savings
      if other.value is not None: 
         self.value = other.value.copy()
      self.status = None
      if other.status is not None:
	 self.status = other.status.copy()
      #--------------------------------------
      # The following things are not copied. 
      #--------------------------------------
      self.axes  = None
      self.clst  = None
      self.idlst = None 
      #--------------------------------------

   #@profile
   def dump(self,f1,name):
      grp = f1.create_group(name)
      grp.attrs['rank']  = self.rank
      grp.attrs['nqnum'] = self.nqnum
      grp.attrs['nsyms'] = self.nsyms
      grp.attrs['shape'] = self.shape
      grp.attrs['size']  = self.size
      grp.attrs['nblks'] = self.nblks
      grp.attrs['nblks_allowed'] = self.nblks_allowed  # int64  1
      grp.attrs['size_allowed' ] = self.size_allowed   # int64  1
      grp.attrs['savings'] = self.savings
      grp.attrs['status' ] = self.status
      # Arrays
      grp['iblks_allowed'] = self.iblks_allowed  # int32  nsyms
      grp['dims_allowed' ] = self.dims_allowed   # int32  nblks
      grp['idx_allowed'  ] = self.idx_allowed    # int64  nblks
      grp['ndim_allowed' ] = self.ndim_allowed   # int64  nblks
      grp['ioff_allowed' ] = self.ioff_allowed   # int64  nblks
      # DArrays
      grp['value'] = self.value
      # Its advantages to pack all data together!
      grp['qsyms'] = numpy.vstack(self.qsyms)
      grp['ndims'] = numpy.hstack(self.ndims)
      if self.idlst is not None:
         grp['idlst'] = numpy.hstack([qtensor_idlst.flatten(self.idlst[irank]) \
           	      		      for irank in range(self.rank)])
      return 0

   #@profile
   def load(self,f1,name):
      grp = f1[name]
      self.rank    = grp.attrs['rank' ] 
      self.nqnum   = grp.attrs['nqnum'] 
      self.nsyms   = grp.attrs['nsyms'] 
      self.shape   = grp.attrs['shape']
      self.size    = grp.attrs['size' ] 
      self.nblks   = grp.attrs['nblks'] 
      self.nblks_allowed = grp.attrs['nblks_allowed']  # int64  1
      self.size_allowed  = grp.attrs['size_allowed' ]  # int64  1
      self.savings = grp.attrs['savings'] 
      self.status  = numpy.array(grp.attrs['status'])
      # Arrays
      self.iblks_allowed = grp['iblks_allowed'].value # int32  nsyms
      self.dims_allowed  = grp['dims_allowed' ].value # int32  nblks
      self.idx_allowed   = grp['idx_allowed'  ].value # int64  nblks
      self.ndim_allowed  = grp['ndim_allowed' ].value # int64  nblks
      self.ioff_allowed  = grp['ioff_allowed' ].value # int64  nblks
      # DArrays
      self.value = grp['value' ].value
      # Sparsing data
      # > qsyms
      self.qsyms = [0]*self.rank
      qsyms = grp['qsyms'].value 
      ioff = 0
      for irank in range(self.rank):
         self.qsyms[irank] = qsyms[ioff:ioff+self.nsyms[irank]]
	 ioff += self.nsyms[irank]
      # > ndims
      self.ndims = [0]*self.rank
      ndims = grp['ndims'].value 
      ioff = 0
      for irank in range(self.rank):
         self.ndims[irank] = ndims[ioff:ioff+self.nsyms[irank]]
	 ioff += self.nsyms[irank]
      # > idlst
      if 'idlst' in grp:
         self.idlst = [0]*self.rank
         idlst = grp['idlst'].value 
         ioff = 0
         for irank in range(self.rank):
            self.idlst[irank] = qtensor_idlst.seperate(self.ndims[irank],\
           		 			       idlst[ioff:ioff+self.shape[irank]])
            ioff += self.shape[irank]
      return 0

   def equalSym(self,other):
      assert self.rank == other.rank
      assert numpy.all([numpy.array_equal(self.qsyms[i],other.qsyms[i]) for i in range(self.rank)])
      assert numpy.all(self.nsyms == other.nsyms)
      #assert self.idlst == other.idlst
      return True

   # This is the central subroutine
   #@profile
   def fromQsyms(self,rank,qsyms,ndims,ifallocate=True,idlst=None):
      debug = False
      # 3
      self.rank  = rank
      self.qsyms = copy.deepcopy(qsyms) # [[1,0],...] for each dimension  
      self.ndims = copy.deepcopy(ndims) #  size for each dimension
      if idlst is not None:
	 self.idlst = copy.deepcopy(idlst) #  indices for each symmetry sector
      # information for qnumbers
      self.nqnum = self.qsyms[0].shape[1] 
      # derived full information
      self.shape = numpy.array([sum(dims) for dims in self.ndims])
      self.size  = numpy.prod(self.shape)
      # derived symmetry blocks
      self.nsyms = numpy.array([qsyms.shape[0] for qsyms in self.qsyms],dtype=numpy.int32) # no. of symmetry sectors
      self.nblks = numpy.prod(self.nsyms)
      if debug:
         print '\n[fromQsyms]'
         print ' rank =',self.rank
         print ' qsyms=',self.qsyms
         print ' ndims=',self.ndims
         #print ' idlst=',self.idlst
         print ' nqnum=',self.nqnum
         print ' shape=',self.shape
         print ' size =',self.size
         print ' nsyms=',self.nsyms
         print ' nblks=',self.nblks
	 # [fromQsyms]
	 #  rank = 3
	 #  qsyms= [array([[ 2. ,  0. ],
	 #        [ 1. , -0.5]]), array([[ 1. ,  0.5],
	 #        [ 0. ,  0. ],
	 #        [ 1. , -0.5]]), array([[ 2. ,  0. ],
	 #        [ 3. ,  1.5]])]
	 #  ndims= [array([4, 1, 2, 2, 1, 2, 1, 1, 2]), array([1, 1, 1, 1]), array([9, 3, 1, 3, 3, 9, 3, 1, 3, 1, 9, 9, 3, 3, 3, 1])]
	 #  idlst= [[[6, 7, 8, 9], [5], [11, 12], [3, 4], [0], [13, 14], [15], [10], [1, 2]], [[2], [3], [0], [1]], [[10, 11, 12, 13, 14, 15, 16, 17, 18], [60, 61, 62], [63], [57, 58, 59], [7, 8, 9], [23, 24, 25, 26, 27, 28, 29, 30, 31], [4, 5, 6], [0], [42, 43, 44], [22], [32, 33, 34, 35, 36, 37, 38, 39, 40], [45, 46, 47, 48, 49, 50, 51, 52, 53], [19, 20, 21], [1, 2, 3], [54, 55, 56], [41]]]
	 #  nqnum= 2
	 #  shape= [16  4 64]
	 #  size = 4096
	 #  nsyms= [ 9  4 16]
	 #  nblks= 576
      assert self.status is not None
      self.iblks_allowed = qtensor_util.blks_allowed1(self.nblks,self.rank,self.nqnum,\
		      				      self.nsyms,self.status,self.qsyms)
      self.nblks_allowed = numpy.sum(self.iblks_allowed)
      # For efficiency reason, store additional information
      self.idx_allowed = numpy.flatnonzero(self.iblks_allowed)
      self.iblks_allowed[self.idx_allowed] = self.idx_allowed+1 
      self.iblks_allowed = self.iblks_allowed.reshape(self.nsyms).copy()
      # Note that iblks_allowed and idx_allowed becomes inversion to each other.
      self.dims_allowed = numpy.array(qtensor_util.cartesian_prod(self.ndims),dtype=numpy.int32)
      self.ndim_allowed = numpy.array(numpy.prod(self.dims_allowed,axis=1),dtype=numpy.int64)
      self.ioff_allowed = numpy.zeros((self.nblks),dtype=numpy.int64)
      ioff = 0
      for idx in self.idx_allowed:
	 self.ioff_allowed[idx] = ioff 
	 ioff += self.ndim_allowed[idx]
      # vals
      self.size_allowed = ioff
      if ifallocate:
         self.value = numpy.zeros(self.size_allowed)
      else:
	 self.value = None
      self.savings = float(self.size_allowed)/self.size
      if debug:
         print ' Symmetry allowed:'
         print ' iblks_allowed  =',self.iblks_allowed
         print ' dims_allowed  =',self.dims_allowed
         print ' ndim_allowed  =',self.ndim_allowed
         print ' ioff_allowed  =',self.ioff_allowed
         print ' nblks_allowed =',self.nblks_allowed,' nblks =',self.nblks
         print ' size_allowed  =',self.size_allowed,' size =',self.size,\
               ' savings=',self.savings
      assert self.savings < 1.0+1.e-6
      return 0

   def prt(self):
      print 'Basic information:'
      print ' rank =',self.rank,' shape =',self.shape,' nsyms =',self.nsyms
      print ' nblks_allowed =',self.nblks_allowed,' nblks =',self.nblks
      print ' size_allowed =',self.size_allowed,' size =',self.size,\
            ' savings =',self.savings
      return 0

   def fromDenseTensor(self,val,qnums,ifcollect=None):
      rank,qsyms,ndims,idlst = qtensor_util.fromQnums(qnums,ifcollect)
      self.fromQsyms(rank,qsyms,ndims)
      self.fromDense(val,idlst)
      # Here, idlst is stored in qtensor!
      self.idlst = copy.deepcopy(idlst)
      return 0

   def fromDense(self,val,idlst):
      for idx in self.idx_allowed:
	 ioff  = self.ioff_allowed[idx]
	 dims  = self.dims_allowed[idx]
 	 ndim  = self.ndim_allowed[idx]
         isyms = numpy.unravel_index(idx,self.nsyms)
	 indices = numpy.ix_(*[idlst[i][isyms[i]] for i in range(self.rank)])
	 self.value[ioff:ioff+ndim] = val[indices].reshape(ndim)
      debug = False
      if debug:
         norm0 = numpy.linalg.norm(val)
         norm1 = numpy.linalg.norm(self.value)
         diff  = abs(norm0-norm1)
         print ' diff of vals [fromDenseTensor] =',diff
         if diff > 1.e-10: exit(1)
      return 0

   #@profile
   def toDenseTensor(self,idlst):
      val = numpy.zeros(self.shape) 
      for idx in self.idx_allowed:
	 ioff  = self.ioff_allowed[idx]
	 dims  = self.dims_allowed[idx]
	 ndim  = self.ndim_allowed[idx]
         isyms = numpy.unravel_index(idx,self.nsyms)
	 indices = numpy.ix_(*[idlst[i][isyms[i]] for i in range(self.rank)])
	 val[indices] = self.value[ioff:ioff+ndim].reshape(dims)
      debug = False
      if debug:
         norm0 = numpy.linalg.norm(val)
         norm1 = numpy.linalg.norm(self.value)
         diff  = abs(norm0-norm1)
         print ' diff of vals [toDenseTensor] =',diff
         if diff > 1.e-10: exit(1)
      return val

   # Diag for the last two dimensions
   def toDenseDiag(self,idlst):
      val = numpy.zeros(self.shape[:-1]) 
      for idx in self.idx_allowed:
	 ioff  = self.ioff_allowed[idx]
	 dims  = self.dims_allowed[idx]
	 ndim  = self.ndim_allowed[idx]
         isyms = numpy.unravel_index(idx,self.nsyms)
	 if isyms[-1] != isyms[-2]: continue
	 indices = numpy.ix_(*[idlst[i][isyms[i]] for i in range(self.rank-1)])
	 val[indices] = numpy.einsum('...ii->...i',self.value[ioff:ioff+ndim].reshape(dims))
      return val

   # A brute-force way to make Hd from H in MPO representation
   #@profile
   def diagH(self):
      qt = qtensor(self.status)
      qt.fromQsyms(self.rank,self.qsyms,self.ndims,idlst=self.idlst)
      for idx in self.idx_allowed:
	 ioff  = self.ioff_allowed[idx]
	 ndim  = self.ndim_allowed[idx]         
	 dims  = self.dims_allowed[idx]
	 # find new position
	 isyms = numpy.unravel_index(idx,self.nsyms)
	 if isyms[-1] != isyms[-2]: continue
	 tmp = self.value[ioff:ioff+ndim].reshape(dims)
	 dim = dims[-1]
	 tmp[...,~numpy.eye(dim,dtype=bool)] = 0.
	 qt.value[ioff:ioff+ndim] = tmp.reshape(ndim)
      return qt

   #
   # Projection map:
   #
   # Compared to prjmap from the elementwise dpt result,
   # the entries must be the same, but the ordering is 
   # generally different due to the blockwise operations
   # and the fromQnums constructions, which do reorderings.
   #
   # this prjmap: [ 3 19  6 22  9 25 12 28 34 40 49 52 64]
   # dptSymmetry: [ 3  6  9 12 19 22 25 28 34 40 49 52 64]
   #
   def prjmap(self,idlst):
      prjmap = []
      for idx in self.idx_allowed:
	 ioff  = self.ioff_allowed[idx]
	 dims  = self.dims_allowed[idx]
	 ndim  = self.ndim_allowed[idx]
         isyms = numpy.unravel_index(idx,self.nsyms)
	 indices = [idlst[i][isyms[i]] for i in range(self.rank)]
	 ntuple = qtensor_util.cartesian_prod(indices)
	 idx = map(lambda x:numpy.ravel_multi_index(x,self.shape),ntuple)
	 prjmap += idx
      return numpy.array(prjmap)

   # Assuming continous groups, otherwise tranpose can be applied first!
   # Note that the merge operation does NOT re-collect the symmetry info
   # for the merged index! The block structure is not changed too much.
   # That is, in the qsysm_new, it is allowed to have sectors with the
   # same symmetry labels. Maybe used in future to improve performance!
   #
   # However, one must be careful for contraction such quanties, and a 
   # sorting procedure may requires sometimes.
   #
   #@profile
   def merge(self,groups):
      rank = len(groups)
      assert rank >= 2
      # outerproduct/merging of quantum number?
      # So only quantum number with the same status can be added.
      qsyms_new = []
      ndims_new = []
      #idlst_new = []
      status_new = []
      for ig in range(rank):
         ng = len(groups[ig])
	 status = [self.status[i] for i in groups[ig]] 
         assert numpy.array_equal(status,[status[0]]*ng)
	 status_new.append(status[0])
	 qsyms = [self.qsyms[i] for i in groups[ig]]
	 ndims = [self.ndims[i] for i in groups[ig]]
         #idlst = [self.idlst[i] for i in groups[ig]]
	 shape = [self.shape[i] for i in groups[ig]] # global shape
	 qsyms_prod = qtensor_util.cartesian_prod_qsyms(qsyms)
	 ndims_prod = qtensor_util.cartesian_prod_nsyms(ndims)
	 #idlst_prod = qtensor_util.cartesian_prod_idlst(idlst,shape)
	 qsyms_new.append(qsyms_prod)
	 ndims_new.append(ndims_prod)
	 #idlst_new.append(idlst_prod)
      # create new object
      qt = qtensor(status_new)
      # direct construct block structures via qsyms,
      # therefore, there is no merging of same syms!
      qt.fromQsyms(rank,qsyms_new,ndims_new)
      # under the contiguous assumption, only 
      # reshaping is needed for each block,
      # and in 1d settings, just copy it!
      qt.value = self.value.copy()
      return qt

   #@profile
   def transpose(self,*args):
      rank  = self.rank
      qsyms = [self.qsyms[i] for i in args]
      ndims = [self.ndims[i] for i in args]
      #idlst = [self.idlst[i] for i in args]
      status = [self.status[i] for i in args]
      qt = qtensor(status)
      qt.fromQsyms(rank,qsyms,ndims)
      for idx in self.idx_allowed:
	 ioff  = self.ioff_allowed[idx]
	 ndim  = self.ndim_allowed[idx]         
	 dims  = self.dims_allowed[idx]
	 # find new position
	 isyms = numpy.unravel_index(idx,self.nsyms)
	 isyms_new = tuple([isyms[i] for i in args])
	 idx_new  = qt.iblks_allowed[isyms_new]-1
	 ioff_new = qt.ioff_allowed[idx_new]
	 ndim_new = qt.ndim_allowed[idx_new]
	 assert ndim == ndim_new
	 tmp = self.value[ioff:ioff+ndim].reshape(dims)
	 qt.value[ioff_new:ioff_new+ndim] = tmp.transpose(*args).reshape(ndim)
      return qt

   # very similar to transpose: in fact, even isyms
   # do not need to be changed, if the same symmetries 
   # are allowed to be splitted into several blocks.
   # >>> THE BLOCK STRUCTURE should be consistent with MPO[N]
   def reduceQsymsToN(self):
      qt = qtensor(self.status)
      qsyms = qtensor_util.reduceQsymsToN(self.qsyms)
      qt.fromQsyms(self.rank,qsyms,self.ndims,idlst=self.idlst)
      # map the entries into new arrays
      for idx in self.idx_allowed:
	 ioff  = self.ioff_allowed[idx]
	 ndim  = self.ndim_allowed[idx]         
	 # find new position
	 isyms = numpy.unravel_index(idx,self.nsyms)
	 idx_new  = qt.iblks_allowed[isyms]-1
	 ioff_new = qt.ioff_allowed[idx_new]
	 ndim_new = qt.ndim_allowed[idx_new]
	 assert ndim == ndim_new
	 qt.value[ioff_new:ioff_new+ndim] = self.value[ioff:ioff+ndim]
      return qt

   # "reverse" of reduceQsymsToN
   def projectionNMs(self,qsyms0):
      qt = qtensor(self.status)
      qt.fromQsyms(self.rank,qsyms0,self.ndims)
      # extract the entries into new arrays
      for idx in qt.idx_allowed:
	 ioff  = qt.ioff_allowed[idx]
	 ndim  = qt.ndim_allowed[idx]         
	 # find new position
	 isyms = numpy.unravel_index(idx,qt.nsyms)
	 idx_old = self.iblks_allowed[isyms]-1
	 ioff_old = self.ioff_allowed[idx_old]
	 ndim_old = self.ndim_allowed[idx_old]
	 assert ndim == ndim_old
	 qt.value[ioff:ioff+ndim] = self.value[ioff_old:ioff_old+ndim]
      return qt

   #@profile
   def tensordotSYM(self,qt1,qt2,axes):
      # indices
      r1 = range(qt1.rank)
      r2 = range(qt2.rank)
      i1,i2 = axes
      e1 = list(set(r1)-set(i1))
      e2 = list(set(r2)-set(i2))
      ne1 = len(e1)
      ne2 = len(e2)
      nii = len(i1)
      # synthesize symmetry
      intqsyms1 = [qt1.qsyms[i] for i in i1]
      intqsyms2 = [qt2.qsyms[i] for i in i2]
      ifequal = numpy.all([numpy.array_equal(intqsyms1[i],intqsyms2[i]) for i in range(nii)])
      assert ifequal == True
      # qsyms
      extqsyms1 = [qt1.qsyms[i] for i in e1]
      extqsyms2 = [qt2.qsyms[i] for i in e2]
      extqsyms12 = extqsyms1+extqsyms2
      # ndims
      extndims1 = [qt1.ndims[i] for i in e1]
      extndims2 = [qt2.ndims[i] for i in e2]
      extndims12 = extndims1+extndims2
      # idlst
      classes = [extqsyms12,extndims12]#,extidlst12]
      # status
      ints1 = [qt1.status[i] for i in i1]
      ints2 = [qt2.status[i] for i in i2]
      exts1 = [qt1.status[i] for i in e1]
      exts2 = [qt2.status[i] for i in e2]
      if nii != 0:
         # flip if in the same direction
         if ints1[0] == ints2[0]:
            ints2 = map(lambda x:not x,ints2)
            exts2 = map(lambda x:not x,exts2)
         # check all status, such that those pairs are T/F pairs.
         sta = ints1[0]
         for is1,is2 in zip(ints1,ints2):
            assert is1 != is2 
      exts12 = exts1+exts2
      # symmetry screening
      rank = ne1+ne2
      self.status = numpy.array(exts12) # This is essential!
      # Generation of final information
      self.fromQsyms(rank,extqsyms12,extndims12,ifallocate=False)
      # Compute contraction
      self.axes = axes
      self.clst = []
      # Set up table
      sdx1 = numpy.array(e1+i1)
      sdx2 = numpy.array(e2+i2)
      iblks1 = qt1.iblks_allowed.transpose(sdx1).copy()
      iblks2 = qt2.iblks_allowed.transpose(sdx2).copy()
      for idx in self.idx_allowed:
         ioff = self.ioff_allowed[idx]
         dims = self.dims_allowed[idx]
         ndim = self.ndim_allowed[idx]
         # Only loop over nonzero blocks 
         isyms = numpy.unravel_index(idx,self.nsyms)
         nzpt1 = iblks1[isyms[:ne1]]
         nzpt2 = iblks2[isyms[ne1:]]
         nzpt12 = numpy.nonzero(numpy.logical_and(nzpt1,nzpt2))
         # many internal case, e.g., shape = (1, 4) or (2, 3).
         for inzpt in zip(*nzpt12): 
            idx1 = nzpt1[inzpt]-1
            idx2 = nzpt2[inzpt]-1
            ioff1 = qt1.ioff_allowed[idx1]
            dims1 = qt1.dims_allowed[idx1]
            ndim1 = qt1.ndim_allowed[idx1]
            ioff2 = qt2.ioff_allowed[idx2]
            dims2 = qt2.dims_allowed[idx2]
            ndim2 = qt2.ndim_allowed[idx2]
	    # clst: information for contraction
	    self.clst.append([ioff,ioff1,ioff2,ndim,ndim1,ndim2,dims,dims1,dims2])
      return len(self.idx_allowed)


   # General contraction case
   #@profile
   def tensordotCAL(self,qt1,qt2,ifc1=False,ifc2=False,thresh=1.e-20,tsize=10000):
      self.value = numpy.zeros(self.size_allowed)
      for item in self.clst:
         ioff,ioff1,ioff2,ndim,ndim1,ndim2,dims,dims1,dims2 = item
         t1 = qt1.value[ioff1:ioff1+ndim1].reshape(dims1)
	 t2 = qt2.value[ioff2:ioff2+ndim2].reshape(dims2)
	 # Size should be sufficiently large 
	 if ndim1 > tsize:
	    amax1 = numpy.max(numpy.abs(t1))
	    if amax1 < thresh: continue
	 if ndim2 > tsize:
	    amax2 = numpy.max(numpy.abs(t2))
	    if amax2 < thresh: continue
         # T1*T2
	 if ifc1: t1 = t1.conj()
	 if ifc2: t2 = t2.conj()
	 self.value[ioff:ioff+ndim] += numpy.tensordot(t1,t2,self.axes).reshape(ndim)
      return 0 

# 
# More general tensordot
#
#@profile
def tensordot(qt1,qt2,ifc1=False,ifc2=False,axes=None):
   debug = False
   if debug: print '[tensordot]'
   assert axes is not None
   # indices
   r1 = range(qt1.rank)
   r2 = range(qt2.rank)
   if axes is None:
      i1 = range(qt1.rank)
      i2 = range(qt2.rank)
   else:
      i1,i2 = axes
   e1 = list(set(r1)-set(i1))
   e2 = list(set(r2)-set(i2))
   ne1 = len(e1)
   ne2 = len(e2)
   nii = len(i1)
   if debug:
      print ' t1.shape =',qt1.shape
      print ' t2.shape =',qt2.shape
      print ' r1,i1,e1,ne1 =',r1,i1,e1,ne1
      print ' r2,i2,e2,ne2 =',r2,i2,e2,ne2
   # synthesize symmetry
   intqsyms1 = [qt1.qsyms[i] for i in i1]
   intqsyms2 = [qt2.qsyms[i] for i in i2]
   ifequal = numpy.all([numpy.array_equal(intqsyms1[i],intqsyms2[i]) for i in range(nii)])
   assert ifequal == True
   # qsyms
   extqsyms1 = [qt1.qsyms[i] for i in e1]
   extqsyms2 = [qt2.qsyms[i] for i in e2]
   extqsyms12 = extqsyms1+extqsyms2
   # ndims
   extndims1 = [qt1.ndims[i] for i in e1]
   extndims2 = [qt2.ndims[i] for i in e2]
   extndims12 = extndims1+extndims2
   ## idlst
   #extidlst1 = [qt1.idlst[i] for i in e1]
   #extidlst2 = [qt2.idlst[i] for i in e2]
   #extidlst12 = extidlst1+extidlst2
   classes = [extqsyms12,extndims12]#,extidlst12]
   # status
   ints1 = [qt1.status[i] for i in i1]
   ints2 = [qt2.status[i] for i in i2]
   exts1 = [qt1.status[i] for i in e1]
   exts2 = [qt2.status[i] for i in e2]
   if nii != 0:
      # flip if in the same direction
      if ints1[0] == ints2[0]:
         ints2 = map(lambda x:not x,ints2)
         exts2 = map(lambda x:not x,exts2)
      # check all status, such that those pairs are T/F pairs.
      sta = ints1[0]
      for is1,is2 in zip(ints1,ints2):
         assert is1 != is2 
   exts12 = exts1+exts2
   if debug:
      print 'check status'
      print ' s1 =',exts1,ints1
      print ' s2 =',exts2,ints2
   # symmetry screening
   rank = ne1+ne2
   #
   # Scalar case
   #
   if rank == 0:
      # Compute contraction
      sdx1 = numpy.array(e1+i1)
      sdx2 = numpy.array(e2+i2)
      nzpt1 = qt1.iblks_allowed.transpose(sdx1)
      nzpt2 = qt2.iblks_allowed.transpose(sdx2)
      nzpt12 = numpy.nonzero(numpy.logical_and(nzpt1,nzpt2))
      qt = 0.0 
      for inzpt in zip(*nzpt12): 
         idx1 = nzpt1[inzpt]-1
         idx2 = nzpt2[inzpt]-1
         ioff1 = qt1.ioff_allowed[idx1]
         dims1 = qt1.dims_allowed[idx1]
         ndim1 = qt1.ndim_allowed[idx1]
         ioff2 = qt2.ioff_allowed[idx2]
         dims2 = qt2.dims_allowed[idx2]
         ndim2 = qt2.ndim_allowed[idx2]
         t1 = qt1.value[ioff1:ioff1+ndim1].reshape(dims1)
         t2 = qt2.value[ioff2:ioff2+ndim2].reshape(dims2)
         # contract
	 if ifc1: t1 = t1.conj() 
	 if ifc2: t2 = t2.conj() 
 	 qt += numpy.tensordot(t1,t2,axes)
   else:
      qt = qtensor(exts12)
      # Generation of final information
      qt.fromQsyms(rank,extqsyms12,extndims12)
      if debug:
         print ' nblks_allowed =',qt.nblks_allowed,' nblks =',qt.nblks
         print ' size_allowed  =',qt.size_allowed,' size =',qt.size,\
               ' savings=',qt.savings
      # Compute contraction
      sdx1 = numpy.array(e1+i1)
      sdx2 = numpy.array(e2+i2)
      iblks1 = qt1.iblks_allowed.transpose(sdx1)
      iblks2 = qt2.iblks_allowed.transpose(sdx2)
      #
      # Outer product case
      #
      if nii == 0: 
         for idx in qt.idx_allowed:
            ioff = qt.ioff_allowed[idx]
            dims = qt.dims_allowed[idx]
            ndim = qt.ndim_allowed[idx]
            # Only loop over nonzero blocks 
            isyms = numpy.unravel_index(idx,qt.nsyms)
            nzpt1 = iblks1[isyms[:ne1]]
            nzpt2 = iblks2[isyms[ne1:]]
            if nzpt1 ==0 or nzpt2 ==0: continue 
            idx1  = nzpt1-1
            idx2  = nzpt2-1
            ioff1 = qt1.ioff_allowed[idx1]
            dims1 = qt1.dims_allowed[idx1]
            ndim1 = qt1.ndim_allowed[idx1]
            ioff2 = qt2.ioff_allowed[idx2]
            dims2 = qt2.dims_allowed[idx2]
            ndim2 = qt2.ndim_allowed[idx2]
            t1 = qt1.value[ioff1:ioff1+ndim1].reshape(dims1)
            t2 = qt2.value[ioff2:ioff2+ndim2].reshape(dims2)
            # direct product
	    if ifc1: t1 = t1.conj() 
	    if ifc2: t2 = t2.conj() 
            tmp12 = numpy.tensordot(t1,t2,axes)
            qt.value[ioff:ioff+ndim] = tmp12.reshape(ndim)
      #
      # General contraction case
      #
      else:

         for idx in qt.idx_allowed:
            ioff = qt.ioff_allowed[idx]
            dims = qt.dims_allowed[idx]
            ndim = qt.ndim_allowed[idx]
            tmp12 = numpy.zeros(dims)
            # Only loop over nonzero blocks 
            isyms = numpy.unravel_index(idx,qt.nsyms)
            nzpt1 = iblks1[isyms[:ne1]]
            nzpt2 = iblks2[isyms[ne1:]]
            nzpt12 = numpy.nonzero(numpy.logical_and(nzpt1,nzpt2))
            # many internal case, e.g., shape = (1, 4) or (2, 3).
            for inzpt in zip(*nzpt12): 
               idx1 = nzpt1[inzpt]-1
               idx2 = nzpt2[inzpt]-1
               ioff1 = qt1.ioff_allowed[idx1]
               dims1 = qt1.dims_allowed[idx1]
               ndim1 = qt1.ndim_allowed[idx1]
               ioff2 = qt2.ioff_allowed[idx2]
               dims2 = qt2.dims_allowed[idx2]
               ndim2 = qt2.ndim_allowed[idx2]
               t1 = qt1.value[ioff1:ioff1+ndim1].reshape(dims1)
               t2 = qt2.value[ioff2:ioff2+ndim2].reshape(dims2)
               # contract
	       if ifc1: t1 = t1.conj() 
	       if ifc2: t2 = t2.conj() 
               tmp12 += numpy.tensordot(t1,t2,axes)
            # store
            qt.value[ioff:ioff+ndim] = tmp12.reshape(ndim)

   return qt 
