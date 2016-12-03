import copy
import numpy
import ctypes
from qcmpodmrg.source.sysutil_include import libqsym 

##############
# main codes
##############
def reduceQnumsToN(qnums):
   return numpy.array(map(lambda x:numpy.array([x[0]]),qnums))

# Input:
# [array([[ 0.,  0.]]), array([[ 0. ,  0. ],[ 1. , -0.5],[ 1. ,  0.5],[ 2. ,  0. ]])]
# Output:
# [array([[ 0.]]), array([[ 0.],[ 1.],[ 1.],[ 2.]])]
def reduceQsymsToN(qsyms):
   return map(lambda x:(x.T[0:1]).T,qsyms)

# Return: rank,qsyms,ndims,idlst
def fromQnums(qnums,ifcollect=None):
   # [[qsyms,ndims,idlst],...] 
   rank = len(qnums)
   if ifcollect is None: 
      ifclt = [1]*rank
   else:
      ifclt = ifcollect
   classes = [classification(x,y) for x,y in zip(qnums,ifclt)]
   qsyms = [cls[0] for cls in classes] 
   ndims = [cls[1] for cls in classes] 
   idlst = [cls[2] for cls in classes] 
   return rank,qsyms,ndims,idlst

# Classification of qnums
def classification(qnums,ifclt=1):
   if ifclt:
      dic = {}
      for idx,val in enumerate(qnums):
         dic.setdefault(str(map(float,val)),[]).append(idx)
      qsyms = map(eval,dic.keys())
      idlst = dic.values()
      # Sorting
      arg = numpy.argsort(map(lambda x:min(x),idlst))
      qsyms = numpy.array([qsyms[i] for i in arg]) 
      idlst = [idlst[i] for i in arg]
      ndims = numpy.array([len(indices) for indices in idlst])
   else:
      lenq = len(qnums)
      qsyms = numpy.array(qnums)
      idlst = [[i] for i in range(lenq)]
      ndims = numpy.array([1]*lenq)
   return qsyms,ndims,idlst

def blks_allowed0(nblks,rank,nqnum,nsyms,status,qsyms):
   blks_allowed = numpy.zeros((nblks),dtype=numpy.int32)
   for idx in range(nblks):
      isyms = numpy.unravel_index(idx,nsyms)
      okey = numpy.zeros(nqnum)
      ikey = numpy.zeros(nqnum)
      for i in range(rank):
         isym = isyms[i]
         if status[i]:
            okey += qsyms[i][isym]
         else:
            ikey += qsyms[i][isym]
      diff = numpy.amax(abs(okey-ikey))
      if diff<1.e-10: blks_allowed[idx] = 1
   return blks_allowed 

#@profile
def blks_allowed1(nblks,rank,nqnum,nsyms,status,qsyms,debug=False):
   blks_allowed = numpy.zeros((nblks),dtype=numpy.int32)
   sint = numpy.zeros(rank,dtype=numpy.int32)
   sint[status] = 1
   null = ctypes.c_void_p()
   maxn = 9
   assert rank <= 9
   args = [qsyms[i].ctypes.data_as(ctypes.c_void_p) for i in range(rank)]\
        + [null]*(maxn-rank)
   libqsym.symAllowed(ctypes.c_int(rank),
      	              ctypes.c_int(nqnum),
      	              nsyms.ctypes.data_as(ctypes.c_void_p),
      	              sint.ctypes.data_as(ctypes.c_void_p),
      	              blks_allowed.ctypes.data_as(ctypes.c_void_p),
      	              *args)
   if debug: 
      tmp = blks_allowed0(nblks,rank,nqnum,nsyms,status,qsyms)
      diff = numpy.linalg.norm(tmp-blks_allowed)
      print 'diff=',diff
      assert diff<1.e-10
   return blks_allowed

def genOffset(nsyms):
  rank = len(nsyms)	
  noff = numpy.zeros(rank,dtype=numpy.int32)
  noff[rank-1] = 1
  for i in range(rank-1,0,-1):
     noff[i-1] = nsyms[i]*noff[i]     
  return noff

# nsyms - merged dimension: d1*d2*...
def cartesian_prod_nsyms(arrays):
   return numpy.prod(cartesian_prod(arrays),axis=1)

# qsyms - abelian case
def cartesian_prod_qsyms(arrays):
   def cartesian_prod_qsyms_pair(array1,array2):
      lst = []
      for i1 in array1:
         for i2 in array2:
            lst.append(i1+i2)
      return numpy.array(lst)
   n = len(arrays)
   if n == 1:
      return arrays[0]
   else:
      return reduce(cartesian_prod_qsyms_pair,arrays)

##############
# from pyscf
##############
def cartesian_prod(arrays, out=None):
    '''
    Generate a cartesian product of input arrays.
    http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Args:
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.

    Returns:
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.

    Examples:

    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    '''
    arrays = [numpy.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    nd = len(arrays)
    dims = [nd] + [len(x) for x in arrays]

    if out is None:
        out = numpy.empty(dims, dtype)
    else:
        out = numpy.ndarray(dims, dtype, buffer=out)
    tout = out.reshape(dims)

    shape = [-1] + [1] * nd
    for i, arr in enumerate(arrays):
        tout[i] = arr.reshape(shape[:nd-i])

    return tout.reshape(nd,-1).T

def direct_sum(subscripts, *operands):
    '''Apply the summation over many operands with the einsum fashion.

    Examples:

    >>> a = numpy.ones((6,5))
    >>> b = numpy.ones((4,3,2))
    >>> direct_sum('ij,klm->ijklm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('ij,klm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('i,j,klm->mjlik', a[0], a[:,0], b).shape
    (2, 6, 3, 5, 4)
    >>> direct_sum('ij-klm->ijklm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('ij+klm', a, b).shape
    (6, 5, 4, 3, 2)
    >>> direct_sum('-i-j+klm->mjlik', a[0], a[:,0], b).shape
    (2, 6, 3, 5, 4)
    '''

    def sign_and_symbs(subscript):
        ''' sign list and notation list'''
        subscript = subscript.replace(' ', '').replace(',', '+')

        if subscript[0] not in '+-':
            subscript = '+' + subscript
        sign = [x for x in subscript if x in '+-']

        symbs = subscript[1:].replace('-', '+').split('+')
        return sign, symbs

    if '->' in subscripts:
        src, dest = subscripts.split('->')
        sign, src = sign_and_symbs(src)
        dest = dest.replace(' ', '')
    else:
        sign, src = sign_and_symbs(subscripts)
        dest = ''.join(src)
    assert(len(src) == len(operands))

    for i, symb in enumerate(src):
        op = numpy.asarray(operands[i])
        assert(len(symb) == op.ndim)
        if i == 0:
            if sign[i] is '+':
                out = op
            else:
                out = -op
        elif sign[i] == '+':
            out = out.reshape(out.shape+(1,)*op.ndim) + op
        else:
            out = out.reshape(out.shape+(1,)*op.ndim) - op

    return numpy.einsum('->'.join((''.join(src), dest)), out)


if __name__ == '__main__':
   print genOffset([3])
   noff = genOffset([3,4,5])
   print noff
   print numpy.dot([2,3,4],noff)
