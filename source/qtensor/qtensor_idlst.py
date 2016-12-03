import numpy
import itertools
import qtensor_util

# Merging group of indices
def cartesian_prod_idlst(arrays,shape):
   def cartesian_prod_idlst_pair(array1,array2):
      lst = []
      for i1 in array1:
         for i2 in array2:
            lst.append(qtensor_util.cartesian_prod([i1,i2]))
      return lst
   n = len(arrays)
   if n == 1:
      idlst_prod = copy.deepcopy(arrays[0])
   else:
      tmp = reduce(cartesian_prod_idlst_pair,arrays)
      # map the multi index into new flat index
      idlst_prod = [map(lambda x:numpy.ravel_multi_index(x,shape),idx) for idx in tmp]
   return idlst_prod

# each item in idlst is [[0], [1], [2], [3]] => [0,1,2,3]
def flatten(idlsti):
   return numpy.array(list(itertools.chain(*idlsti)))

# each item in idlst is [0,1,2,3] => [[0], [1], [2], [3]]  
def seperate(ndimsi,idlsti):
   ioff = 0
   idlst = []
   for idx,indim in enumerate(ndimsi):
      idlst.append(idlsti[ioff:ioff+indim])
      ioff += indim
   return idlst 
