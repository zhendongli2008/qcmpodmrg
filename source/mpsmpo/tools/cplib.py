#
# CP decomposition
#
#    T[p1,p2,p3] => sum_r lambda[r]A1[r,p3]A2[r,p2]A3[r,p3]
#
# as Tucker with super-diagonal core
# as MPS with diagonal site tensor
#
import numpy
import tensorSubs

def cp_check(tensor,shape,cp,iprt=0):
   # recover
   tensor1=cp_prod(cp)
   tensor2=tensor.copy().reshape(shape)
   diff=numpy.linalg.norm(tensor1-tensor2)
   if iprt>0: print "DIFF=",diff
   return diff

def cp_pdim(cp):
   return map(lambda x:x.shape[1],cp[1])

def cp_prod(cp,iprt=0):
   if iprt>0: print "\n[cp_prod]: form full T[n1,n2,...,nk]"
   cp_core=cp[0]
   cp_site=cp[1]
   N=len(cp_site)
   Rcp=len(cp_core)
   pindx=cp_pdim(cp)
   #for r in range(Rcp):
   #   tmp=numpy.array([1.0])	   
   #   for i in range(N):
   #      # a1r[p1]*a2r[p2]...     
   #      tmp=numpy.kron(tmp,cp_site[i][r])
   #   tensor+=cp_core[r]*tmp
   ctensor=reduce(tensorSubs.matrix_KRprod,cp_site)
   tensor=numpy.array(cp_core).dot(ctensor)
   tensor=tensor.reshape(pindx)
   return tensor
