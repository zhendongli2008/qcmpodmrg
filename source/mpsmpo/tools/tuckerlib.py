#
# Tucker decomposition:
#
#    T[p1,p2,p3] => C[a1,a2,a3]A1[p1,a1]A2[p2,a2]A3[p3,a3]
#
import numpy

def tucker_check(tensor,shape,tucker):
   # recover
   tensor1=tucker_prod(tucker)
   tensor2=tensor.copy().reshape(shape)
   diff=numpy.linalg.norm(tensor1-tensor2)
   print "DIFF=",diff
   return diff

def tucker_pdim(tucker):
   return map(lambda x:x.shape[0],tucker[1])

def tucker_prod(tucker):
   print "\n[tucker_prod]: form full T[n1,n2,...,nk]"
   tucker_core=tucker[0]
   tucker_site=tucker[1]
   N=len(tucker_site)
   shape=tucker_core.shape
   tmp=tucker_core.copy()
   tmp=tmp.reshape(shape[0],-1)
   # pack all site tensors
   for i in range(N):
      # C[a1|...]B[p1,a1]=>C'[p1|...]=>C'[...|p1]=>C'[a2|...p1]
      tmp=numpy.einsum("pa,ab->pb",tucker_site[i],tmp)
      tmp=tmp.transpose(1,0)
      if i<N-1:
         tmp=tmp.reshape(shape[i+1],-1)	      
      else:
    	 pindx=tucker_pdim(tucker)
         tensor=tmp.reshape(pindx)
   return tensor
