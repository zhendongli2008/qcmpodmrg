import numpy

def tensor_matricization(tensor,shape,imode):
   N=len(shape)
   order=range(N)
   order.remove(imode)
   # new indices
   order=tuple([imode]+order)
   # matricization
   tmat=tensor.copy()
   tmat=tmat.reshape(shape).transpose(order)
   tmat=tmat.reshape((shape[imode],-1))
   return order,tmat

#@profile
def matrix_KRprod(a,b):
   # a[r,x]b[r,y]=>c[r,xy]
   r1,p1=a.shape
   r2,p2=b.shape
   if r1 != r2: 
      print 'inconsistent shape in KRprod !',r1,r2
      exit(1)
   r=r1   
   #c=numpy.zeros((r1,p1*p2))
   #for i in range(r):
   #   c[i]=numpy.kron(a[i],b[i])
   c=numpy.einsum('rx,ry->rxy',a,b)
   c=c.reshape(r,p1*p2)
   return c   
