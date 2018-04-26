import numpy

#
# i *---*---*---*
#   0   1   2   3
#
def t1d(n,t):
   nsite = n
   tmatrix = numpy.zeros((nsite*2,nsite*2))
   tmat = numpy.zeros((nsite,nsite))
   # Row:
   for i in range(n-1):
      tmat[i,i+1] = -t
      tmat[i+1,i] = -t
   # Save 
   tmatrix[0::2,0::2] = tmat
   tmatrix[1::2,1::2] = tmat
   return nsite,tmatrix

#
# i/j 0   1   2   3
# 3  *---*---*---*
#    |12 |13 |14 |15
# 2  *---*---*---*
#    |8  |9  |10 |11
# 1  *---*---*---*
#    |4  |5  |6  |7
# 0  *---*---*---*
#     0   1   2   3
#
def t2d(n,t,ifpbc=True):
   nsite = n*n
   tmatrix = numpy.zeros((nsite*2,nsite*2))
   tmat = numpy.zeros((nsite,nsite))
   if ifpbc:
      for i in range(n):
         tmat[i*n,(i+1)*n-1] = -t
         tmat[(i+1)*n-1,i*n] = -t
      for i in range(n):
         tmat[i,(n-1)*n+i] = -t
         tmat[(n-1)*n+i,i] = -t
   # Row:
   for i in range(n):
      for j in range(n-1):
         ijC = i*n+j
         ijR = i*n+j+1
         tmat[ijC,ijR] = -t
         tmat[ijR,ijC] = -t
   # Up:
   for i in range(n-1):
      for j in range(n):
         ijC = i*n+j
         ijU = (i+1)*n+j
         tmat[ijC,ijU] = -t
         tmat[ijU,ijC] = -t
   # Save 
   tmatrix[0::2,0::2] = tmat
   tmatrix[1::2,1::2] = tmat
   return nsite,tmatrix

if __name__ == '__main__':
   t2d(4,1.0)
