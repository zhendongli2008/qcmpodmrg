#
# Fiedler
#
import numpy
import scipy

#
# It seems necessary to exclude -1.0, otherwise dii=0.0
# and sometime the first eigenvector is not zero any more
# due to a large negative eigenvalue ->
#
# (POSITIVE definiteness of the laplacian matrix???)
#
# Or it does not matter due to the constant shift???
#

#
# dij = Jij/Sqrt[Jii*Jjj] - 1.0
#
def distanceMatrix(eri):
   nb = eri.shape[0]
   dij = numpy.zeros((nb,nb))
   for i in range(nb):
      for j in range(nb):
	 dij[i,j] = eri[i,i,j,j]/numpy.sqrt(eri[i,i,i,i]*eri[j,j,j,j]) 
   return dij

# Kij = (ij|ij)
def exchangeMatrix(eri):
   nb = eri.shape[0]
   kij = numpy.zeros((nb,nb))
   for i in range(nb):
      for j in range(nb):
         kij[i,j] = eri[i,j,i,j] 
   return kij

# L = D - K
def laplacian(dij):
   nb  = dij.shape[0]
   lap = numpy.zeros((nb,nb))
   lap = -dij
   # See DMRG in practive 2015 Dii = sum_j Kij
   diag = numpy.einsum('ij->i',dij)
   lap += numpy.diag(diag)
   return lap

# Get the orbital ordering
def orbitalOrdering(eri,mode='kij',debug=False):
   if debug: print '\n[fielder.orbitalOrdering] determing ordering based on',mode.lower()
   nb  = eri.shape[0]
   if mode.lower() == 'dij':
      dij = distanceMatrix(eri)
   elif mode.lower() == 'kij':
      dij = exchangeMatrix(eri)	  
   elif mode.lower() == 'kmat':
      dij = eri.copy() 
   lap = laplacian(dij)
   eig,v = scipy.linalg.eigh(lap)
   # From postive to negative
   order=numpy.argsort(v[:,1])[::-1]
   if debug: 
      print 'dij:\n',dij
      print 'eig:\n',eig
      print 'v[1]=',v[:,1]
      print 'new order:',order
   return order
