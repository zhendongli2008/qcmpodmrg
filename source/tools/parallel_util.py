import numpy

# Partition data according to size & rank
def partitionSites(nsites,size,rank,iop=1,debug=False):
   blocksize = nsites//size
   sites = range(nsites) 
   if size == 1:
      isites = sites
   elif size == 2:
      if rank == 0:
         isites = sites[:blocksize]
      elif rank == 1:
         isites = sites[blocksize:]
   else:
      nres = nsites%size
      nparts = [blocksize]*size
      if nres != 0:
	 # symmetric distribution for nres
	 if iop == 0:
 	    nl = nres//2
	    nr = nres-nl
	    res = [1]*nl+[0]*(size-nres)+[1]*nr
	 # asymmetric distribution for nres
	 else:
	    res = [1]*nres+[0]*(size-nres)
	 nparts = [nparts[i]+res[i] for i in range(size)]
      if debug: print 'parts=',nparts
      if rank == 0:
         isites = sites[:nparts[0]]
      elif rank == size-1:
         isites = sites[nsites-nparts[-1]:]
      else:
	 noff = sum(nparts[:rank])
         isites = sites[noff:noff+nparts[rank]]
   if debug: print ' size=',size,' rank=',rank,' nisites=',len(isites),' isites=',isites
   return numpy.array(isites)

# ij = i*nj+j
def unrank(ij,nj):
   i = ij // nj
   j = ij % nj
   return i,j

if __name__ == '__main__':
   nsites = 59
   size = 4
   for i in range(size):
      print partitionSites(nsites,size,i)

   nj = 4
   for i in range(14):
      print unrank(i,nj)
