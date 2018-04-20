import numpy
import itertools

# Vacuum
def vacuum(isym=1):
   if isym == 1:
      qvac = [[0]]
   elif isym == 2:
      qvac = [[0,0]]
   return qvac

# Spin-orbitals  
def init(norb,isym=1):
   if isym == 0:
      lst = None		
   elif isym == 1:
      lst = [[[0],[1]]]*norb
   elif isym == 2:
      assert norb%2 == 0	   
      lst = [[[0,0],[1,0.5]],[[0,0],[1,-0.5]]]*(norb/2)
   else:
      print 'error in mpo_qphys.init: no such isym=',isym
      exit(1)
   return lst

def prt(qphys):
   n = 50
   nsite = len(qphys)
   print '[mpo_qphys.prt]'
   print '-'*60
   print ' Number of sites=',nsite
   for i in range(nsite):
      print ' isite=',i,qphys[i]
   print '-'*60
   return 0

# Merge two indices (n1,n2) into a combined index
def dpt(q1,q2):
   lst = []
   for qi,qj in itertools.product(q1,q2):
      assert len(qi) == len(qj)
      lst.append([qi[i]+qj[i] for i in range(len(qi))])
   return lst

# Merge more physical indices
def merge(qphys,partition,debug=False):
   if debug: 
      print '[mpo_qphys.merge]'
      print ' partition = ',partition
      print ' qphys = ',qphys
   if qphys is None: return None
   qphys_new = []
   for ipart in partition:
      npart = len(ipart)
      qnew = qphys[ipart[0]]
      for j in range(1,npart):
         qnew = dpt(qnew,qphys[ipart[j]])
      if debug:
	 print ' >>> ipart = ',ipart
	 print ' qnew = ',qnew
      qphys_new.append(qnew)
   return qphys_new

if __name__ == '__main__':
   partition = [[0, 1, 2, 3], [4, 5], [6], [7]]
   
   # Generate a new set of local quantum numbers
   qphys = init(8)
   prt(qphys)
   qnew = merge(qphys,partition)
   for idx,iqnum in enumerate(qnew):
      print ' idx=',idx,iqnum
   
   qphys = init(8,isym=2)
   prt(qphys)
   qnew = merge(qphys,partition)
   for idx,iqnum in enumerate(qnew):
      print ' idx=',idx,iqnum
