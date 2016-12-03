#!/usr/bin/env python
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def vacuum(isym=1):
# def initSpinOrb(norb,isym=1):
# def initSpatialOrb(norb,isym=1):
# def prt(qphys):
# def dpt(q1,q2):
# def merge(qphys,partition,debug=False):
# 
import itertools

# Vacuum
def vacuum(isym=1):
   # (N) - Particle number
   if isym == 1:
      qvac = [[0.]]
   # (N,Sz) - Spin projection 
   elif isym == 2:
      qvac = [[0.,0.]]
   else:
      qvac = None
   return qvac

# Spin-orbitals  
def initSpinOrb(norb,isym=1):
   if isym == 0:
      lst = None		
   elif isym == 1:
      lst = [[[0.],[1.]]]*norb
   elif isym == 2:
      assert norb%2 == 0	   
      lst = [[[0.,0.],[1.,0.5]],[[0.,0.],[1.,-0.5]]]*(norb/2)
   else:
      print 'error in mpo_dmrg_qphys.initSpinOrb: no such isym=',isym
      exit(1)
   return lst

def initSpatialOrb(norb,isym=1):
   if isym == 0:
      lst = None
   else:
      lst = initSpinOrb(norb*2,isym)
      partition = [[2*i,2*i+1] for i in range(norb)]
      lst = merge(lst,partition)
   return lst

def prt(qphys):
   n = 50
   nsite = len(qphys)
   print '[mpo_dmrg_qphys.prt]'
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
      print '[mpo_dmrg_qphys.merge]'
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
   qphys = initSpinOrb(8)
   prt(qphys)
   qnew = merge(qphys,partition)
   for idx,iqnum in enumerate(qnew):
      print ' idx=',idx,iqnum
   
   qphys = initSpinOrb(8,isym=2)
   prt(qphys)
   qnew = merge(qphys,partition)
   for idx,iqnum in enumerate(qnew):
      print ' idx=',idx,iqnum

   print
   print initSpinOrb(4,isym=2)
   print initSpinOrb(4,isym=1)
   print initSpatialOrb(4,isym=2)
   print initSpatialOrb(4,isym=1)
