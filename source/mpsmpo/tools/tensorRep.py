#===========================================
# Convert from various representation of 
# the Full CI cofficients:
# C[Ia,Ib]
# -->C[I] - C[K,N] spin orbital rep.
# -->c_T  - K^N: distinguishable particles 
# -->c_ON - 2^K: fermion Fock space
# these three formats are stored as 1Darray
#===========================================
import itools
import numpy
import scipy.linalg
import math
import misc
from pyscf import fci

#==========================================
# Strings 
#==========================================

def bit2string(intx,norb):
   conf=[]
   for i in range(norb):
      if(intx & (1 << i)): conf.append(i)
   return conf

def string2bit(confx):
   s=0
   for i in confx:
      s = (1<<i) | s
   return s
#string2bit([0,3,5])

def string_sgn(string):
   n=len(string)
   sgn=1.0
   for i in range(n):
      for j in range(i+1,n):
         if string[i]>string[j]: sgn=-sgn	      
   return sgn

# FROM QIMING'S CODE:
def num_strings(n, m):
   return math.factorial(n)//(math.factorial(n-m)*math.factorial(m))

def str2addr_o1(norb, nelec, string):
    #TODO: assert norb > first-bit-in-string, nelec == num-1-in-string
    addr = 0
    nelec_left = nelec
    for norb_left in reversed(range(norb)):
        if nelec_left == 0 or norb_left < nelec_left:
            break
        elif (1<<norb_left) & string:
            addr += num_strings(norb_left, nelec_left)
            nelec_left -= 1
    return addr

def addr2str_o1(norb, nelec, addr):
    assert(num_strings(norb, nelec) > addr)
    if addr == 0 or nelec == norb or nelec == 0:
        return (1<<nelec) - 1   # ..0011..11
    str1 = 0
    nelec_left = nelec
    for norb_left in reversed(range(norb)):
        addrcum = num_strings(norb_left, nelec_left)
        if nelec_left == 0:
            break
        elif addr == 0:
            str1 |= (1<<nelec_left) - 1
            break
        elif addrcum <= addr:
            str1 |= 1<<norb_left
            addr -= addrcum
            nelec_left -= 1
    return str1

#==========================================
# CI coefficients 
#==========================================

def toCIanalyze(norb,ne,civec):
   print '\n[toCIanalyze]: analyze CI_coeff'
   assert civec.ndim == 2
   print 'norb=',norb,' ne=',ne
   mrkA=min(norb-ne[0],ne[0])
   mrkB=min(norb-ne[1],ne[1])
   print 'mrkA=',mrkA,' mrkB=',mrkB
   # HF reference
   refA=2**ne[0]-1
   refB=2**ne[1]-1
   nsA,nsB=civec.shape
   mrk=mrkA+mrkB
   rCount=[0]*(mrk+1)
   cCount=[0.0]*(mrk+1)
   for isA in range(nsA):
      bsA=addr2str_o1(norb,ne[0],isA)
      # 000111 & 101001 => no. of bit that is not moved. 
      rkA=ne[0]-bin(bsA&refA).count("1")
      for isB in range(nsB):
	 bsB=addr2str_o1(norb,ne[1],isB)
	 rkB=ne[1]-bin(bsB&refB).count("1")
	 rCount[rkA+rkB]+=1
	 cCount[rkA+rkB]+=civec[isA,isB]**2
   # PRINT
   print "Statistics of CI coeff:"
   for rc in enumerate(zip(rCount,cCount)):
      print 'rank=',rc[0],' stat=',rc[1]
   print "sum of rCount =",sum(rCount)
   print "sum of cCount =",sum(cCount)

def toCIspinorb(norb,ne,civec):
   print '\n[toCIspinorb]: spin-orbit rep C(Ia,Ib)->C(I)'
   nsA,nsB= civec.shape
   nelec  = ne[0]+ne[1]
   nsorb  = 2*norb
   ndim   = misc.binomial(nsorb,nelec)
   print 'nsA,nsB=',nsA,nsB
   print 'nstring=',ndim
   print 'nfactor=',ndim*1.0/(nsA*nsB)
   civec2=numpy.zeros(ndim)
   for isA in range(nsA):
      bsA =addr2str_o1(norb,ne[0],isA)
      strA=bit2string(bsA,norb)
      strA=[2*i for i in strA]
      for isB in range(nsB):
         bsB =addr2str_o1(norb,ne[1],isB)
         strB=bit2string(bsB,norb)
	 strB=[2*i+1 for i in strB]
	 # conversion
	 string=strA+strB
	 sgn = string_sgn(string)
	 string.sort()
	 #print string2bit(string)
	 #print string
	 #print civec[isA,isB]
	 #print addr
	 #addr=fci.cistring.str2addr_o1(nsorb,nelec,string2bit(string))
	 addr=str2addr_o1(nsorb,nelec,string2bit(string))
	 #print nsorb,string,sgn,addr,bin(string2bit(string))
	 civec2[addr]=civec[isA,isB]*sgn
   #print civec2[numpy.argwhere(civec2>1.e-10)]
   norm=numpy.linalg.norm(civec2)
   print "NORM1=",norm
   if(abs(norm-1.0)>1.e-6): exit(1)
   return civec2 
 
def toCItensor(nsorb,nelec,civec,iop=0):
   print '\n[toCItensor]: iop=',iop
   ndim = misc.binomial(nsorb,nelec)
   print 'nsorb=',nsorb
   print 'nelec=',nelec
   print 'C_K^N=',ndim,civec.size
   if ndim != civec.size: 
      print 'ndim='
      print 'civec.size='
      exit(1)
   citensor=numpy.zeros(nsorb**nelec)
   print 'd_K^N=',citensor.shape
   stride=[nsorb**(nelec-i-1) for i in range(nelec)]
   if iop==0:
      for i in range(civec.size):
         bstring=bin(addr2str_o1(nsorb,nelec,i))
         orblst=tuple(bit2string( int(bstring,2),nsorb ))
	 addr  =numpy.dot(orblst,stride)
         citensor[addr]=civec[i]
   else:
      for i in range(civec.size):
         bstring=bin(addr2str_o1(nsorb,nelec,i))
         orblst=tuple(bit2string( int(bstring,2),nsorb ))
	 for j in itools.permutations(orblst,nelec):
	    addr=numpy.dot(j,stride)
	    citensor[addr]=string_sgn(j)*civec[i]
      fac=1.0/math.sqrt(math.factorial(nelec))
      citensor=citensor*fac
   print 'NORM of Tensor=',numpy.linalg.norm(citensor)
   return citensor

def toCItensor2(nsorb,nelec,civec):
   print '\n[toCItensor2]'
   ndim = misc.binomial(nsorb,nelec)
   print 'nsorb=',nsorb
   print 'nelec=',nelec
   print 'C_K^N=',ndim,civec.size
   if ndim != civec.size: 
      print 'ndim='
      print 'civec.size='
      exit(1)
   # phys=K-N+1   
   nphys=nsorb-nelec+1
   citensor=numpy.zeros(nphys**nelec)
   print 'd_K^N=',citensor.shape
   stride=[nphys**(nelec-i-1) for i in range(nelec)]
   offset=numpy.array(range(nelec))
   for i in range(civec.size):
      bstring=bin(addr2str_o1(nsorb,nelec,i))
      orblst0=numpy.array(bit2string( int(bstring,2),nsorb ))
      orblst1=orblst0-offset
      addr  =numpy.dot(orblst1,stride)
      #print orblst0,orblst1,addr
      citensor[addr]=civec[i]
   print 'NORM of Tensor=',numpy.linalg.norm(citensor)
   return citensor

def toONtensor(nsorb,nelec,civec):
   print '\n[toONtensor]:'
   ndim = misc.binomial(nsorb,nelec)
   print 'nsorb=',nsorb
   print 'nelec=',nelec
   print 'C_K^N=',ndim,civec.size
   if ndim != civec.size: exit(1)
   ontensor=numpy.zeros(2**nsorb)
   print 'd_2^K=',ontensor.shape
   for i in range(civec.size):
      bstring=bin(addr2str_o1(nsorb,nelec,i))
      addr=int(bstring,2)
      print 'idx=',i,bstring,addr
      # idx= 0 0b1 1
      # idx= 1 0b10 2
      # idx= 2 0b100 4
      ontensor[addr]=civec[i]
   print 'NORM of Tensor=',numpy.linalg.norm(ontensor)
   return ontensor

#==========================================
# Reverse function
#==========================================
def toCIspinorbReverse(norb,ne,civec2):
   print '\n[toCIspinorbReverse]: spin-orbit rep C(Ia,Ib)<-C(I)'
   nsA = misc.binomial(norb,ne[0])
   nsB = misc.binomial(norb,ne[1])
   nelec = ne[0]+ne[1]
   nsorb = 2*norb
   print 'nsA,nsB=',nsA,nsB
   civec=numpy.zeros((nsA,nsB))
   for isA in range(nsA):
      bsA =addr2str_o1(norb,ne[0],isA)
      strA=bit2string(bsA,norb)
      strA=[2*i for i in strA]
      for isB in range(nsB):
         bsB =addr2str_o1(norb,ne[1],isB)
         strB=bit2string(bsB,norb)
	 strB=[2*i+1 for i in strB]
	 # conversion
	 string=strA+strB
	 sgn = string_sgn(string)
	 string.sort()
	 addr=str2addr_o1(nsorb,nelec,string2bit(string))
	 civec[isA,isB]=civec2[addr]*sgn
   #print civec2[numpy.argwhere(civec2>1.e-10)]
   norm=numpy.linalg.norm(civec)
   print "NORM1=",norm
   #if(abs(norm-1.0)>1.e-6): exit(1)
   return civec 

def toCItensorReverse(nsorb,nelec,citensor,iop=0):
   #print '\n[toCItensorReverse]: iop=',iop
   ndim = misc.binomial(nsorb,nelec)
   #print 'nsorb=',nsorb
   #print 'nelec=',nelec
   #print 'C_K^N=',ndim
   civec=numpy.zeros(ndim)
   stride=[nsorb**(nelec-i-1) for i in range(nelec)]
   for i in range(civec.size):
      bstring=bin(addr2str_o1(nsorb,nelec,i))
      orblst=tuple(bit2string( int(bstring,2),nsorb ))
      addr  =numpy.dot(orblst,stride)
      civec[i]=citensor[addr]
   if iop == 1:
      fac=math.sqrt(math.factorial(nelec))
      civec=civec*fac
   #print 'NORM of Tensor=',numpy.linalg.norm(civec)
   return civec

def toONtensorReverse(nsorb,nelec,ontensor):
   print '\n[toONtensorReverse]:'
   ndim = misc.binomial(nsorb,nelec)
   print 'nsorb=',nsorb
   print 'nelec=',nelec
   print 'C_K^N=',ndim
   civec=numpy.zeros(ndim)
   for i in range(civec.size):
      bstring=bin(addr2str_o1(nsorb,nelec,i))
      addr=int(bstring,2)
      civec[i]=ontensor[addr]
   print 'NORM of Tensor=',numpy.linalg.norm(civec)
   return civec
