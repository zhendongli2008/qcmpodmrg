#
# Direct construciton of MPO for H and T via W factors in SPIN-ORBITAL
#
# Examples: current limitation is about 60spinorbs = 30Hatoms 
#	    MEMORY BOUND - O(K^5) scaling
# 	    W[k] needs to be dumped into disk.
#
import time
import copy
import numpy
import scipy.linalg
import mpo_class
import mpo_consw
from tools import itools

# General a mpo_class object for H' = H - E0 
def genHmpo(h1e,h2e,e0=0.0,partition=None,isym=0):
   print '\n[mpo_direct.genHmpo]'
   if partition is None:
      dimSpinOrb = h1e.shape[0]
      partition = [[2*i,2*i+1] for i in range(dimSpinOrb/2)]
   aeri = antisymmetrizeTwoBodyOpers(h2e)
   result = directHmpo(h1e,aeri,e0,partition,isym=isym)
   return result


# From storage of int2e to reduced ERI defined as follow:
# Unique representation: V[ijkl] (i<j,k<l) = -<ij||kl>
def antisymmetrizeTwoBodyOpers(eriA):
   sbas = eriA.shape[0]
   aeri = numpy.zeros(eriA.shape)
   for j in range(sbas):
      for i in range(j):
         for l in range(sbas):
            for k in range(l):
               aeri[i,j,k,l] = 2.0*eriA[i,j,k,l]
   return aeri


def partFlatten(partition):
   support = [] 
   for ipart in partition:
      support += ipart
   return support


def genTablesLCR(partition,debug=False):
   ngroups = len(partition)
   tablcr = []
   tabspt = []
   tabdim = []
   for igroup in range(ngroups):
      # Partition of the basis
      lg = partition[:igroup]
      cg = [partition[igroup]]
      rg = partition[igroup+1:]
      sl = partFlatten(lg) 
      sc = partFlatten(cg)
      sr = partFlatten(rg)
      # cardinality of basis
      nl = len(sl)
      nc = len(sc)
      nr = len(sr)
      tablcr.append([lg,cg,rg])
      tabspt.append([sl,sc,sr])
      tabdim.append([nl,nc,nr])
      if debug:
         print
         print ' igroup =',igroup
         print ' lg = ',lg
         print ' cg = ',cg
         print ' rg = ',rg
         print ' sl = ',sl
         print ' sc = ',sc
         print ' sr = ',sr
         print ' nl,nc,nr = ',nl,nc,nr
   return tablcr,tabspt,tabdim
 

def genTablesDim12(tabdim,igroup):
   lg,cg,rg = tabdim[igroup]
   lg2 = lg*(lg-1)/2
   cg2 = cg*(cg-1)/2
   rg2 = rg*(rg-1)/2
   dims1l = [1,cg,rg,cg,rg,cg2,cg*rg,rg2,cg2,cg*rg,rg2,lg,lg**2,lg,lg,1]
   dims1s = [1,cg+rg,cg+rg,cg2+cg*rg+rg2,cg2+cg*rg+rg2,lg,lg**2,lg,lg,1]
   dims2l = [1,rg,rg,rg2,rg2,lg,cg,lg**2,lg*cg,lg*cg,cg**2,lg,cg,lg,cg,1]
   dims2s = [1,rg,rg,rg2,rg2,lg+cg,lg**2+lg*cg+lg*cg+cg**2,lg+cg,lg+cg,1]
   return lg,cg,rg,dims1l,dims1s,dims2l,dims2s 


def genTablesWdims(tabdim,debug=False):
   #
   # W[a1]*W[a1]
   # W[a1]*W[a1,a2]*W[a2]
   # W[a1]*W[a1,a2]*W[a2,a3]*W[a3]	
   #
   ngroups = len(tabdim)
   wdims = []
   for igroup in range(ngroups):
      lg,cg,rg,dims1l,dims1s,dims2l,dims2s = genTablesDim12(tabdim,igroup)
      dim1 = numpy.sum(dims1l)
      dim2 = numpy.sum(dims2l)
      wdims.append(dim2)
      if debug:
         print 'igroup',igroup,'dimlcr=',(lg,cg,rg),
         print 'dim[%d]='%(igroup-1),dim1,'dim[%d]='%igroup,dim2
         print ' d1l=',dims1l
         print ' d1s=',dims1s
         print ' d2l=',dims2l
         print ' d2s=',dims2s
   wdims.pop()
   assert len(wdims) == ngroups-1
   if debug: print 'Final wdims=',wdims
   return wdims


def sortPcr(cg,rg):
   def id(i,j,n):
      return i*(2*n-i-3)/2+j-1
   n = cg+rg
   ijcc = [id(i,j,n) for i in range(cg) for j in range(cg) if i<j]
   ijcr = [id(i,j+cg,n) for i in range(cg) for j in range(rg)]
   ijrr = [id(i+cg,j+cg,n) for i in range(rg) for j in range(rg) if i<j]
   ij = ijcc+ijcr+ijrr
   ij = numpy.array(ij)
   ijdx = numpy.argsort(ij)
   return ijdx


def sortQlc(lg,cg):
   def id(i,j,n):
      return i*n+j
   n = lg+cg
   ijll = [id(i,j,n) for i in range(lg) for j in range(lg)]
   ijlc = [id(i,j+lg,n) for i in range(lg) for j in range(cg)]
   ijcl = [id(i+lg,j,n) for i in range(cg) for j in range(lg)]
   ijcc = [id(i+lg,j+lg,n) for i in range(cg) for j in range(cg)]
   ij = ijll+ijlc+ijcl+ijcc
   ij = numpy.array(ij)
   ijdx = numpy.argsort(ij)
   return ijdx


#
# Kernel
#
def directHmpo(h1e,h2e,e0,partition=None,debug=False,iprt=0,isym=0):
   if iprt>0: print '\n[directHmpo]'
   if debug:
      print ' h1e.shape=',h1e.shape
      print ' h2e.shape=',h2e.shape
   k = h1e.shape[0]
   if partition is None:
      partition = [[i] for i in range(k)]
   fsupport = partFlatten(partition)
   assert len(fsupport) == k
   assert fsupport[-1] == k-1
   ngroups = len(partition)
   # Special case 
   if ngroups == 1:
      sl = []
      sc = fsupport
      sr = []
      tmp = mpo_consw.l1r16(h1e,h2e,sl,sc,sr)
      nc1,nc2 = tmp.shape
      tmp = tmp.reshape((1,1,nc1,nc2))
      hmpo = mpo_class.class_mpo(ngroups,[tmp])
      return hmpo 
   ndims = map(lambda x:len(x),partition)
   if debug:
      print ' partition=',partition
      print ' ngroups  =',ngroups
      print ' ndims    =',ndims
      print ' fsupport =',fsupport
   #   
   # LOOP over groups
   #
   tablcr,tabspt,tabdim = genTablesLCR(partition,debug)
   wdims = genTablesWdims(tabdim,debug)
   wdims = [1]+wdims+[1]
   wfacs = [0]*ngroups
   qnums = [0]*ngroups
   if debug: 
      print
      print ' tablcr =',tablcr
      print ' tabspt =',tabspt
      print ' tabdim =',tabdim
      print ' wdims  =',wdims
   for igroup in range(ngroups):
      diml = wdims[igroup]
      dimr = wdims[igroup+1]
      lg,cg,rg,dims1l,dims1s,dims2l,dims2s = genTablesDim12(tabdim,igroup)
      sl,sc,sr = tabspt[igroup] 
      wtmp = numpy.zeros((diml,dimr,2**cg,2**cg))
      a1 = [0]+list(itools.accumulate(dims1l)) 
      a2 = [0]+list(itools.accumulate(dims2l))
      if debug: 
	 print
	 print 'igroup=',igroup,diml,dimr
         print ' wfacs[i]=',wtmp.shape
         print ' sl,sc,sr=',sl,sc,sr
         print ' lg,cg,rg=',(lg,cg,rg)
         print ' dims1l = ',len(dims1l),dims1l
         print ' dims2l = ',len(dims2l),dims2l
	 print ' accum1 = ',len(a1),a1
	 print ' accum2 = ',len(a2),a2
      if igroup == 0:
	 # row-1: match last index for a1,a2
	 identity = mpo_consw.l1r1(h1e,h2e,sl,sc,sr)
	 wtmp[0:1,a2[0]:a2[1]] = identity.copy() 
	 wtmp[0:1,a2[1]:a2[2]] = mpo_consw.l1r2(h1e,h2e,sl,sc,sr) 
	 wtmp[0:1,a2[2]:a2[3]] = mpo_consw.l1r3(h1e,h2e,sl,sc,sr)
	 if a2[3]<a2[4]:
	    wtmp[0:1,a2[3]:a2[4]] = mpo_consw.l1r4(h1e,h2e,sl,sc,sr)
	 if a2[4]<a2[5]:
            wtmp[0:1,a2[4]:a2[5]] = mpo_consw.l1r5(h1e,h2e,sl,sc,sr)
         wtmp[0:1,a2[6]:a2[7]] = mpo_consw.l1r7(h1e,h2e,sl,sc,sr)
	 wtmp[0:1,a2[10]:a2[11]] = mpo_consw.l1r11(h1e,h2e,sl,sc,sr) # Qterm
	 wtmp[0:1,a2[12]:a2[13]] = mpo_consw.l1r13(h1e,h2e,sl,sc,sr)
         wtmp[0:1,a2[14]:a2[15]] = mpo_consw.l1r15(h1e,h2e,sl,sc,sr)
         wtmp[0:1,a2[15]:a2[16]] = mpo_consw.l1r16(h1e,h2e,sl,sc,sr) \
			 	 - e0*identity
         #----------------------------------------------
	 # Reordering of Qterm
	 ijdx = sortQlc(lg,cg)+a2[7]
	 assert len(ijdx)==a2[11]-a2[7]
	 wtmp[:,a2[7]:a2[11]] = wtmp[:,ijdx].copy()
         #----------------------------------------------
      elif igroup == ngroups-1:
         # col-1
	 wtmp[a1[0]:a1[1]  ,0:1] = mpo_consw.l1r16(h1e,h2e,sl,sc,sr)
	 wtmp[a1[1]:a1[2]  ,0:1] = mpo_consw.l2r16(h1e,h2e,sl,sc,sr)
         wtmp[a1[3]:a1[4]  ,0:1] = mpo_consw.l4r16(h1e,h2e,sl,sc,sr)
         if a1[5] < a1[6]: 
	    wtmp[a1[5]:a1[6],0:1] = mpo_consw.l6r16(h1e,h2e,sl,sc,sr)
	 if a1[8] < a1[9]:
	    wtmp[a1[8]:a1[9],0:1] = mpo_consw.l9r16(h1e,h2e,sl,sc,sr)
	 wtmp[a1[11]:a1[12],0:1] = mpo_consw.l12r16(h1e,h2e,sl,sc,sr)
         wtmp[a1[12]:a1[13],0:1] = mpo_consw.l13r16(h1e,h2e,sl,sc,sr)
         wtmp[a1[13]:a1[14],0:1] = mpo_consw.l14r16(h1e,h2e,sl,sc,sr)
         wtmp[a1[14]:a1[15],0:1] = mpo_consw.l15r16(h1e,h2e,sl,sc,sr)
         wtmp[a1[15]:a1[16],0:1] = mpo_consw.l16r16(h1e,h2e,sl,sc,sr)
         #----------------------------------------------
	 # Reordering of Pterm
	 ijdx = sortPcr(cg,rg)
	 assert len(ijdx)==a1[8]-a1[5]
	 wtmp[a1[5]:a1[8],:] = wtmp[ijdx+a1[5],:].copy()
	 assert len(ijdx)==a1[11]-a1[8]
	 wtmp[a1[8]:a1[11],:] = wtmp[ijdx+a1[8],:].copy()
         #----------------------------------------------
      else:
	 # row-1 => H 
	 wtmp[a1[0]:a1[1],a2[0]:a2[1]] = mpo_consw.l1r1(h1e,h2e,sl,sc,sr)
	 wtmp[a1[0]:a1[1],a2[1]:a2[2]] = mpo_consw.l1r2(h1e,h2e,sl,sc,sr)
	 wtmp[a1[0]:a1[1],a2[2]:a2[3]] = mpo_consw.l1r3(h1e,h2e,sl,sc,sr)
	 if a2[3]<a2[4]:
	    wtmp[a1[0]:a1[1],a2[3]:a2[4]] = mpo_consw.l1r4(h1e,h2e,sl,sc,sr)
	 if a2[4]<a2[5]:
            wtmp[a1[0]:a1[1],a2[4]:a2[5]] = mpo_consw.l1r5(h1e,h2e,sl,sc,sr)
         wtmp[a1[0]:a1[1],a2[6]:a2[7]] = mpo_consw.l1r7(h1e,h2e,sl,sc,sr)
         wtmp[a1[0]:a1[1],a2[10]:a2[11]] = mpo_consw.l1r11(h1e,h2e,sl,sc,sr)
         wtmp[a1[0]:a1[1],a2[12]:a2[13]] = mpo_consw.l1r13(h1e,h2e,sl,sc,sr)
         wtmp[a1[0]:a1[1],a2[14]:a2[15]] = mpo_consw.l1r15(h1e,h2e,sl,sc,sr)
         wtmp[a1[0]:a1[1],a2[15]:a2[16]] = mpo_consw.l1r16(h1e,h2e,sl,sc,sr)
	 # row-2 => a+[i]
	 wtmp[a1[1]:a1[2],a2[15]:a2[16]] = mpo_consw.l2r16(h1e,h2e,sl,sc,sr)
	 # row-3 => a+[i+1,K]
	 wtmp[a1[2]:a1[3],a2[1]:a2[2]] = mpo_consw.l3r2(h1e,h2e,sl,sc,sr)
	 # row-4 => a[i]
	 wtmp[a1[3]:a1[4],a2[15]:a2[16]] = mpo_consw.l4r16(h1e,h2e,sl,sc,sr)
	 # row-5 => a[i+1,K]
	 wtmp[a1[4]:a1[5],a2[2]:a2[3]] = mpo_consw.l5r3(h1e,h2e,sl,sc,sr)
	 # row-6 => A+[i][i]
	 if a1[5] < a1[6]: 
	    wtmp[a1[5]:a1[6],a2[15]:a2[16]] = mpo_consw.l6r16(h1e,h2e,sl,sc,sr)
	 # row-7 => A+[i][i+1,K]
	 wtmp[a1[6]:a1[7],a2[1]:a2[2]] = mpo_consw.l7r2(h1e,h2e,sl,sc,sr)
	 # row-8 => A+[i+1,K][i+1,K]
	 if a1[7] < a1[8]:
	    wtmp[a1[7]:a1[8],a2[3]:a2[4]] = mpo_consw.l8r4(h1e,h2e,sl,sc,sr)
	 # row-9 => A[i][i]
	 if a1[8] < a1[9]:
	    wtmp[a1[8]:a1[9],a2[15]:a2[16]] = mpo_consw.l9r16(h1e,h2e,sl,sc,sr)
	 # row-10 => A[i][i+1,K]
	 wtmp[a1[9]:a1[10],a2[2]:a2[3]] = mpo_consw.l10r3(h1e,h2e,sl,sc,sr)
	 # row-11 => A[i+1,K]A[i+1,K]
	 if a1[10]<a1[11]:
	    wtmp[a1[10]:a1[11],a2[4]:a2[5]] = mpo_consw.l11r5(h1e,h2e,sl,sc,sr)
	 # row-12 => S[1,i-1]
	 wtmp[a1[11]:a1[12],a2[5]:a2[6]] = mpo_consw.l12r6(h1e,h2e,sl,sc,sr)
	 wtmp[a1[11]:a1[12],a2[15]:a2[16]] = mpo_consw.l12r16(h1e,h2e,sl,sc,sr)
	 # row-13 => Q[1,i-1][1,i-1]
	 wtmp[a1[12]:a1[13],a2[1]:a2[2]] = mpo_consw.l13r2(h1e,h2e,sl,sc,sr)
	 wtmp[a1[12]:a1[13],a2[2]:a2[3]] = mpo_consw.l13r3(h1e,h2e,sl,sc,sr)
	 wtmp[a1[12]:a1[13],a2[7]:a2[8]] = mpo_consw.l13r8(h1e,h2e,sl,sc,sr)
	 wtmp[a1[12]:a1[13],a2[15]:a2[16]] = mpo_consw.l13r16(h1e,h2e,sl,sc,sr)
	 # row-14 => T1[1,i-1]
	 wtmp[a1[13]:a1[14],a2[1]:a2[2]] = mpo_consw.l14r2(h1e,h2e,sl,sc,sr)
	 wtmp[a1[13]:a1[14],a2[2]:a2[3]] = mpo_consw.l14r3(h1e,h2e,sl,sc,sr)
	 if a2[4]<a2[5]:
	    wtmp[a1[13]:a1[14],a2[4]:a2[5]] = mpo_consw.l14r5(h1e,h2e,sl,sc,sr)
	 wtmp[a1[13]:a1[14],a2[8]:a2[9]] = mpo_consw.l14r9(h1e,h2e,sl,sc,sr)
	 wtmp[a1[13]:a1[14],a2[11]:a2[12]] = mpo_consw.l14r12(h1e,h2e,sl,sc,sr)
	 wtmp[a1[13]:a1[14],a2[15]:a2[16]] = mpo_consw.l14r16(h1e,h2e,sl,sc,sr)
	 # row-15 => T3[1,i-1]
	 wtmp[a1[14]:a1[15],a2[1]:a2[2]] = mpo_consw.l15r2(h1e,h2e,sl,sc,sr)
	 wtmp[a1[14]:a1[15],a2[2]:a2[3]] = mpo_consw.l15r3(h1e,h2e,sl,sc,sr)
	 if a2[3]<a2[4]:
	    wtmp[a1[14]:a1[15],a2[3]:a2[4]] = mpo_consw.l15r4(h1e,h2e,sl,sc,sr)
	 wtmp[a1[14]:a1[15],a2[9]:a2[10]] = mpo_consw.l15r10(h1e,h2e,sl,sc,sr)
	 wtmp[a1[14]:a1[15],a2[13]:a2[14]] = mpo_consw.l15r14(h1e,h2e,sl,sc,sr)
	 wtmp[a1[14]:a1[15],a2[15]:a2[16]] = mpo_consw.l15r16(h1e,h2e,sl,sc,sr)
	 # row-16 => I[i,K]
	 wtmp[a1[15]:a1[16],a2[15]:a2[16]] = mpo_consw.l16r16(h1e,h2e,sl,sc,sr)
         #----------------------------------------------
	 # Reordering of Qterm
	 ijdx = sortQlc(lg,cg)+a2[7]
	 assert len(ijdx)==a2[11]-a2[7]
	 wtmp[:,a2[7]:a2[11]] = wtmp[:,ijdx].copy()
         #----------------------------------------------
	 # Reordering of Pterm
	 ijdx = sortPcr(cg,rg)
	 assert len(ijdx)==a1[8]-a1[5]
	 wtmp[a1[5]:a1[8],:] = wtmp[ijdx+a1[5],:].copy()
	 assert len(ijdx)==a1[11]-a1[8]
	 wtmp[a1[8]:a1[11],:] = wtmp[ijdx+a1[8],:].copy()
         #----------------------------------------------
      # Store
      wfacs[igroup] = wtmp.copy()
      #----------------------------------------------
      # Qnumbers
      #----------------------------------------------
      qtmp = [None]*dimr
      if isym == 1:
         qtmp[a2[0] :a2[1]]  = [ [0]  ]*(a2[1] -a2[0] )
         qtmp[a2[1] :a2[2]]  = [ [-1] ]*(a2[2] -a2[1] )
         qtmp[a2[2] :a2[3]]  = [ [+1] ]*(a2[3] -a2[2] )
         qtmp[a2[3] :a2[4]]  = [ [-2] ]*(a2[4] -a2[3] )
         qtmp[a2[4] :a2[5]]  = [ [+2] ]*(a2[5] -a2[4] )
         qtmp[a2[5] :a2[7]]  = [ [-1] ]*(a2[7] -a2[5] )
         qtmp[a2[7] :a2[11]] = [ [0]  ]*(a2[11]-a2[7] )
         qtmp[a2[11]:a2[13]] = [ [+1] ]*(a2[13]-a2[11])
         qtmp[a2[13]:a2[15]] = [ [-1] ]*(a2[15]-a2[13])
         qtmp[a2[15]:a2[16]] = [ [0]  ]*(a2[16]-a2[15])
	 if igroup == ngroups-1:
	    qnums[igroup] = [[0]]
 	 else:
            qnums[igroup] = copy.deepcopy(qtmp)
      elif isym == 2:
	 # Only works for spin-orbital case
	 assert k == ngroups
	 if igroup%2 == 0:
	    sz = 0.5
	 else:
	    sz = -0.5
         qtmp[a2[0] :a2[1]]  = [ [0,0]     ]*(a2[1] -a2[0] )
         qtmp[a2[1] :a2[2]]  = [ [-1,-sz]  ]*(a2[2] -a2[1] ) # T2
         qtmp[a2[2] :a2[3]]  = [ [+1,sz]   ]*(a2[3] -a2[2] ) # S+T4
	 qtmp[a2[3] :a2[4]]  = [ [-2,-2*sz]]*(a2[4] -a2[3] ) # P2ann: In fact, they will 
	 qtmp[a2[4] :a2[5]]  = [ [+2,2*sz] ]*(a2[5] -a2[4] ) # P2cre: not exisit at all.
         qtmp[a2[5] :a2[7]]  = [ [-1,-sz]  ]*(a2[7] -a2[5] )
         qtmp[a2[7] :a2[11]] = [ [0,0]     ]*(a2[11]-a2[7] )	
         qtmp[a2[11]:a2[13]] = [ [+1,sz]   ]*(a2[13]-a2[11])
         qtmp[a2[13]:a2[15]] = [ [-1,-sz]  ]*(a2[15]-a2[13])
         qtmp[a2[15]:a2[16]] = [ [0,0]     ]*(a2[16]-a2[15])
	 if igroup == ngroups-1:
	    qnums[igroup] = [[0,0]]
 	 else:
	    qnums[igroup] = copy.deepcopy(qtmp)
      #----------------------------------------------
   # Finally, form MPO   
   if iprt>0: print ' wdims=',wdims
   #if isym == 0: 
   #   hmpo = mpo_class.class_mpo(ngroups,wfacs)
   #else:
   #   hmpo = mpo_class.class_mpo(ngroups,wfacs,qnums)
   #return hmpo
   return ngroups,wfacs,qnums

#
# Approximation to exp(H*x)
# We associate beta factor to D and C
#
def polyH(hmpo,xfac=1.0):
   print '\n[polyH]'
   N = hmpo.nsite
   expHx = [0]*N
   if hmpo.qnums is None: 
      ifqnum = False
      qnums = None 
   else:
      ifqnum = True 
      qnums = [0]*N
   #
   # site-0
   #
   tmp = hmpo.sites[0]
   diml,dimr,nc1,nc2=tmp.shape 
   assert diml == 1
   assert dimr > 2 # assuming C always exists
   i1 = tmp[0,0].copy()
   c1 = tmp[0,1:dimr-1].copy()
   d1 = tmp[0,dimr-1].copy()
   c1 = xfac*c1
   d1 = xfac*d1
   tmp2 = numpy.zeros((diml,dimr-1,nc1,nc2))
   tmp2[0,0] = i1+d1
   tmp2[0,1:] = c1.copy()
   expHx[0] = tmp2.copy()
   if ifqnum: qnums[0] = copy.deepcopy(hmpo.qnums[0][:-1]) 
   #
   # site-(N-1)
   #
   tmp = hmpo.sites[N-1].copy()
   diml,dimr,nc1,nc2=tmp.shape
   assert dimr == 1
   assert diml > 2
   d1 = tmp[0,0].copy()
   b1 = tmp[1:diml-1,0].copy()
   i1 = tmp[diml-1,0].copy()
   d1 = xfac*d1
   tmp2 = numpy.zeros((diml-1,dimr,nc1,nc2))
   tmp2[0,0] = i1+d1
   tmp2[1:,0] = b1.copy()
   expHx[N-1] = tmp2.copy()
   if ifqnum: qnums[N-1] = copy.deepcopy(hmpo.qnums[N-1]) 
   #
   # site-(1,N-2)
   #
   for isite in range(1,N-1):
      tmp = hmpo.sites[isite]
      diml,dimr,nc1,nc2=tmp.shape
      i1 = tmp[0,0].copy()
      c1 = tmp[0,1:dimr-1].copy()
      d1 = tmp[0,dimr-1].copy()
      a1 = tmp[1:diml-1,1:dimr-1].copy()
      b1 = tmp[1:diml-1,dimr-1].copy()
      i2 = tmp[diml-1,dimr-1].copy()
      c1 = xfac*c1
      d1 = xfac*d1
      assert numpy.linalg.norm(i1-i2)<1.e-10
      tmp2 = numpy.zeros((diml-1,dimr-1,nc1,nc2))
      tmp2[0,0] = i1+d1
      tmp2[0,1:] = c1.copy()
      tmp2[1:,0] = b1.copy()
      tmp2[1:,1:] = a1.copy()  
      expHx[isite]= tmp2.copy()
      if ifqnum: qnums[isite] = copy.deepcopy(hmpo.qnums[isite][:-1])
   empo = mpo_class.class_mpo(N,expHx,qnums)
   return empo

#
# Exact representation of 1+x*H
# We associate beta factor to D and C
#
def linearH(hmpo,xfac=1.0):
   print '\n[linearH]'
   N = hmpo.nsite
   expHx = [0]*N
   if hmpo.qnums is None: 
      ifqnum = False
      qnums = None 
   else:
      ifqnum = True 
      qnums = copy.deepcopy(hmpo.qnums)
   #
   # site-0: [I, tC, tD+I]
   #
   tmp = hmpo.sites[0]
   diml,dimr,nc1,nc2=tmp.shape 
   assert diml == 1
   assert dimr > 2 # assuming C always exists
   i1 = tmp[0,0].copy()
   c1 = tmp[0,1:dimr-1].copy()
   d1 = tmp[0,dimr-1].copy()
   c1 = xfac*c1
   d1 = xfac*d1
   tmp2 = numpy.zeros_like(tmp)
   tmp2[0,0] = i1
   tmp2[0,1:dimr-1] = c1.copy()
   tmp2[0,dimr-1] = d1+i1
   expHx[0] = tmp2.copy()
   #
   # site-(N-1): [tD, B, I]^t
   #
   tmp = hmpo.sites[N-1].copy()
   diml,dimr,nc1,nc2=tmp.shape
   assert dimr == 1
   assert diml > 2
   d1 = tmp[0,0].copy()
   b1 = tmp[1:diml-1,0].copy()
   i1 = tmp[diml-1,0].copy()
   d1 = xfac*d1
   tmp2 = numpy.zeros_like(tmp)
   tmp2[0,0] = d1
   tmp2[1:diml-1,0] = b1
   tmp2[diml-1,0] = i1
   expHx[N-1] = tmp2.copy()
   #
   # site-(1,N-2):
   #
   #	[ I  tC   tD ]
   #    [ 0   A    B ]
   #    [ 0   0    I ]
   #
   for isite in range(1,N-1):
      tmp = hmpo.sites[isite]
      diml,dimr,nc1,nc2=tmp.shape
      i1 = tmp[0,0].copy()
      c1 = tmp[0,1:dimr-1].copy()
      d1 = tmp[0,dimr-1].copy()
      a1 = tmp[1:diml-1,1:dimr-1].copy()
      b1 = tmp[1:diml-1,dimr-1].copy()
      i2 = tmp[diml-1,dimr-1].copy()
      c1 = xfac*c1
      d1 = xfac*d1
      assert numpy.linalg.norm(i1-i2)<1.e-10
      tmp2 = numpy.zeros_like(tmp)
      tmp2[0,0] = i1
      tmp2[0,1:dimr-1] = c1
      tmp2[0,dimr-1] = d1
      tmp2[1:diml-1,1:dimr-1] = a1
      tmp2[1:diml-1,dimr-1] = b1
      tmp2[diml-1,dimr-1] = i2
      expHx[isite]= tmp2.copy()
   empo = mpo_class.class_mpo(N,expHx,qnums)
   return empo

#==================
# Test subroutines
#==================
def genRandomH(k):
   numpy.random.seed(1)
   h1e = numpy.random.uniform(-1,1,size=(k,k))
   h2e = numpy.random.uniform(-1,1,size=(k,k,k,k))
   h1e = 0.5*(h1e+h1e.T)
   h2e = 0.5*(h2e+h2e.transpose((2,3,0,1)))
   h2e = antisymmetrizeTwoBodyOpers(h2e)
   return h1e,h2e

def testTwoSiteW():
   print '**************'
   print ' testTwoSiteW '
   print '**************'
   k = 2
   h1e,h2e = genRandomH(k) 
   partition = [[i] for i in range(k)]
   hmpo1 = directHmpo(h1e,h2e,partition)
   print 'sites0=\n',hmpo1.sites[0]
   print 'sites1=\n',hmpo1.sites[1]
   hmpo1.prt()
   mat1 = hmpo1.toMat()
   print 'mat1=\n',mat1
   hmpo2 = directHmpo(h1e,h2e,[range(k)])
   hmpo2.prt()
   mat2 = hmpo2.toMat()
   print 'mat2=\n',mat2
   diff = numpy.linalg.norm(mat1-mat2)
   print 'diff=',diff
   hmpo0 = mpo_class.genHmpo(h1e,h2e)
   hmpo0.prt()
   mat0 = hmpo0.toMat()
   print 'mat0=\n',mat2
   diff = numpy.linalg.norm(mat0-mat1)
   print 'diff1=',diff
   diff = numpy.linalg.norm(mat0-mat2)
   print 'diff2=',diff

   # Direct formation
   sl1 = []
   sc1 = [0]
   sr1 = [1]
   sl2 = [0]
   sc2 = [1]
   sr2 = []

   m1l=mpo_consw.l1r1(h1e,h2e, sl1,sc1,sr1)
   m2l=mpo_consw.l1r2(h1e,h2e, sl1,sc1,sr1) 
   m3l=mpo_consw.l1r3(h1e,h2e, sl1,sc1,sr1)
   #m4l=mpo_consw.l1r4(h1e,h2e, sl1,sc1,sr1)
   #m5l=mpo_consw.l1r5(h1e,h2e, sl1,sc1,sr1)
   m7l=mpo_consw.l1r7(h1e,h2e, sl1,sc1,sr1)
   m11l=mpo_consw.l1r11(h1e,h2e,sl1,sc1,sr1)
   m13l=mpo_consw.l1r13(h1e,h2e,sl1,sc1,sr1)
   m15l=mpo_consw.l1r15(h1e,h2e,sl1,sc1,sr1)
   m16l=mpo_consw.l1r16(h1e,h2e,sl1,sc1,sr1)
   
   m1r=mpo_consw.l1r16(h1e,h2e, sl2,sc2,sr2)
   m2r=mpo_consw.l2r16(h1e,h2e, sl2,sc2,sr2)
   m4r=mpo_consw.l4r16(h1e,h2e, sl2,sc2,sr2)
   #m6r=mpo_consw.l6r16(h1e,h2e, sl2,sc2,sr2)
   #m9r=mpo_consw.l9r16(h1e,h2e, sl2,sc2,sr2)
   m12r=mpo_consw.l12r16(h1e,h2e,sl2,sc2,sr2)
   m13r=mpo_consw.l13r16(h1e,h2e,sl2,sc2,sr2)
   m14r=mpo_consw.l14r16(h1e,h2e,sl2,sc2,sr2)
   m15r=mpo_consw.l15r16(h1e,h2e,sl2,sc2,sr2)
   m16r=mpo_consw.l16r16(h1e,h2e,sl2,sc2,sr2)

   t = [0]*8
   t[0] = numpy.kron(m1l,m1r)	# I[0]*H[1]
   t[1] = numpy.kron(m16l,m16r) # H[0]*I[1]
   t[2] = numpy.kron(m2l.reshape(2,2),m2r.reshape(2,2))   # T2*a1+
   t[3] = numpy.kron(m3l.reshape(2,2),m4r.reshape(2,2))   # (S+T4)*a1
   t[4] = numpy.kron(m7l.reshape(2,2),m12r.reshape(2,2))  # (-a)*S
   t[5] = numpy.kron(m11l.reshape(2,2),m13r.reshape(2,2)) # (-a^+a)*Q 
   t[6] = numpy.kron(m13l.reshape(2,2),m14r.reshape(2,2)) # a+*T1
   t[7] = numpy.kron(m15l.reshape(2,2),m15r.reshape(2,2)) # a*T3
   for i in range(8):
      print 't[i]',i,'\n',t[i]

   mat = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7]
   print 'Hmat=\n',mat
   print '... testTwoSiteW finished ...'
   return 0

def testMPOpartition():
   print '******************'
   print ' testMPOpartition '
   print '******************'
   k = 7
   h1e,h2e = genRandomH(k)
   # MPO-1
   partition = [[0,1,2],[3,4],[5,6]]
   hmpo1 = directHmpo(h1e,h2e,partition)
   hmpo1.prt()
   mat1 = hmpo1.toMat()
   print 'mat1=\n',mat1
   # MPO-2
   hmpo2 = directHmpo(h1e,h2e,[range(k)])
   hmpo2.prt()
   mat2 = hmpo2.toMat()
   print 'mat2=\n',mat2
   # Check
   diff = numpy.linalg.norm(mat1-mat2)
   print 'diff=',diff
   #----------------------------------------------------
   # The HS-norm of H should be invariant to partitions
   print hmpo1.HSnorm()
   print numpy.linalg.norm(hmpo2.sites[0])
   #----------------------------------------------------
   # MPO-0
   hmpo0 = mpo_class.genHmpo(h1e,h2e)
   hmpo0.prt()
   mat0 = hmpo0.toMat()
   print 'mat0=\n',mat2
   diff = numpy.linalg.norm(mat0-mat1)
   print 'diff1=',diff
   diff = numpy.linalg.norm(mat0-mat2)
   print 'diff2=',diff
   print '... tMPOpartition finished ...'
   return 0

def testMPOkSites(k):
   # 
   # MPOinfo:
   #  nsite =  60
   #  Site :  0  Shape :  (1, 3546, 2, 2)  Val =  3.97810678972
   #  Site :  1  Shape :  (3546, 3434, 2, 2)  Val =  90.5613461065
   #  Site :  2  Shape :  (3434, 3326, 2, 2)  Val =  95.6971003523
   #  Site :  3  Shape :  (3326, 3222, 2, 2)  Val =  100.459752155
   #  Site :  4  Shape :  (3222, 3122, 2, 2)  Val =  104.917938675
   #  Site :  5  Shape :  (3122, 3026, 2, 2)  Val =  109.660792892
   #  Site :  6  Shape :  (3026, 2934, 2, 2)  Val =  114.180368841
   #  Site :  7  Shape :  (2934, 2846, 2, 2)  Val =  118.142568447
   #  Site :  8  Shape :  (2846, 2762, 2, 2)  Val =  121.925820424
   #  Site :  9  Shape :  (2762, 2682, 2, 2)  Val =  126.26124108
   #  Site :  10  Shape :  (2682, 2606, 2, 2)  Val =  129.450989663
   #  Site :  11  Shape :  (2606, 2534, 2, 2)  Val =  133.38308444
   #  Site :  12  Shape :  (2534, 2466, 2, 2)  Val =  135.80955642
   #  Site :  13  Shape :  (2466, 2402, 2, 2)  Val =  139.864257545
   #  Site :  14  Shape :  (2402, 2342, 2, 2)  Val =  142.479165437
   #  Site :  15  Shape :  (2342, 2286, 2, 2)  Val =  145.571964279
   #  Site :  16  Shape :  (2286, 2234, 2, 2)  Val =  148.620492882
   #  Site :  17  Shape :  (2234, 2186, 2, 2)  Val =  151.1898681
   #  Site :  18  Shape :  (2186, 2142, 2, 2)  Val =  153.611643717
   #  Site :  19  Shape :  (2142, 2102, 2, 2)  Val =  155.704351935
   #  Site :  20  Shape :  (2102, 2066, 2, 2)  Val =  158.683299839
   #  Site :  21  Shape :  (2066, 2034, 2, 2)  Val =  160.727470807
   #  Site :  22  Shape :  (2034, 2006, 2, 2)  Val =  163.211999983
   #  Site :  23  Shape :  (2006, 1982, 2, 2)  Val =  163.91389449
   #  Site :  24  Shape :  (1982, 1962, 2, 2)  Val =  166.195839436
   #  Site :  25  Shape :  (1962, 1946, 2, 2)  Val =  168.055595352
   #  Site :  26  Shape :  (1946, 1934, 2, 2)  Val =  169.543384443
   #  Site :  27  Shape :  (1934, 1926, 2, 2)  Val =  170.510247763
   #  Site :  28  Shape :  (1926, 1922, 2, 2)  Val =  172.360605378
   #  Site :  29  Shape :  (1922, 1922, 2, 2)  Val =  173.062154854
   #  Site :  30  Shape :  (1922, 1926, 2, 2)  Val =  174.605897886
   #  Site :  31  Shape :  (1926, 1934, 2, 2)  Val =  174.141287234
   #  Site :  32  Shape :  (1934, 1946, 2, 2)  Val =  175.502451569
   #  Site :  33  Shape :  (1946, 1962, 2, 2)  Val =  176.133362118
   #  Site :  34  Shape :  (1962, 1982, 2, 2)  Val =  176.182010052
   #  Site :  35  Shape :  (1982, 2006, 2, 2)  Val =  176.216098444
   #  Site :  36  Shape :  (2006, 2034, 2, 2)  Val =  176.105327039
   #  Site :  37  Shape :  (2034, 2066, 2, 2)  Val =  175.064230665
   #  Site :  38  Shape :  (2066, 2102, 2, 2)  Val =  174.864950924
   #  Site :  39  Shape :  (2102, 2142, 2, 2)  Val =  174.67239765
   #  Site :  40  Shape :  (2142, 2186, 2, 2)  Val =  173.96943905
   #  Site :  41  Shape :  (2186, 2234, 2, 2)  Val =  173.287507894
   #  Site :  42  Shape :  (2234, 2286, 2, 2)  Val =  171.834718691
   #  Site :  43  Shape :  (2286, 2342, 2, 2)  Val =  169.99506417
   #  Site :  44  Shape :  (2342, 2402, 2, 2)  Val =  167.993420909
   #  Site :  45  Shape :  (2402, 2466, 2, 2)  Val =  166.524851081
   #  Site :  46  Shape :  (2466, 2534, 2, 2)  Val =  163.895012736
   #  Site :  47  Shape :  (2534, 2606, 2, 2)  Val =  160.856190923
   #  Site :  48  Shape :  (2606, 2682, 2, 2)  Val =  158.104318978
   #  Site :  49  Shape :  (2682, 2762, 2, 2)  Val =  155.602255854
   #  Site :  50  Shape :  (2762, 2846, 2, 2)  Val =  151.601852993
   #  Site :  51  Shape :  (2846, 2934, 2, 2)  Val =  147.523211848
   #  Site :  52  Shape :  (2934, 3026, 2, 2)  Val =  143.266208065
   #  Site :  53  Shape :  (3026, 3122, 2, 2)  Val =  138.222080043
   #  Site :  54  Shape :  (3122, 3222, 2, 2)  Val =  132.892349665
   #  Site :  55  Shape :  (3222, 3326, 2, 2)  Val =  126.323869804
   #  Site :  56  Shape :  (3326, 3434, 2, 2)  Val =  119.528242993
   #  Site :  57  Shape :  (3434, 3546, 2, 2)  Val =  111.727242372
   #  Site :  58  Shape :  (3546, 3662, 2, 2)  Val =  103.328526326
   #  Site :  59  Shape :  (3662, 1, 2, 2)  Val =  34.9337373147
   # End of MPOinfo
   # 
   print '***************'
   print ' testMPOkSites '
   print '***************'
   h1e,h2e = genRandomH(k)
   # MPO-1
   partition = [[i] for i in range(k)]
   hmpo1 = directHmpo(h1e,h2e,partition)
   hmpo1.prt()
   print '... testMPOkSites finished ...'
   return hmpo1

def Dg(k):
   print '\n[Dg] bond dimension'
   print ' mean (k-1)/2=',float(k-1)/2
   return [(i,((2*i-(k-1))**2+(k+1)*(k+3))/2) for i in range(1,k)]

def testMOL():
   print '*********'
   print ' testMOL '
   print '*********'
   from pyscf import gto,scf
   mol = gto.Mole()
   natoms = 4
   R = 1
   mol.atom = [['H',(0,0,R*i)] for i in range(natoms)]
   mol.spin = 0
   mol.basis='sto-3g'
   mol.verbose = 5
   mol.build()
   enuc = mol.energy_nuc()
  
   mf = scf.RHF(mol)
   ehf = mf.scf()
   mo_coeff = mf.mo_coeff
  
   partition = [[i] for i in range(2*mf.mo_coeff.shape[0])]
   hmpo1 = genHmpo(1,mol,mo_coeff,partition)
   hmpo1.prt()
   hmpo0 = genHmpo(0,mol,mo_coeff)
   hmpo0.prt()
  
   # HS-norm test   
   diff = mpo_class.mpo_diff(hmpo1,hmpo0)
   print 'diff=',diff
   mat1 = hmpo1.toMat()
   mat0 = hmpo0.toMat()
   e1,v = scipy.linalg.eigh(mat1)
   e0,v = scipy.linalg.eigh(mat0)
   print 'mat1=\n',mat1
   print 'mat0=\n',mat0
   print 'eigs1=',e1
   print 'eigs0=',e0

   # HF energy test
   import mps_class
   k = 2*mf.mo_coeff.shape[0]
   n = mol.nelectron
   mps0 = mps_class.class_mps(k)
   mps0.hfstate(n)
   mps0.prt()
   enuc = mol.energy_nuc()
   ehf0 = mps0.dot(hmpo0.dotMPS(mps0))
   print 'etot0[HF]=',ehf0+enuc
   ehf1 = mps0.dot(hmpo1.dotMPS(mps0))
   print 'etot1[HF]=',ehf1+enuc
   print '... testMOL finished ...'
   return 0

def testH6():
   print '********'
   print ' testH6 '
   print '********'
   from pyscf import gto,scf
   mol = gto.Mole()
   natoms = 6
   R = 1
   mol.atom = [['H',(0,0,R*i)] for i in range(natoms)]
   mol.spin = 0
   mol.basis='sto-3g'
   mol.verbose = 5
   mol.build()
   enuc = mol.energy_nuc()
   mf = scf.RHF(mol)
   ehf = mf.scf()
   mo_coeff = mf.mo_coeff
   partition = [[i] for i in range(2*mf.mo_coeff.shape[0])]
   hmpo1 = genHmpo(1,mol,mo_coeff,partition)
   hmpo1.prt()
   # HF energy test
   import mps_class
   k = 2*mf.mo_coeff.shape[0]
   n = mol.nelectron
   mps0 = mps_class.class_mps(k)
   mps0.hfstate(n)
   mps0.prt()
   enuc = mol.energy_nuc()
   ehf1 = mps0.dot(hmpo1.dotMPS(mps0))
   print 'etot0[HF]=',ehf1
   print 'etot1[HF]=',ehf1+enuc
   print 'etot1[PYSCF]=',ehf
   ediff = ehf1+enuc-ehf
   print 'etot1(diff) =',ediff
   assert abs(ediff)<1.e-8
   print '... testH6 finished ...'
   return 0

def testAll():
   testTwoSiteW()
   testMPOpartition()
   testMPOkSites(20)
   testMOL()
   testH6()
   return 0

if __name__ == '__main__':
  
   testAll()
