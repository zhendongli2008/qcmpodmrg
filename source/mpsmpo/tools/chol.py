##############################
#   Chokesky Decomposition   #
##############################
import h5py
import math
import numpy
import scipy
from pyscf import gto

def cdDiag(mol,intor,shells=None,ifsym=True):
    atm = numpy.array(mol._atm, dtype=numpy.int32)
    bas = numpy.array(mol._bas, dtype=numpy.int32)
    env = numpy.array(mol._env)
    Qij = []
    ioff= 0
    # (i>=j)
    lst = []
    if shells is None: shells=range(mol.nbas)
    nbas = 0
    for i in shells:
       for j in shells:
	  if ifsym and j<i: continue     
          shls = (i,j,j,i)
          buf = gto.moleintor.getints_by_shell(intor, shls, atm, bas, env)
	  buf = buf.transpose(0,1,3,2)
          di, dj, dk, dl = buf.shape
          buf = buf.reshape((di*dj,di*dj))
          diag = numpy.diag(buf)
          dmax = numpy.amax(diag)
	  lst.append(dmax)
          Qij.append([(i,j),ioff,dmax,diag,di*dj])
          ioff = ioff + di*dj
       nbas += di
    #print numpy.max(lst)
    #print "MAXIMUM=",sorted(lst,reverse=True)[0]
    return nbas,Qij

def cdMain(mol,intor,thresh=1.e-8,shells=None,ifsym=True,fname='cdvec.h5'):
    # Control
    debug= False
    iop  = 1 # Sort
    # Open file for storing CD vector
    f = h5py.File(fname, "w")
    blks = f.create_dataset("nblock",(1,),dtype='int64',data=[0])
    # Begin
    atm = numpy.array(mol._atm, dtype=numpy.int32)
    bas = numpy.array(mol._bas, dtype=numpy.int32)
    env = numpy.array(mol._env)
    nbas,Qij = cdDiag(mol,intor,shells,ifsym)
    if iop !=0: Qij = sorted(Qij,key=lambda x:x[2])
    Dij = sorted(Qij,key=lambda x:x[4],reverse=True)[0][-1]
    #
    # MEM control
    #
    nao = mol.nao_nr()
    mem = (nao*nao)*Dij*numpy.zeros(1).itemsize/1024.0**2 #"M"
    mem = max(mem,100.0)
    # Print info 
    if shells is None: shells=range(mol.nbas)
    nshl = len(shells)
    if ifsym:
       npair = nshl*(nshl+1)/2	   
    else:
       npair = nshl*nshl
    if debug:
       print "\nINFORMATION for Cholesky Decomposition:"
       print "nshl =",nshl
       print "npair=",npair
       for ij in range(npair):
           print "ij-shell pair=",ij
           print Qij[ij]
       print "\nBegin Cholesky Decomposition:"
    # 
    # LOOP over shells |IJ)
    #
    nblk   = 0
    cdbuf  = []
    cdbas  = []
    for ij in range(npair):
	if iop == 0:
	   indx = ij # Without pivoting  
	else:
	   indx = -1 # With pivoting
        ijshl= Qij[indx][0]
	ishl,jshl=ijshl
 	#
	# 1. COMPUTE [kl|ji] - Nbas*Nbas*const
	#
	dmax = Qij[indx][2]
	if debug: print '\n>>> ij=',ij,'/',npair,'i=',ishl,'j=',jshl,' dmax=',dmax
	if dmax < thresh:
	   if debug: print "    Skipped based on Dmax:",ij,ijshl,dmax
	   continue
  	   # Actually, it is safe to break here when Qij is ordered
	   #break
	buff = []
	for kshl in shells:
	   for lshl in shells:
	      if ifsym and lshl<kshl: continue     
              shls = (kshl,lshl,jshl,ishl)
              buf2 = gto.moleintor.getints_by_shell(intor, shls, atm, bas, env)
	      buf2 = buf2.transpose(0,1,3,2)
	      dk,dl,di,dj = buf2.shape
	      buf2 = buf2.reshape(dk*dl,di*dj)
	      if buff == []:
 	         buff = buf2
	      else:
	         buff = numpy.vstack((buff,buf2))         
	#
 	# 2. Construct decomposition
	#
	lenij = len(Qij[indx][3])
	diag  = [[i,Qij[indx][3][i]] for i in range(lenij)]
	if iop !=0: diag = sorted(diag,key=lambda x:x[1])
	cdvec = []
	noff  = Qij[indx][1]
	#
	# FOR THIS column - (*|IJ)
	#
	for i in range(lenij):
	    if iop == 0:
	       idx = i 
	    else:
	       idx = -1
	    # Check diagonal of L
	    vdiag = diag[idx][1]
	    if abs(vdiag.imag) > 1.e-10:
	       print "ERROR: Imaginary part is larger than 1.e-10 !"
	       exit(1)
	    if vdiag.real < thresh:
	       if debug: print "    Skipped based on vdiag=",i,diag
	       break
	    lij  = math.sqrt(vdiag.real)
      	    #======================================== 
	    # Construct CD vector via subtraction
      	    #======================================== 
 	    j = diag[idx][0]
	    cdbas.append([ishl,jshl,j,di,dj,vdiag])
	    addr = noff + j
	    vec  = buff[:,j]
	    for j in range(len(cdvec)):
	       vec=vec-cdvec[j]*numpy.conj(cdvec[j][addr])
	    # Buff
	    for j in range(len(cdbuf)):
	       vec=vec-cdbuf[j]*numpy.conj(cdbuf[j][addr])
	    # Load previous CD vec from file
	    if nblk > 0:
	       for iblk in range(nblk):
 	          sblk = "block"+str(iblk)
	          nvec = f[sblk].shape[0]
	          for j in range(nvec):
	             vec=vec-f[sblk][j]*numpy.conj(f[sblk][j][addr])
	    # Update
	    vec = vec/lij
	    cdvec.append(vec)
      	    #======================================== 
	    # Update diagonal
	    if iop != 0: diag.pop()
	    qlen = len(diag)
	    for q in range(qlen):
	       # Orbital index	   
	       j = diag[q][0]
	       diag[q][1]=diag[q][1]-vec[noff+j]*numpy.conj(vec[noff+j])
	    # Re-ordering
	    if iop != 0: diag=sorted(diag,key=lambda x:x[1])
	#
	# Update all Qij and resort
	#
	if iop !=0: Qij.pop()
	qlen = len(Qij)
	for ijp in range(qlen):
	   qoff =Qij[ijp][1]
	   qdiag=Qij[ijp][3].copy()
	   for j in range(len(qdiag)):
	      qaddr=qoff+j
	      for i in range(len(cdvec)):
	         qdiag[j]=qdiag[j]-cdvec[i][qaddr]*numpy.conj(cdvec[i][qaddr])
		 if qdiag[j]<-0.1: 
		    print "ERROR diag < 0.0",ijp,j,qdiag[j]
		    exit(1)     
	   Qij[ijp][2]=numpy.amax(qdiag)
	   Qij[ijp][3]=qdiag.copy()
	# Re-ordering
	if iop != 0: Qij=sorted(Qij,key=lambda x:x[2])
	# 
	# BUFF += DUMP (**|IJ)
  	# 
	cdbuf = cdbuf + cdvec
	size  = len(cdbuf)*cdbuf[0].nbytes/1024.0**2
	if debug: 
           print "    di*dj=",di*dj," lencdvec=",len(cdvec)," lenbuf=",len(cdbuf),\
	   	 " MEM(M)=",size
	if size > mem:
 	   sblk = "block"+str(nblk)
	   nblk = nblk + 1
	   s=f.create_dataset(sblk, data=cdbuf) #dtype='float64')
	   cdbuf = []
    #
    # END: Dump remaining CD vector
    #
    if len(cdbuf) != 0: 
       sblk = "block"+str(nblk)
       nblk = nblk + 1
       s=f.create_dataset(sblk, data=cdbuf)
       cdbuf = []
    blks[0] = nblk
    # CHECK
    print "\nCholesky Decomposition:"
    print 'nblk =',nblk
    print "NBAS =",nbas
    print "NPAIR=",nbas*(nbas+1)/2 # NOT EXACTLY IN SYM-CASE
    for name in f:
	print name,'shape =',f[name].shape
    # CLOSE
    f.close()
    return nbas,cdbas


def cdCheck(mol,intor,shells=None,ifsym=True,fname='cdvec.h5'):
    if shells is None: shells=range(mol.nbas)
    atm = numpy.array(mol._atm, dtype=numpy.int32)
    bas = numpy.array(mol._bas, dtype=numpy.int32)
    env = numpy.array(mol._env)
    for idx,i in enumerate(shells):
       for jdx,j in enumerate(shells):
          if ifsym and j<i: continue
	  # The full column
	  buff = None
	  for k in shells:
              pl = 0
	      for l in shells:
	          if ifsym and l<k: continue     
                  # A[i,j,k,l] = [kl|ij]
	          shls = (k,l,j,i)
                  buf = gto.moleintor.getints_by_shell(intor, shls, atm, bas, env)
	          buf = buf.transpose(0,1,3,2)
	          dk, dl, di, dj = buf.shape
	          buf = buf.reshape(dk*dl,di*dj)
		  if buff is None: 
 	             buff = buf
	          else:
	             buff = numpy.vstack((buff,buf))
	  # Store into full matrix
	  if idx==0 and jdx==0:
 	     mat = buff.copy()
	  else:
	     mat = numpy.hstack((mat,buff))
    print "\nCDcheck:"
    print "mat_sym  =",numpy.linalg.norm(mat-mat.T.conj())
    eri=mat

    #
    # Compared with eris
    #
    nb = mol.nao_nr()
    eri3 = numpy.zeros((nb,nb,nb,nb))
    fill(mol,eri3,'cint2e_sph')
    eri3 = eri3.reshape((nb**2,nb**2))
    print 'diff0[before reorgblk2d] =',numpy.linalg.norm(eri3-eri)
    # reorganize 2D
    indx = reorgblk2d(mol)
    eri4 = mat[numpy.ix_(indx,indx)]
    print 'diff1[after reorgblk2d] = ',numpy.linalg.norm(eri3-eri4)
    #
    # Load
    #
    f = h5py.File(fname, "r")
    nblk = f['nblock'][0]
    eri2 = numpy.zeros(eri.shape)
    lvec = []
    for iblk in range(nblk):
 	sblk = "block"+str(iblk)
	nvec = f[sblk].shape[0]
	print sblk,'shape=',f[sblk].shape 
	for j in range(nvec):
	   l=f[sblk][j]
	   lvec.append(l)
    f.close()
    lvec=numpy.array(lvec)
    # L[mu,vec]*L[mu,vec]^* in such case
    eri2=numpy.dot(lvec.T,lvec.conj())
    print "Norm(Mat)=",numpy.linalg.norm(eri)
    print "Norm(LtL)=",numpy.linalg.norm(eri2)
    eri2 = eri2 - eri
    print "ERI_DIFF(cd-exact)=",numpy.linalg.norm(eri2)
    #
    # Further check
    #
    eri  = numpy.dot(lvec[:,indx].T,lvec[:,indx].conj())
    eri2 = mol.intor('cint2e_sph',aosym='s1').reshape(nb*nb,nb*nb)
    print 'diff[reorgblk2d]=',numpy.linalg.norm(eri-eri2)
    return 0


def fill(mol, eri, intor='cint2e_sph'):
    atm = numpy.array(mol._atm, dtype=numpy.int32)
    bas = numpy.array(mol._bas, dtype=numpy.int32)
    env = numpy.array(mol._env)
    pi = 0
    for i in range(mol.nbas):
        pj = 0
        for j in range(mol.nbas):
            pk = 0
            for k in range(mol.nbas):
                pl = 0
                for l in range(mol.nbas):
                    shls = (i,j,k,l)
                    buf = gto.moleintor.getints_by_shell(intor, shls, atm, bas, env)
                    di, dj, dk, dl = buf.shape
                    eri[pi:pi+di,pj:pj+dj,pk:pk+dk,pl:pl+dl] = buf
                    pl += dl
                pk += dk
            pj += dj
        pi += di
    return eri

#
# Map blocked indices to contious one:
# [1,1] [1,2] [2,1], [2,2] | [1,3] | ... to [1,:] [2,:]
#
def reorgblk2d(mol):
   pk = 0
   indx = []
   idx = 0
   for k in range(mol.nbas):
      kang  = mol.bas_angular(k)
      kcntr = mol.bas_nctr(k)
      kbas  = kcntr*(2*kang+1)
      kindx = numpy.arange(pk,pk+kbas)
      pl = 0
      for l in range(mol.nbas):
         lang  = mol.bas_angular(l)
         lcntr = mol.bas_nctr(l)
	 lbas  = lcntr*(2*lang+1)
	 lindx = numpy.arange(pl,pl+lbas)
         for kb in kindx:
	    for lb in lindx:
	       indx.append([kb,lb,idx])
	       idx +=1
         pl += lbas    	
      pk += kbas
   nindx = map(lambda x:x[2],sorted(indx))
   return numpy.array(nindx)
