import h5py
import copy

# Dump MPS into file: mps-rk3,qnum-(n+1)
def dumpMPS(mps,qnum=None,fname='mps.h5'):
   print '\n[mps_io.dumpMPS] fname = ',fname
   f = h5py.File(fname,'w')
   nsite = len(mps)
   f.create_dataset('nsite',data=nsite)
   # Site
   for isite in range(nsite):
      f.create_dataset('mps_'+str(isite),data=mps[isite])
   # Qnum
   f.create_dataset('qnum',(1,),dtype='i')
   f['qnum'].attrs['status'] = 0
   if qnum is not None:
      f['qnum'].attrs['status'] = 1
      for isite in range(nsite+1):
         f.create_dataset('qnum_'+str(isite),data=qnum[isite])
   f.close()
   return 0 

def loadMPS(fname='mps.h5'):
   print '\n[mps_io.loadMPS] fname = ',fname
   f = h5py.File(fname,'r')
   nsite = f['nsite'].value
   mps = [0]*nsite
   # Site
   for isite in range(nsite):
      mps[isite] = f['mps_'+str(isite)].value
   # Qnum
   iqnum = f['qnum'].attrs['status']
   qnum = None
   if iqnum != 0:
      qnum = [None]*(nsite+1)
      for isite in range(nsite+1):
          qnum[isite] = f['qnum_'+str(isite)].value
   f.close()
   return mps,qnum

# Interface by adding the head
def qnumsItrf(nbond,qnums):
   # site :    0   1   2  
   # qnum : -0-*-1-*-2-*-3-
   nq = len(qnums) 
   assert nbond == nq+1
   qnuml = [None]*nbond
   final = qnums[-1]
   assert len(final) == 1
   if len(final[0]) == 1:
      qnuml[0] = [[0.]]
   elif len(final[0]) == 2:
      qnuml[0] = [[0.,0.]]
   else:
      print 'error: no such situation! finalSym =',final[0]
      exit(1)
   for idx in range(nq):
      qnuml[idx+1] = copy.deepcopy(qnums[idx])
   return qnuml
