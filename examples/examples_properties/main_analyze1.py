#
# Analysis for integral independent properties 
#
import math
import numpy
import h5py
from mpodmrg.source import mpo_dmrg_class
from mpodmrg.source.properties import mpo_dmrg_propsItrf

prefix = 'lmps_data_for_h3_631g/'

for ifQt in [False,True]:
   for ifs2proj in [False,True]:

      fnames = ['lmps_NQt_NProj','lmps_NQt_Proj','lmps_Qt_NProj','lmps_Qt_Proj']
      if not ifQt:
         if not ifs2proj:
            fname = prefix + fnames[0] # Correct !
         else:
            fname = prefix + fnames[1] # Correct !
      else:
         if not ifs2proj:
            fname = prefix + fnames[2] # Correct !
         else:
            fname = prefix + fnames[3] # Correct !
      print
      print '#'*80
      print 'fname=',fname
      print '#'*80
      flmps = h5py.File(fname,'r')
      nsite = flmps['nsite'].value
      
      # Fake object to passing parameters
      dmrg = mpo_dmrg_class.mpo_dmrg()
      dmrg.ifQt = ifQt
      dmrg.ifs2proj = ifs2proj
      dmrg.nsite = nsite
      
      # Analysis 
      if not ifs2proj:
         info=None
      else:
         info=[5,0.5,0.5]
         mpo_dmrg_propsItrf.eval_P(dmrg,flmps,info,True)
      groups = [[0,1],[2,3],[4,5]]
      print
      
      # <S2>	 
      expect = mpo_dmrg_propsItrf.eval_S2Global(dmrg,flmps,spinfo=info)
      print 'expect_S2=',expect
      
      # <Omega>
      expect = mpo_dmrg_propsItrf.eval_Global(dmrg,flmps,'Omega',spinfo=info)
      print 'expect_Omega=',expect
      
      # <NA> 
      expect = mpo_dmrg_propsItrf.eval_Local(dmrg,flmps,groups,'N',spinfo=info)
      print 'expect_N=\n',expect
      print 'nelec=',numpy.sum(expect)
       
      # <NA*NB>
      expect = mpo_dmrg_propsItrf.eval_Local2(dmrg,flmps,groups,'N','N',spinfo=info)
      print 'expect_NN=\n',expect
      print 'nelec2=',numpy.sum(expect)
      
      # <Si*Sj>
      expect = mpo_dmrg_propsItrf.eval_SiSj(dmrg,flmps,groups,spinfo=info)
      print 'expect_SS=\n',expect
      s2exp = numpy.sum(expect)
      print 's2exp=',s2exp
      print 'seff=',mpo_dmrg_propsItrf.seff(s2exp)
      
      # RDM1 = Gamma[p,q]
      rdm1t,rdm1s = mpo_dmrg_propsItrf.eval_rdm1BF(dmrg,flmps,spinfo=info)
      print 'rdm1t=\n',rdm1t
      print 'rdm1s=\n',rdm1s
         
      # Final
      flmps.close()
