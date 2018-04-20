import numpy
import h5py

def dump(info,ordering=None,fname='mole.h5'):
   int1e,int2e = info
   # dump information
   nbas = int1e.shape[0]/2
   sbas = nbas*2
   print '\n[tools_itrf.dump] interface from FCIDUMP with nbas=',nbas
   f = h5py.File(fname, "w")
   cal = f.create_dataset("cal",(1,),dtype='i')
   cal.attrs["nelec"] = 0.
   cal.attrs["sbas"]  = sbas
   cal.attrs["enuc"]  = 0.
   cal.attrs["ecor"]  = 0.
   cal.attrs["escf"]  = 0. # Not useful at all
   # Intergrals
   flter = 'lzf'
   # INT1e:
   h1e = int1e.copy()
   # INT2e:
   h2e = int2e.copy()
   # <ij|kl> = [ik|jl]
   h2e = h2e.transpose(0,2,1,3)
   # Antisymmetrize V[pqrs]=-1/2*<pq||rs> - In MPO construnction, only r<s part is used. 
   h2e = -0.5*(h2e-h2e.transpose(0,1,3,2))
   int1e = f.create_dataset("int1e", data=h1e, compression=flter)
   int2e = f.create_dataset("int2e", data=h2e, compression=flter)
   # Occupation
   occun = numpy.zeros(sbas)
   orbsym = numpy.array([0]*sbas)
   spinsym = numpy.array([[0,1] for i in range(nbas)]).flatten()
   f.create_dataset("occun",data=occun)
   f.create_dataset("orbsym",data=orbsym)
   f.create_dataset("spinsym",data=spinsym)
   f.close()
   print ' Successfully dump information for MPO-DMRG calculations! fname=',fname
   return 0

if __name__ == '__main__':
   h1e = numpy.load('h1body.dat.npy')
   h2e = numpy.load('h2body.dat.npy')
   print h1e
   print h2e
   info = [h1e,h2e]
   dump(info)
