import numpy 
import matplotlib.pyplot as plt
   
nsite = 8
ttotal = 3.0
nt = 30
tau = ttotal/nt
tarray = numpy.arange(nt+1)*tau

def loadData(prefix,isite):
   nm = numpy.load(prefix+'/nm'+str(isite)+'.npy')
   gfdiag = numpy.load(prefix+'/gf'+str(isite)+'.npy')
   gfr = map(lambda x:x.real,gfdiag)
   gfi = map(lambda x:x.imag,gfdiag)
   return nm,gfr,gfi

nm1,gfr1,gfi1 = loadData('./dataOrder1',0)
nm2,gfr2,gfi2 = loadData('./dataOrder2',0)
nm4,gfr4,gfi4 = loadData('./dataOrder4',0)
nm4old,gfr4old,gfi4old = loadData('./dataOrder4old',0)
nm4sep,gfr4sep,gfi4sep = loadData('./dataOrder4sep',0)

plt.plot(tarray,nm1,'o-',label='order1')
plt.plot(tarray,nm2,'o-',label='order2')
plt.plot(tarray,nm4,'o-',label='order4')
plt.plot(tarray,nm4old,'o-',label='order4old')
plt.plot(tarray,nm4sep,'o-',label='order4sep')
plt.legend()
plt.show()

#plt.plot(tarray,gfr1,'o-',label='order1')
#plt.plot(tarray,gfr2,'o-',label='order2')
#plt.plot(tarray,gfr4,'o-',label='order4')
#plt.legend()
#plt.show()
#
#plt.plot(tarray,gfi1,'o-',label='order1')
#plt.plot(tarray,gfi2,'o-',label='order2')
#plt.plot(tarray,gfi4,'o-',label='order4')
#plt.legend()
#plt.show()

#
# Reference data from Enrico's code
#
prefix = './dataSite0/'
tarray_ref = numpy.loadtxt(prefix+"rt_real.txt").T[0] #[:nt+1] 
gfr_ref = numpy.loadtxt(prefix+"rt_real.txt").T[1] #[:nt+1]
gfi_ref = numpy.loadtxt(prefix+"rt_imag.txt").T[1] #[:nt+1]

plt.plot(tarray_ref,gfr_ref,'ko-',label='Block')
plt.plot(tarray,gfr1,'o-',label='order1')
plt.plot(tarray,gfr2,'o-',label='order2')
plt.plot(tarray,gfr4,'o-',label='order4')
plt.plot(tarray,gfr4old,'o-',label='order4old')
plt.plot(tarray,gfr4sep,'o-',label='order4sep')
plt.legend()
plt.show()

plt.plot(tarray_ref,gfi_ref,'ko-',label='Block')
plt.plot(tarray,gfi1,'o-',label='order1')
plt.plot(tarray,gfi2,'o-',label='order2')
plt.plot(tarray,gfi4,'o-',label='order4')
plt.plot(tarray,gfi4old,'o-',label='order4old')
plt.plot(tarray,gfi4sep,'o-',label='order4sep')
plt.legend()
plt.show()
