#!/usr/bin/python                                                                                                                                             \
#
# Author: Enrico Ronca <enrico.r8729@gmail.com>
#


import os
import numpy as np
from scipy import fft, arange

def run(t,gfr,gfi,eta=0.3, rem_add='rem'):

    time_array = t
    real_part = np.asarray(gfr)
    imag_part = np.asarray(gfi)

    npoints = time_array.shape[0]
    delta_t = time_array[1]

    frq = np.fft.fftfreq(npoints, delta_t)
    frq = np.fft.fftshift(frq)*2.0*np.pi

    if (rem_add == 'rem'):
       fftinp = 1j*(real_part + 1j*imag_part)
    elif (rem_add == 'add'):
       fftinp = 1j*(real_part - 1j*imag_part)
    else:
       print 'Addition or Removal has not been specified!'
       return

    for i in range(npoints):
        fftinp[i] = fftinp[i]*np.exp(-eta*time_array[i])

    Y = fft(fftinp)
    Y = np.fft.fftshift(Y)

    Y_real = Y.real
    Y_real = (Y_real*time_array[-1]/npoints)
    Y_imag = Y.imag
    Y_imag = (Y_imag*time_array[-1]/npoints)/np.pi

    # Plot the results
    with open('ldos.out', 'w') as fout:
        fout.write('#     Omega          A(Omega)\n')
        for i in range(npoints):
            fout.write('%12.8f  %12.8f\n' % (frq[i], Y_imag[i]))

    with open('real_part.txt', 'w') as fout:
        fout.write('#     Omega          A(Omega)\n')
        for i in range(npoints):
            fout.write('%12.8f  %12.8f\n' % (frq[i], Y_real[i]))

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(frq,Y_real,'r-')
    plt.plot(frq,Y_imag,'b-')
    plt.show()
    plt.savefig("result.png")

if __name__=="__main__":
   
    nsite = 8
    ttotal = 3.0
    nt = 30
    tau = ttotal/nt

    tarray = np.arange(nt+1)*tau
    gfdiag = np.zeros(nt+1,dtype=np.complex128)
    gfsums = np.zeros(nt+1,dtype=np.complex128)
    slst = [0] #range(nsite)
    prefix = './dataOrder1/gf'
    for isite in slst:

       gfdiag = np.load(prefix+str(isite)+'.npy')
       gfsums += gfdiag

       gfr = map(lambda x:x.real,gfdiag)
       gfi = map(lambda x:x.imag,gfdiag)
       import matplotlib.pyplot as plt
       plt.plot(tarray,gfr,'ro-')
       plt.plot(tarray,gfi,'ro-')

    tarray4 = np.array(tarray)
    gfr4 = np.array(gfr)
    gfi4 = np.array(gfi)
    plt.show()

#     nsite = 8
#     ttotal = 30.0
#     nt = 600
#     tau = ttotal/nt
# 
#     tarray = np.arange(nt+1)*tau
#     gfdiag = np.zeros(nt+1,dtype=np.complex128)
#     gfsums = np.zeros(nt+1,dtype=np.complex128)
#     slst = [0] #range(nsite)
#  
#     prefix = './dataE2T30/gf'
#     for isite in slst:
# 
#        gfdiag = np.load(prefix+str(isite)+'.npy')
#        gfsums += gfdiag
# 
#        gfr = map(lambda x:x.real,gfdiag)
#        gfi = map(lambda x:x.imag,gfdiag)
#        import matplotlib.pyplot as plt
#        plt.plot(tarray,gfr,'go-')
#        plt.plot(tarray,gfi,'go-')
#  
# #    plt.show()
# #
# #    gfr = map(lambda x:x.real,gfsums)
# #    gfi = map(lambda x:x.imag,gfsums)
# #
# #    #
# #    # reference data
# #    #
# #    import matplotlib.pyplot as plt
# #
# #    prefix = './t500/'
# #    tarray1 = np.loadtxt(prefix+"refR.txt").T[0] #[:nt+1] 
# #    gfr1 = np.loadtxt(prefix+"refR.txt").T[1] #[:nt+1]
# #    gfi1 = np.loadtxt(prefix+"refI.txt").T[1] #[:nt+1]
# #    
# #    plt.plot(tarray,gfr,'ro-')
# #    plt.plot(tarray,gfi,'bo-')
# #    plt.show()
# #    
# #    plt.plot(tarray,gfr,'ro-')
# #    plt.plot(tarray1,gfr1,'g-')
# #    plt.show()
# #    
# #    plt.plot(tarray,gfi,'bo-')
# #    plt.plot(tarray1,gfi1,'k-')
# #    plt.show()
#     
# #    prefix = './t500/'
# #    tarray2 = np.loadtxt(prefix+"refRfci.txt").T[0] #[:nt+1] 
# #    gfr2 = np.loadtxt(prefix+"refRfci.txt").T[1]/2.0 #[:nt+1]
# #    gfi2 = np.loadtxt(prefix+"refIfci.txt").T[1]/2.0 #[:nt+1]
# #    plt.plot(tarray2,gfr2,'go-')
# #    plt.plot(tarray2,gfi2,'ko-')
# #    plt.show()
# 
#     prefix = './dataSite0/'
#     tarray2 = np.loadtxt(prefix+"rt_real.txt").T[0] #[:nt+1] 
#     gfr2 = np.loadtxt(prefix+"rt_real.txt").T[1] #[:nt+1]
#     gfi2 = np.loadtxt(prefix+"rt_imag.txt").T[1] #[:nt+1]
#     plt.plot(tarray2,gfr2,'bo-')
#     plt.plot(tarray2,gfi2,'bo-')
#     plt.show()
# 
#     run(tarray4,gfr4,gfi4,eta=0.05, rem_add='rem')
