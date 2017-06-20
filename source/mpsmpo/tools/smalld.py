#
# https://en.wikipedia.org/wiki/Wigner_D-matrix#Wigner_.28small.29_d-matrix
#
import math
import numpy
import scipy.special
def value(j,mp,ms,beta0):
   r1 = j-math.floor(j)
   r2 = mp-math.floor(mp)
   r3 = ms-math.floor(ms)
   thresh = 1.e-10
   # Inconsistent j,mp,ms values
   if abs(r1-r2)>thresh or abs(r2-r3)>thresh or abs(r1-r3)>thresh:
      return 0.0 
   beta = float(beta0)	
   f1 = scipy.special.gamma(j+mp+1)
   f2 = scipy.special.gamma(j-mp+1)
   f3 = scipy.special.gamma(j+ms+1)
   f4 = scipy.special.gamma(j-ms+1)
   f1234 = math.sqrt(f1*f2*f3*f4)
   cb = math.cos(beta/2.0)
   sb = math.sin(beta/2.0)
   val = 0.0
   thresh = -0.1
   for s in numpy.arange(0.0,2*j+0.1):
      if j+ms-s>thresh and \
 	 mp-ms+s>thresh and \
	 j-mp-s>thresh:
	 sgn = math.pow(-1.0,mp-ms+s)
         a1 = 1.0/scipy.special.gamma(j+ms-s+1)
         a2 = 1.0/scipy.special.gamma(s+1)
         a3 = 1.0/scipy.special.gamma(mp-ms+s+1)
         a4 = 1.0/scipy.special.gamma(j-mp-s+1)
         cs = math.pow(cb,2*j+ms-mp-2*s)
         ss = math.pow(sb,mp-ms+2*s)
         tmp = sgn*a1*a2*a3*a4*cs*ss
         val += tmp
   val = f1234*val 
   return val 

if __name__ == '__main__':

   from sympy.physics.quantum.spin import Rotation

   s = 1
   mp = 1
   ms = 1
   beta = 1.0
   print '\ntest-1'
   print 'd1,1,1=',0.5*(1+math.cos(beta))
   print 'sympy =',Rotation.d(s,mp,ms,beta).doit()
   print 'myval =',value(s,mp,ms,beta)

   s = 2
   mp = 2
   ms = -1
   beta = 1.0
   print '\ntest-2'
   print 'd2,2,-1=',-0.5*math.sin(beta)*(1-math.cos(beta))
   print 'sympy  =',Rotation.d(s,mp,ms,beta).doit()
   print 'myval  =',value(s,mp,ms,beta)

   s = 0.5
   mp = 0.5
   ms = 0.5
   betalst = [1.0,numpy.pi]
   print '\ntest-1 [half j]'
   for beta in betalst:
      print 'd1/2,1/2,1/2=',math.cos(beta/2.0)
      print 'myval 	 =',value(s,mp,ms,beta)

   s = 1.5
   mp = 1.5
   ms = 0.5
   betalst = [0.1,1.0,numpy.pi]
   print '\ntest-2 [half j]'
   for beta in betalst:
      print 'd3/2,3/2,1/2=',-math.sqrt(3)*(1+math.cos(beta))/2*math.sin(beta/2)
      print 'myval 	 =',value(s,mp,ms,beta)
      print 'd3/2,3/2,-1/2=',math.sqrt(3)*(1-math.cos(beta))/2*math.cos(beta/2)
      print 'myval 	 =',value(s,mp,-ms,beta)
      
   beta = 1.0   
   print
   print 'd000=',value(0,0,0,beta),1.0
   print 'd100=',value(1,0,0,beta),math.cos(beta)
   print 'd200=',value(2,0,0,beta),1/2.0*(3*math.cos(beta)**2-1)
   
   print value(2,0.5,0,beta)
   print value(0,0.5,0,beta)
   print value(1.5,0,0,beta)
