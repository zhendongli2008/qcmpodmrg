#!/usr/bin/env python
#
# Constant matrices for MPO representation.
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
import numpy
from sysutil_include import dmrg_dtype,dmrg_mtype

sgn  = numpy.array([[ 1., 0.],[ 0.,-1.]],dtype=dmrg_dtype)
idn  = numpy.array([[ 1., 0.],[ 0., 1.]],dtype=dmrg_dtype)
idnt = numpy.array([[ 1., 0.],[ 0.,-1.]],dtype=dmrg_dtype)
cre  = numpy.array([[ 0., 0.],[ 1., 0.]],dtype=dmrg_dtype)
cret = numpy.array([[ 0., 0.],[ 1., 0.]],dtype=dmrg_dtype)
ann  = numpy.array([[ 0., 1.],[ 0., 0.]],dtype=dmrg_dtype)
annt = numpy.array([[ 0.,-1.],[ 0., 0.]],dtype=dmrg_dtype)
nii  = numpy.array([[ 0., 0.],[ 0., 1.]],dtype=dmrg_dtype)
niit = numpy.array([[ 0., 0.],[ 0.,-1.]],dtype=dmrg_dtype)

if __name__ == '__main__':
   print sgn  
   print idn  
   print idnt 
   print cre  
   print cret 
   print ann  
   print annt 
   print nii  
   print niit 
