#!/usr/bin/env python
#
# Some global settings: especially for data type used in DMRG!
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
import os
import ctypes
import numpy
from mpi4py import MPI

pth = os.path.dirname(os.path.abspath(__file__))
pth = os.path.split(pth)[0] 
pth = os.path.join(pth,'libs/libqsym.so')
libqsym = ctypes.CDLL(pth)

dmrg_type = 'real'
dmrg_dtype = numpy.float_
dmrg_mtype = MPI.DOUBLE
