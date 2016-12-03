#!/usr/bin/env python
#
# Author: Zhendong Li@2016-2017
#
# Subroutines:
#
# def createDIR(path):
# def deleteDIR(path,iop=1):
# def final(path,iop):
# #def rm(path,fname1,debug=False):
# 
import os
import h5py
import shutil

# MKDIR DIR
def createDIR(path,debug=True):
   if debug: print '[sysutil_io.createDIR] path=',path
   # Create directory
   if not os.path.exists(path):
      os.system('mkdir -p '+path)
   else:
      shutil.rmtree(path)
      os.mkdir(path)
   return 0

# DELETE WHOLE DIR
def deleteDIR(path,iop=1,debug=True):
   if iop == 1:
      if debug: print '[sysutil_io.deleteDIR] path=',path
      shutil.rmtree(path)
   return 0

# DELETE WHOLE TMPDIR
def final(path,iop):
   if iop == 1:
      print '\n[sysutil_io.final] delete path=',path
      shutil.rmtree(path)

## REMOVE FILES
#def rm(path,fname1,debug=False):
#   filename1 = path+'/opers_'+fname1+'.h5'
#   if debug: print " remove file = ",filename1
#   os.remove(filename1)
