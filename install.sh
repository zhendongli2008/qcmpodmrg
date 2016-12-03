#
# The installation is very simple
#

echo '* Start compling qtensor utils ...'
mkdir -p libs
cd ctypes
gcc -fPIC -shared -g -O2 -o libqsym.so qsym.c
mv libqsym.so ../libs

echo '* QCMPODMRG is successfully installed!'
