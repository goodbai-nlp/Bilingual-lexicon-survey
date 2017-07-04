#! /bin/bash
cd 3rdparty/srilm
export SRILM=$PWD
make World

cd ../../
make

