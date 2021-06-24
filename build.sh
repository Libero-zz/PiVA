#!/bin/sh

mkdir build
cd build
cmake ../
make -j4
make install
ln -s ../../facedb install/
ln -s ../../models install/
cd ..

