# Neuron-C
## Neural network library written in C

Want to train networks in Keras/TensorFlow and run them in stand-alone programs? Then this is for you.

## Requirements
### [BLAS](http://www.netlib.org/blas/)

Too long to read? Watch [this guy](https://www.youtube.com/watch?v=fiNG_Btbx0g).

BLAS is a Fortran library of expertly optimized linear algebraic functions. Neuron-C uses BLAS to do matrix-matrix and matrix-vecotr multiplication as quickly as possible. Why depend on BLAS rather than write these routines myself? BLAS is the bedrock of linear algebraic operations. As [Dr. Shusen Wang](http://wangshusen.github.io/) says, "Do not try to write and optimize these operations yourself. There are just... too many tricks."

For you to use the Neuron-C library, you'll need the "neuron.h" file included here plus three files you'll have to build yourself so that they're tailored to your machine. You can name these files anything you wish, but let's call them:
* cblas_LINUX.a
* libblas.a
* cblas.h

These files should then live in the same directory as "neuron.h" and your project code. You'll have to call your compiler to include the BLAS library. This is explained below.

The following steps successfully create BLAS utilities version 3.8.0 on Ubuntu 16.04.

Download the following files:
* [blas-3.8.0.tgz](http://www.netlib.org/blas/blas-3.8.0.tgz)
* [cblas.tgz](http://www.netlib.org/blas/blast-forum/cblas.tgz)
* [cblas.h](http://www.netlib.org/blas/cblas.h)

Open a command-line terminal and go to the directory containing these downloads. Issue the following commands:
```
tar -xvzf blas-3.8.0.tgz
cd BLAS-3.8.0/
gfortran -c -O3 *.f
ar rv libblas.a *.o
```
This builds libblas.a. Now issue the following commands:
```
tar -xvzf cblas.tgz
cd CBLAS/
nano Makefile.in
```
This last line opens nano, a Linux file editor. Go to line 25 of `Makefile.in` and change it to read
```
BLLIB = ../lib/libblas.a
```
In nano, press `Ctrl-O` `Enter` `Ctrl-X` to save and quit.
Move the file "libblas.a" you just created into `lib`, the directory you identified in `Makefile.in`.
Issue the command
```
make all
```
Now you have all three BLAS files. You can compile anything to use BLAS utilities.

## Transfer Trained Weights from Keras

(to be continued)
