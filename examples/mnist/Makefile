all: build_neuron_model run
.PHONY: all

build_neuron_model: build_neuron_model.c neuron.h dense.h conv2d.h accum.h lstm.h gru.h pooling.h upres.h normalization.h
	gcc -c -Wall build_neuron_model.c
	gfortran -o build_neuron_model build_neuron_model.o cblas_LINUX.a libblas.a

run: run.c neuron.h dense.h conv2d.h accum.h lstm.h gru.h pooling.h upres.h normalization.h
	gcc -c -Wall run.c
	gfortran -o run run.o cblas_LINUX.a libblas.a
