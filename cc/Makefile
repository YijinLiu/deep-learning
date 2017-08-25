.PHONY: train_mnist benchmark mnist_data

PROJECT_ROOT=$(shell readlink -f $(dir $(lastword $(MAKEFILE_LIST)))/..)

train_mnist: bin/train_mnist mnist_data
	bin/train_mnist -v=1 -logtostderr

benchmark: bin/benchmark mnist_data
	bin/benchmark

BLAS?=MKL

BLAS_CXXFLAGS/ATLAS:=-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE
BLAS_CXXFLAGS/OpenBLAS:=-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE
BLAS_CXXFLAGS/MKL:=-DEIGEN_USE_MKL_ALL -I/usr/local/intel/mkl/include
BLAS_CXXFLAGS=$(BLAS_CXXFLAGS/$(BLAS))

BLAS_LDFLAGS/ATLAS:=-L/usr/local/ATLAS/lib -llapack -lcblas -lf77blas -latlas -lgfortran -lquadmath
BLAS_LDFLAGS/OpenBLAS:=-L/usr/local/OpenBLAS/lib -lopenblas -lgfortran -lquadmath
# See https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
BLAS_LDFLAGS/MKL:=-L/usr/local/intel/mkl/lib/intel64 \
	-Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group
BLAS_LDFLAGS=$(BLAS_LDFLAGS/$(BLAS))

bin/train_mnist: train_mnist.o feedforward_network.o mnist.o
	mkdir -p bin
	g++ -o $@ $^ -Wl,-Bstatic $(BLAS_LDFLAGS) -lglog -lgflags -Wl,-Bdynamic -lpthread -ldl

bin/benchmark: benchmark.o feedforward_network.o mnist.o
	mkdir -p bin
	g++ -o $@ $^ -Wl,-Bstatic $(BLAS_LDFLAGS) -lbenchmark -lglog -lgflags -Wl,-Bdynamic -lpthread -ldl

feedforward_network.o: feedforward_network.hpp

mnist.o: mnist.hpp

benchmark.o: feedforward_network.hpp mnist.hpp

train_mnist.o: feedforward_network.hpp mnist.hpp

%.o: %.cc
	g++ -c -g -O2 -std=gnu++11 -D_POSIX_C_SOURCE=200809L -I/usr/local/include/eigen3 \
		$(BLAS_CXXFLAGS) -o $@ $<

mnist_files:=$(PROJECT_ROOT)/mnist_data/train-images-idx3-ubyte \
             $(PROJECT_ROOT)/mnist_data/train-labels-idx1-ubyte \
			 $(PROJECT_ROOT)/mnist_data/t10k-images-idx3-ubyte \
             $(PROJECT_ROOT)/mnist_data/t10k-labels-idx1-ubyte

mnist_data: $(mnist_files)

$(mnist_files): % : %.gz
	gunzip -k $<

$(PROJECT_ROOT)/mnist_data/%.gz:
	mkdir -p $(dir $@)
	wget -O $@ http://yann.lecun.com/exdb/mnist/$(notdir $@)
