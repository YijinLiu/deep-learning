.PHONY: train_mnist caffe2_mnist tf_mnist benchmark mnist_data clean

PROJECT_ROOT=$(shell readlink -f $(dir $(lastword $(MAKEFILE_LIST)))/..)

train_mnist: bin/train_mnist mnist_data
	bin/train_mnist -logtostderr

caffe2_mnist: bin/caffe2_mnist mnist_data
	bin/caffe2_mnist -logtostderr

tf_mnist: bin/tf_mnist mnist_data
	bin/tf_mnist -logtostderr

BENCHMARK_FILTER?=

benchmark: bin/benchmark mnist_data
	bin/benchmark --benchmark_filter=$(BENCHMARK_FILTER)

BLAS?=MKL
BLAS_CXX_FLAGS/ATLAS:=-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE
BLAS_CXX_FLAGS/OpenBLAS:=-DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE
BLAS_CXX_FLAGS/MKL:=-DEIGEN_USE_MKL_ALL -I/usr/local/intel/mkl/include

LALIB?=Eigen
LA_CXX_FLAGS/Eigen:=-I/usr/local/include/eigen3 -DUSE_EIGEN
LA_CXX_FLAGS/Armadillo:=-DUSE_ARMADILLO

CAFFE2_CXX_FLAGS/MKL:=-DCAFFE2_USE_MKL
CAFFE2_CXX_FLAGS:=$(CAFFE2_CXX_FLAGS/$(BLAS)) -I/usr/local/include/eigen3

CAFFE2_LD_FLAGS:=-Wl,--whole-archive -lCaffe2_CPU -Wl,--no-whole-archive -lprotobuf

TF_LD_FLAGS:=-ltensorflow_cc

CXX_FLAGS:=-std=gnu++11 -D_POSIX_C_SOURCE=200809L -mavx2 \
    $(CAFFE2_CXX_FLAGS) $(BLAS_CXX_FLAGS/$(BLAS)) $(LA_CXX_FLAGS/$(LALIB))

BLAS_LD_FLAGS/ATLAS:=-L/usr/local/ATLAS/lib -llapack -lcblas -lf77blas -latlas -lgfortran -lquadmath
BLAS_LD_FLAGS/OpenBLAS:=-L/usr/local/OpenBLAS/lib -lopenblas -lgfortran -lquadmath
# See https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
BLAS_LD_FLAGS/MKL:=-L/usr/local/intel/mkl/lib/intel64 \
    -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group
BLAS_LD_FLAGS=$(BLAS_LD_FLAGS/$(BLAS))

bin/train_mnist: bin/train_mnist.o bin/simple_network.o bin/mnist.o
	mkdir -p bin
	g++ -o $@ $^ -Wl,-Bstatic $(BLAS_LD_FLAGS) -lglog -lgflags -Wl,-Bdynamic -lpthread -ldl

bin/caffe2_mnist: bin/caffe2.o bin/caffe2_mnist.o bin/mnist.o
	mkdir -p bin
	g++ -o $@ $^ -Wl,-Bstatic $(CAFFE2_LD_FLAGS) $(BLAS_LD_FLAGS) -lglog -lgflags -Wl,-Bdynamic -lpthread -ldl

bin/tf_mnist: bin/tf_mnist.o bin/tensorflow.o bin/mnist.o
	mkdir -p bin
	g++ -o $@ $^ -Wl,-Bstatic $(BLAS_LD_FLAGS) -lglog -lgflags -Wl,-Bdynamic $(TF_LD_FLAGS) -lpthread -ldl

bin/benchmark: bin/benchmark.o bin/caffe2.o bin/simple_network.o bin/mnist.o
	mkdir -p bin
	g++ -o $@ $^ -Wl,-Bstatic $(CAFFE2_LD_FLAGS) $(BLAS_LD_FLAGS) -lbenchmark -lglog -lgflags -Wl,-Bdynamic -lpthread -ldl

bin/simple_network.o: common.hpp simple_network.hpp linear_algebra.hpp

bin/mnist.o: common.hpp linear_algebra.hpp mnist.hpp

bin/benchmark.o: common.hpp simple_network.hpp linear_algebra.hpp mnist.hpp

bin/train_mnist.o: common.hpp simple_network.hpp linear_algebra.hpp mnist.hpp

bin/caffe2.o: caffe2.hpp common.hpp linear_algebra.hpp

bin/caffe2_mnist.o: caffe2.hpp common.hpp linear_algebra.hpp mnist.hpp

bin/tensorflow.o: tensorflow.hpp common.hpp linear_algebra.hpp

bin/tf_mnist.o: tensorflow.hpp common.hpp linear_algebra.hpp mnist.hpp

bin/%.o: %.cc
	mkdir -p bin
	g++ -c -g -O2 $(CXX_FLAGS) -o $@ $<

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

clean:
	rm -rf bin
