.PHONY: train_mnist benchmark mnist_data

PROJECT_ROOT=$(shell readlink -f $(dir $(lastword $(MAKEFILE_LIST)))/..)

train_mnist: mnist_data
	MKL_NUM_THREADS=1 python simple.py --down_sample_rate=2

benchmark: mnist_data
	MKL_NUM_THREADS=1 py.test --benchmark-sort=mean benchmark.py

mnist_data: $(PROJECT_ROOT)/mnist_data/train-images-idx3-ubyte.gz \
            $(PROJECT_ROOT)/mnist_data/train-labels-idx1-ubyte.gz \
			$(PROJECT_ROOT)/mnist_data/t10k-images-idx3-ubyte.gz \
            $(PROJECT_ROOT)/mnist_data/t10k-labels-idx1-ubyte.gz

$(PROJECT_ROOT)/mnist_data/%.gz:
	mkdir -p $(dir $@)
	wget -O $@ http://yann.lecun.com/exdb/mnist/$(notdir $@)
