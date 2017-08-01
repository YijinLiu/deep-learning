run_benchmark: bin/benchmark
	bin/benchmark

BLAS?=mkl

BENCHMARK_LDFLAGS/atlas=-Wl,-rpath=/usr/local/ATLAS/lib -L/usr/local/ATLAS/lib -lsatlas 
BENCHMARK_LDFLAGS/openblas=-Wl,-rpath=/usr/local/OpenBLAS/lib -L/usr/local/OpenBLAS/lib -lopenblas
BENCHMARK_LDFLAGS/mkl=-Wl,-rpath=/usr/local/intel/mkl/lib/intel64 -L/usr/local/intel/mkl/lib/intel64 -lmkl_rt

BENCHMARK_LDFLAGS=$(BENCHMARK_LDFLAGS/$(BLAS))

bin/benchmark: benchmark.cc
	mkdir -p bin
	g++ -o bin/benchmark benchmark.cc $(BENCHMARK_LDFLAGS) -l benchmark -l pthread
