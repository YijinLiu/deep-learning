run_benchmark: bin/benchmark
	bin/benchmark

bin/benchmark: benchmark.cc
	mkdir -p bin
	g++ -o bin/benchmark benchmark.cc -L/usr/local/ATLAS/lib -lsatlas -l benchmark -l pthread
