import numpy as np
import pytest
import time

@pytest.mark.benchmark(
    group="matrix_mul",
    min_time=0.1,
    max_time=1.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_matrix_mul_128(benchmark):
    A = np.random.random([128, 128])
    B = np.random.random([128, 128])
    @benchmark
    def matrix_mul():
        A.dot(B)

@pytest.mark.benchmark(
    group="matrix_mul",
    min_time=0.1,
    max_time=1.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_matrix_mul_256(benchmark):
    A = np.random.random([256, 256])
    B = np.random.random([256, 256])
    @benchmark
    def matrix_mul():
        A.dot(B)

@pytest.mark.benchmark(
    group="matrix_mul",
    min_time=0.1,
    max_time=1.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_matrix_mul_512(benchmark):
    A = np.random.random([512, 512])
    B = np.random.random([512, 512])
    @benchmark
    def matrix_mul():
        A.dot(B)

@pytest.mark.benchmark(
    group="matrix_mul",
    min_time=0.1,
    max_time=1.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_matrix_mul_1024(benchmark):
    A = np.random.random([1024, 1024])
    B = np.random.random([1024, 1024])
    @benchmark
    def matrix_mul():
        A.dot(B)

@pytest.mark.benchmark(
    group="matrix_mul",
    min_time=0.1,
    max_time=1.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_matrix_mul_2048(benchmark):
    A = np.random.random([2048, 2048])
    B = np.random.random([2048, 2048])
    @benchmark
    def matrix_mul():
        A.dot(B)


@pytest.mark.benchmark(
    group="matrix_rank",
    min_time=0.1,
    max_time=1.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_matrix_rank_128(benchmark):
    A = np.random.random([128, 128])
    @benchmark
    def matrix_rank():
        np.linalg.matrix_rank(A)

@pytest.mark.benchmark(
    group="matrix_rank",
    min_time=0.1,
    max_time=1.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_matrix_rank_256(benchmark):
    A = np.random.random([256, 256])
    @benchmark
    def matrix_rank():
        np.linalg.matrix_rank(A)

@pytest.mark.benchmark(
    group="matrix_rank",
    min_time=0.1,
    max_time=1.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_matrix_rank_512(benchmark):
    A = np.random.random([512, 512])
    @benchmark
    def matrix_rank():
        np.linalg.matrix_rank(A)

@pytest.mark.benchmark(
    group="matrix_rank",
    min_time=0.1,
    max_time=1.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_matrix_rank_1024(benchmark):
    A = np.random.random([1024, 1024])
    @benchmark
    def matrix_rank():
        np.linalg.matrix_rank(A)

@pytest.mark.benchmark(
    group="matrix_rank",
    min_time=0.1,
    max_time=1.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_matrix_rank_2048(benchmark):
    A = np.random.random([2048, 2048])
    @benchmark
    def matrix_rank():
        np.linalg.matrix_rank(A)


'''
1. Linux i5-5575R 4.8.0-58-generic NumPy-1.13.1 with ATLAS-3.10.3
Name (time in us)               Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_matrix_mul_128          342 (1.0)          344 (1.0)          342 (1.0)          1 (1.0)          342 (1.0)
test_matrix_mul_256        1,277 (3.73)       1,311 (3.80)       1,289 (3.76)        14 (12.15)      1,281 (3.74)
test_matrix_mul_512       10,725 (31.36)     10,943 (31.74)     10,777 (31.43)       64 (54.81)     10,755 (31.42)
test_matrix_mul_1024      82,407 (240.95)    84,762 (245.82)    82,951 (241.88)     774 (653.47)    82,563 (241.18)
test_matrix_mul_2048     653,666 (>1000.0)  658,745 (>1000.0)  656,140 (>1000.0)  1,938 (>1000.0)  656,108 (>1000.0)
---------------------------------------------------------------------------------------------------------------------
test_matrix_rank_128        1.71 (1.0)         1.74 (1.0)         1.73 (1.0)       0.00 (1.0)         1.73 (1.0)
test_matrix_rank_256       12.14 (7.09)       12.34 (7.08)       12.27 (7.09)      0.05 (6.63)       12.28 (7.09)
test_matrix_rank_512       79.56 (46.41)      80.91 (46.42)      80.34 (46.40)     0.38 (43.77)      80.43 (46.40)
test_matrix_rank_1024     590.21 (344.27)    614.59 (352.56)    596.42 (344.46)    7.54 (866.31)    594.10 (342.76)
test_matrix_rank_2048   4,249.64 (>1000.0) 4,402.44 (>1000.0) 4,296.88 (>1000.0)  48.44 (>1000.0) 4,280.46 (>1000.0)
---------------------------------------------------------------------------------------------------------------------

2. Linux i5-5575R 4.8.0-58-generic NumPy-1.13.1 with OpenBLAS-0.2.19
Name (time in us)               Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_matrix_mul_128          160 (1.0)          163 (1.0)          161 (1.0)          1 (1.0)          160 (1.0)
test_matrix_mul_256        1,085 (6.78)       1,091 (6.67)       1,086 (6.74)         2 (1.60)       1,086 (6.76)
test_matrix_mul_512        8,334 (52.05)      8,884 (54.30)      8,428 (52.31)      175 (150.08)     8,354 (52.02)
test_matrix_mul_1024      65,441 (408.69)    67,739 (414.01)    66,277 (411.31)     782 (667.97)    65,986 (410.92)
test_matrix_mul_2048     527,343 (>1000.0)  535,370 (>1000.0)  529,124 (>1000.0)  2,316 (>1000.0)  528,266 (>1000.0)
---------------------------------------------------------------------------------------------------------------------
test_matrix_rank_128        1.64 (1.0)         1.69 (1.0)         1.66 (1.0)       0.01 (1.0)         1.66 (1.0)
test_matrix_rank_256        8.87 (5.39)        9.07 (5.36)        8.95 (5.37)      0.06 (4.63)        8.94 (5.37)
test_matrix_rank_512       51.01 (30.96)      52.07 (30.74)      51.62 (30.97)     0.35 (24.41)      51.66 (31.03)
test_matrix_rank_1024     388.56 (235.86)    395.24 (233.33)    391.21 (234.69)    1.83 (127.54)    390.85 (234.74)
test_matrix_rank_2048   3,308.09 (>1000.0) 3,439.03 (>1000.0) 3,385.63 (>1000.0)  37.10 (>1000.0) 3,394.25 (>1000.0) 
---------------------------------------------------------------------------------------------------------------------

3. Linux i5-5575R 4.8.0-58-generic NumPy-1.13.1 with MKL-2017.3.196
Name (time in us)               Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_matrix_mul_128          113 (1.0)          115 (1.0)          114 (1.0)           1 (1.0)          113 (1.0)
test_matrix_mul_256          776 (6.83)         797 (6.88)         782 (6.86)          7 (10.58)        778 (6.85)
test_matrix_mul_512        6,048 (53.22)      6,171 (53.24)      6,074 (53.28)        35 (52.35)      6,064 (53.31)
test_matrix_mul_1024      45,679 (401.94)    46,210 (398.64)    45,890 (402.53)      198 (288.92)    45,857 (403.09)
test_matrix_mul_2048     384,822 (>1000.0)  413,831 (>1000.0)  396,973 (>1000.0)  10,316 (>1000.0)  396,849 (>1000.0)
---------------------------------------------------------------------------------------------------------------------
test_matrix_rank_128        1.02 (1.0)         1.06 (1.0)         1.03 (1.0)        0.01 (1.0)         1.03 (1.0)
test_matrix_rank_256        5.95 (5.82)        6.17 (5.78)        6.01 (5.81)       0.09 (6.12)        5.96 (5.78)
test_matrix_rank_512       39.40 (38.52)      45.85 (42.87)      42.66 (41.18)      1.80 (120.37)     43.12 (41.81)
test_matrix_rank_1024     333.06 (325.60)    465.16 (434.92)    392.55 (378.92)    44.87 (>1000.0)   398.13 (385.97)
test_matrix_rank_2048   3,112.80 (>1000.0) 3,186.77 (>1000.0) 3,141.65 (>1000.0)   24.16 (>1000.0) 3,138.09 (>1000.0)
---------------------------------------------------------------------------------------------------------------------
'''
