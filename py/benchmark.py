import time

import numpy as np
import pytest

import feedforward
import mnist

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
    A = np.random.random([128, 128]).astype(np.float32)
    B = np.random.random([128, 128]).astype(np.float32)
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
    A = np.random.random([256, 256]).astype(np.float32)
    B = np.random.random([256, 256]).astype(np.float32)
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
    A = np.random.random([512, 512]).astype(np.float32)
    B = np.random.random([512, 512]).astype(np.float32)
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
    A = np.random.random([1024, 1024]).astype(np.float32)
    B = np.random.random([1024, 1024]).astype(np.float32)
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
    A = np.random.random([2048, 2048]).astype(np.float32)
    B = np.random.random([2048, 2048]).astype(np.float32)
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
    A = np.random.random([128, 128]).astype(np.float32)
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
    A = np.random.random([256, 256]).astype(np.float32)
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
    A = np.random.random([512, 512]).astype(np.float32)
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
    A = np.random.random([1024, 1024]).astype(np.float32)
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
    A = np.random.random([2048, 2048]).astype(np.float32)
    @benchmark
    def matrix_rank():
        np.linalg.matrix_rank(A)


training_data = None

@pytest.mark.benchmark(
    group="feedforward_network",
    min_time=1.0,
    max_time=5.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_feedforward_network_n128(benchmark):
    global training_data
    if not training_data:
        training_data = mnist.load("train")
    net = simple.Network([28 * 28, 128, 10])
    @benchmark
    def train():
        net.stochastic_gradient_descent(training_data, 1000, epochs=10, mini_batch_size=10,
                learning_rate=3.0)

@pytest.mark.benchmark(
    group="feedforward_network",
    min_time=1.0,
    max_time=5.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_feedforward_network_n256(benchmark):
    global training_data
    if not training_data:
        training_data = mnist.load("train")
    net = simple.Network([28 * 28, 256, 10])
    @benchmark
    def train():
        net.stochastic_gradient_descent(training_data, 1000, epochs=10, mini_batch_size=10,
                learning_rate=3.0)

@pytest.mark.benchmark(
    group="feedforward_network",
    min_time=1.0,
    max_time=5.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_feedforward_network_n512(benchmark):
    global training_data
    if not training_data:
        training_data = mnist.load("train")
    net = simple.Network([28 * 28, 512, 10])
    @benchmark
    def train():
        net.stochastic_gradient_descent(training_data, 1000, epochs=10, mini_batch_size=10,
                learning_rate=3.0)

@pytest.mark.benchmark(
    group="feedforward_network",
    min_time=1.0,
    max_time=5.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_feedforward_network_n1024(benchmark):
    global training_data
    if not training_data:
        training_data = mnist.load("train")
    net = simple.Network([28 * 28, 1024, 10])
    @benchmark
    def train():
        net.stochastic_gradient_descent(training_data, 1000, epochs=10, mini_batch_size=10,
                learning_rate=3.0)

@pytest.mark.benchmark(
    group="feedforward_network",
    min_time=1.0,
    max_time=5.0,
    min_rounds=10,
    timer=time.time,
    disable_gc=True,
    warmup=True
)
def test_feedforward_network_n2048(benchmark):
    global training_data
    if not training_data:
        training_data = mnist.load("train")
    net = simple.Network([28 * 28, 2048, 10])
    @benchmark
    def train():
        net.stochastic_gradient_descent(training_data, 1000, epochs=10, mini_batch_size=10,
                learning_rate=3.0)

'''
1. Linux i5-5575R 4.8.0-58-generic NumPy-1.13.1 with ATLAS-3.10.3
Name (time in us)                    Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_matrix_mul_128               285 (1.0)          292 (1.0)          287 (1.0)          2 (1.0)          286 (1.0)
test_matrix_mul_256               782 (2.74)         818 (2.80)         787 (2.74)        10 (4.66)         783 (2.73)
test_matrix_mul_512             5,983 (20.96)      6,076 (20.80)      6,011 (20.90)       30 (12.95)      6,004 (20.95)
test_matrix_mul_1024           48,022 (168.22)    48,436 (165.77)    48,121 (167.30)     118 (50.17)     48,099 (167.79)
test_matrix_mul_2048          350,508 (>1000.0)  354,861 (>1000.0)  351,747 (>1000.0)  1,284 (544.52)   351,403 (>1000.0)

Name (time in ms)                    Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_matrix_rank_128             1.72 (1.0)         1.75 (1.0)         1.73 (1.0)       0.00 (1.0)         1.72 (1.0)
test_matrix_rank_256            11.70 (6.78)       11.80 (6.71)       11.74 (6.78)      0.03 (3.60)       11.73 (6.79)
test_matrix_rank_512            77.17 (44.70)      77.80 (44.25)      77.32 (44.61)     0.19 (20.08)      77.24 (44.67)
test_matrix_rank_1024          565.43 (327.51)    575.13 (327.05)    569.05 (328.31)    4.11 (421.32)    566.48 (327.61)
test_matrix_rank_2048        4,354.50 (>1000.0) 4,446.46 (>1000.0) 4,395.12 (>1000.0)  31.75 (>1000.0) 4,391.54 (>1000.0)

Name (time in s)                     Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_feedforward_network_n128    2.25 (1.0)         2.27 (1.0)        2.25 (1.0)      0.0079 (1.0)        2.25 (1.0)
test_feedforward_network_n256    4.23 (1.88)        4.28 (1.89)       4.24 (1.88)     0.0151 (1.90)       4.24 (1.88)
test_feedforward_network_n512    9.31 (4.14)        9.50 (4.19)       9.43 (4.18)     0.0596 (7.54)       9.44 (4.19)
test_feedforward_network_n1024  40.37 (17.94)      41.91 (18.46)     41.32 (18.30)    0.5238 (66.27)     41.53 (18.40)
test_feedforward_network_n2048 106.37 (47.27)     113.47 (49.97)    110.68 (49.01)    2.3463 (296.85)   110.77 (49.09)


2. Linux i5-5575R 4.8.0-58-generic NumPy-1.13.1 with OpenBLAS-0.2.19
Name (time in us)                    Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_matrix_mul_128                72 (1.0)           76 (1.0)           74 (1.0)          1 (1.0)           74 (1.0)
test_matrix_mul_256               466 (6.40)         480 (6.31)         472 (6.33)         4 (4.10)         471 (6.31)
test_matrix_mul_512             3,474 (47.60)      3,586 (47.07)      3,515 (47.11)       43 (35.90)      3,493 (46.74)
test_matrix_mul_1024           26,996 (369.83)    27,828 (365.18)    27,418 (367.45)     298 (247.71)    27,442 (367.15)
test_matrix_mul_2048          213,313 (>1000.0)  219,174 (>1000.0)  215,669 (>1000.0)  1,762 (>1000.0)  215,440 (>1000.0)

Name (time in ms)                    Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_matrix_rank_128             1.75 (1.0)         1.79 (1.0)         1.76 (1.0)       0.01 (1.0)         1.76 (1.0)
test_matrix_rank_256             8.85 (5.06)        9.29 (5.19)        9.00 (5.10)      0.15 (11.87)       8.94 (5.07)
test_matrix_rank_512            51.49 (29.39)      52.82 (29.48)      51.90 (29.37)     0.45 (34.47)      51.69 (29.29)
test_matrix_rank_1024          389.33 (222.25)    401.40 (224.01)    393.01 (222.35)    3.84 (294.43)    392.40 (222.32)
test_matrix_rank_2048        3,411.22 (>1000.0) 3,547.44 (>1000.0) 3,463.55 (>1000.0)  41.92 (>1000.0) 3,445.71 (>1000.0)

Name (time in s)                     Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_feedforward_network_n128    2.23 (1.0)         2.26 (1.0)         2.23 (1.0)      0.0088 (1.47)       2.23 (1.0)
test_feedforward_network_n256    4.27 (1.91)        4.28 (1.90)        4.27 (1.91)     0.0060 (1.0)        4.27 (1.91)
test_feedforward_network_n512    9.62 (4.31)        9.86 (4.36)        9.74 (4.35)     0.0840 (14.06)      9.78 (4.37)
test_feedforward_network_n1024  26.50 (11.88)      28.44 (12.58)      27.54 (12.30)    0.6463 (108.17)    27.66 (12.37)
test_feedforward_network_n2048  74.53 (33.39)      77.07 (34.10)      76.06 (33.96)    0.7468 (124.99)    75.99 (33.98)


3. Linux i5-5575R 4.8.0-58-generic NumPy-1.13.1 with MKL-2017.3.196
Name (time in us)                    Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_matrix_mul_128                53 (1.0)           54 (1.0)           53 (1.0)         0.22 (1.0)          53 (1.0)
test_matrix_mul_256               389 (7.28)         392 (7.24)         390 (7.27)        0.98 (4.42)        390 (7.27)
test_matrix_mul_512             3,047 (56.93)      3,096 (57.11)      3,075 (57.21)      17.68 (79.39)     3,080 (57.38)
test_matrix_mul_1024           23,001 (429.65)    23,545 (434.25)    23,304 (433.56)    168.77 (757.76)   23,314 (434.31)
test_matrix_mul_2048          194,142 (>1000.0)  197,216 (>1000.0)  195,306 (>1000.0) 1,079.77 (>1000.0) 195,051 (>1000.0)

Name (time in ms)                    Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_matrix_rank_128             1.01 (1.0)         1.02 (1.0)         1.01 (1.0)        0.00 (1.0)         1.01 (1.0)
test_matrix_rank_256             5.95 (5.87)        5.99 (5.83)        5.97 (5.86)       0.01 (3.29)        5.97 (5.86)
test_matrix_rank_512            37.48 (36.93)      37.97 (36.90)      37.69 (36.96)      0.14 (29.60)      37.71 (37.04)
test_matrix_rank_1024          344.51 (339.42)    347.99 (338.15)    345.89 (339.16)     1.29 (261.39)    345.66 (339.41)
test_matrix_rank_2048        3,116.04 (>1000.0) 3,148.08 (>1000.0) 3,131.82 (>1000.0)   11.51 (>1000.0) 3,131.41 (>1000.0)

Name (time in s)                     Min               Max                Mean            StdDev             Median
---------------------------------------------------------------------------------------------------------------------
test_feedforward_network_n128    1.81 (1.0)         1.83 (1.0)         1.82 (1.0)      0.0057 (1.0)         1.82 (1.0)
test_feedforward_network_n256    3.40 (1.87)        3.41 (1.86)        3.40 (1.87)     0.0060 (1.05)        3.41 (1.87)
test_feedforward_network_n512    7.97 (4.39)        8.07 (4.39)        8.02 (4.40)     0.0365 (6.35)        8.02 (4.40)
test_feedforward_network_n102   20.08 (11.04)      21.47 (11.68)      20.68 (11.33)    0.4192 (72.95)      20.61 (11.31)
test_feedforward_network_n2048  58.17 (31.98)      60.23 (32.76)      59.32 (32.51)    0.6438 (112.05)     59.42 (32.59)
'''
