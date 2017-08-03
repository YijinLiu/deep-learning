import math

import numpy as np

def down_sample(img, width, height, width_rate, height_rate):
    data = []
    for row in xrange(0, width, width_rate):
        for col in xrange(0, height, height_rate):
            sum = 0.0
            for i in xrange(0, width_rate):
                for j in xrange(0, height_rate):
                    index = (row + i) * width + col + j
                    sum += img[index][0]
            data.append(sum / (width_rate * height_rate))
    return np.array(data, dtype=np.float32, ndmin=2).transpose()
