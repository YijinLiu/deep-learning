import gzip
import logging
import os
import struct
import urllib

import numpy as np

IMAGE_SIZE = 28

IDX_DATA_TYPE_U8 = 0x8
IDX_DATA_TYPE_S8 = 0x9
IDX_DATA_TYPE_I16 = 0xb
IDX_DATA_TYPE_I32 = 0xc
IDX_DATA_TYPE_F32 = 0xd
IDX_DATA_TYPE_F64 = 0xe

DATA_DIR = "mnist_data"

def load(name, expand=False):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    images_file_name = name + "-images-idx3-ubyte.gz"
    images_file_path = os.path.join(DATA_DIR, images_file_name)
    if not os.path.isfile(images_file_path):
        url = "http://yann.lecun.com/exdb/mnist/" + images_file_name
        logging.info("Downloading %s ..." % url)
        urllib.urlretrieve(url, images_file_path)
    images = load_images(images_file_path)

    labels_file_name = name + "-labels-idx1-ubyte.gz"
    labels_file_path = os.path.join(DATA_DIR, labels_file_name)
    if not os.path.isfile(labels_file_path):
        url = "http://yann.lecun.com/exdb/mnist/" + labels_file_name
        logging.info("Downloading %s ..." % url)
        urllib.urlretrieve(url, labels_file_path)
    labels = load_labels(labels_file_path)
    if expand:
        expanded_labels = [None] * len(labels)
        for i in xrange(len(labels)):
            v = np.zeros((10, 1), dtype=np.float32)
            v[labels[i]] = 1.0
            expanded_labels[i] = v
        labels = expanded_labels
    return zip(images, labels)

def load_images(file_name):
    fh = gzip.open(file_name, "rb")
    (data_type, dimensions) = read_idx_file(file_name, fh)
    assert data_type == IDX_DATA_TYPE_U8, "invalid data type(%x) from file '%s'" % (data_type,
            file_name)
    assert len(dimensions) == 3, "invalid # of dimensions(%d != 3) from file '%s'" % (
            len(dimensions), file_name)
    assert dimensions[1] == 28 and dimensions[2] == 28, \
            "invalid image size(%dx%d != 28x28) from file '%s'" % (dimensions[1], dimensions[2],
                    file_name)
    num_images = dimensions[0]
    images = [None] * num_images
    for i in xrange(num_images):
        data = map(lambda x: x/256.0, bytearray(fh.read(28*28)))
        images[i] = np.array(data, dtype=np.float32, ndmin=2).transpose()
    fh.close()
    return images

def load_labels(file_name):
    fh = gzip.open(file_name, "rb")
    (data_type, dimensions) = read_idx_file(file_name, fh)
    assert data_type == IDX_DATA_TYPE_U8, "invalid data type(%x) from file '%s'" % (data_type,
            file_name)
    assert len(dimensions) == 1, "invalid # of dimensions(%d != 1) from file '%s'" % (
            len(dimensions), file_name)
    num_labels = dimensions[0]
    labels = [-1] * num_labels
    for i in xrange(num_labels):
        labels[i] = struct.unpack('B', fh.read(1))[0]
    fh.close()
    return labels

def read_idx_file(file_name, fh):
    magic = struct.unpack(">i", fh.read(4))[0]
    assert (magic & 0xffff0000) == 0, "invalid magic number(%x) from file '%s'" % (magic, file_name)
    data_type = (magic >> 8)
    n = (magic & 0xff)
    dimensions = [0] * n
    for i in xrange(n):
        dimensions[i] = struct.unpack(">i", fh.read(4))[0]
    return (data_type, dimensions)
