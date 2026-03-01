import numpy as np
import gzip
import struct


def load_muist_data():
    with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 784, 1)
        images = images.astype(np.float32) / 255.0

    with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return images, labels

images, labels = load_muist_data()

if __name__ == '__main__':

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))




