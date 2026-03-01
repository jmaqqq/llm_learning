import os
import gzip
import numpy as np
import struct
from urllib.request import urlretrieve

BASE_URL = 'https://raw.githubusercontent.com/fgnt/mnist/master/'
FILES = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz'
]

def download_mnist():
    for f in FILES:
        if not os.path.exists(f):
            print(f"正在下载 {f}...")
            urlretrieve(BASE_URL + f, f)
    print("下载完成！")

def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 读取二进制头部信息
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # 将剩下的字节流转化为 0-1 之间的浮点数矩阵
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
        return images.astype(np.float32) / 255.0

download_mnist()
images = load_images('train-images-idx3-ubyte.gz')

print(f"图片矩阵维度: {images[0].shape}")