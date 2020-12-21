# encoding: utf-8
"""
THE IDX FILE FORMAT

the IDX file format is a simple format for vectors and multidimensional matrices of various numerical types.
The basic format is

magic number
size in dimension 0
size in dimension 1
size in dimension 2
.....
size in dimension N
data

The magic number is an integer (MSB first). The first 2 bytes are always 0.

The third byte codes the type of the data:
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)

The 4-th byte codes the number of dimensions of the vector/matrix: 1 for vectors, 2 for matrices....
The sizes in each dimension are 4-byte integers (MSB first, high endian, like in most non-Intel processors).
The data is stored like in a C array, i.e. the index in the last dimension changes the fastest.
"""
import numpy as np
import struct
import matplotlib.pyplot as plt

#dataset_root = r'C:\Users\user\Desktop\CapsNet-Tensorflow-mnist\data\mnist'
dataset_root = r'C:\Users\user\Desktop\CapsNet-Tensorflow-myself\myself'
'''
train_images_idx3_ubyte_file = dataset_root+'\\train-images-idx3-ubyte'
train_labels_idx1_ubyte_file = dataset_root+'\\train-labels-idx1-ubyte'
test_images_idx3_ubyte_file = dataset_root+'\\t10k-images-idx3-ubyte'
test_labels_idx1_ubyte_file = dataset_root+'\\t10k-labels-idx1-ubyte'
'''
train_images_idx3_ubyte_file = dataset_root+'\\train-images-idx3-ubyte'
train_labels_idx1_ubyte_file = dataset_root+'\\train-labels-idx1-ubyte'
test_images_idx3_ubyte_file = dataset_root+'\\test-images-idx3-ubyte'
test_labels_idx1_ubyte_file = dataset_root+'\\test-labels-idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3檔的通用函數
    :param idx3_ubyte_file: idx3檔案路徑
    :return: 數據集
    """
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析檔案header
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic_number:%d, num_images: %d, num_rows*num_cols: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析數據集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '張')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1檔的通用函數
    :param idx1_ubyte_file: idx1檔案路徑
    :return: 數據集
    """
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析檔案header
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('magic_number:%d, num_images: %d張' % (magic_number, num_images))

    # 解析數據集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '張')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels, num_images


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx檔案路徑
    :return: n*row*col维np.array對象，n為圖片數量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx檔案路徑
    :return: n*1维np.array對象，n為圖片數量
    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx檔案路徑
    :return: n*row*col维np.array對象，n為圖片數量
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx檔案路徑
    :return: n*1维np.array對象，n為圖片數量
    """
    return decode_idx1_ubyte(idx_ubyte_file)




def run():
    train_images = load_train_images()
    train_labels, num_train_images = load_train_labels()
    test_images = load_test_images()
    test_labels, num_test_images = load_test_labels()

    for i in range(num_train_images):
        print(train_labels[i])
        plt.imshow(train_images[i], cmap='gray')
        plt.show()
    for i in range(num_test_images):
        print(test_labels[i])
        plt.imshow(test_images[i], cmap='gray')
        plt.show()
    print('done')

if __name__ == '__main__':
    run()