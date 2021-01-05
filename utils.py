import os
import scipy
import numpy as np
import tensorflow as tf


def load_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'mnist')
    train_num, test_num = 60000, 10000
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((train_num, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((train_num)).astype(np.int32)

        trX = trainX[:] / 255.
        trY = trainY[:]
        num_tr_batch = train_num // batch_size

        return trX, trY, num_tr_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((test_num, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((test_num)).astype(np.int32)

        num_te_batch = test_num // batch_size
        return teX / 255., teY, num_te_batch


def load_fashion_mnist(batch_size, is_training=True):
    path = os.path.join('data', 'fashion-mnist')
    train_num, test_num = 60000, 10000
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((train_num, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((train_num)).astype(np.int32)

        trX = trainX[:] / 255.
        trY = trainY[:]
        num_tr_batch = train_num // batch_size

        return trX, trY, num_tr_batch
    else:
        fd = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((test_num, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((test_num)).astype(np.int32)

        num_te_batch = test_num // batch_size
        return teX / 255., teY, num_te_batch


def load_myself(batch_size, is_training=True):
    path = os.path.join('data', 'myself')
    train_num, test_num = 719, 359
    if is_training:
        fd = open(os.path.join(path, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((train_num, 28, 28, 1)).astype(np.float32)

        fd = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((train_num)).astype(np.int32)

        trX = trainX[:] / 255.
        trY = trainY[:]
        num_tr_batch = train_num // batch_size

        return trX, trY, num_tr_batch
    else:
        fd = open(os.path.join(path, 'test-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((test_num, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(path, 'test-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((test_num)).astype(np.int32)

        num_te_batch = test_num // batch_size
        return teX / 255., teY, num_te_batch

def load_quantitative_precipitation(batch_size, is_training=True):
    target_path = os.path.join('data', 'quantitative_precipitation')
    train_num, test_num = 719, 349
    if is_training:
        fd = np.load(os.path.join(target_path, 'training_data2.npy'))
        trainX = np.reshape(fd, (train_num, 28, 28, 1))

        fd = np.load(os.path.join(target_path, 'training_label2.npy'))
        trainY = np.reshape(fd, (train_num))

        trX = trainX[:] / 255.
        trY = trainY[:]
        num_tr_batch = train_num // batch_size

        return trX, trY, num_tr_batch
    else:
        # normal
        fd = np.load(os.path.join(target_path, 'testing_data2.npy'))
        teX = np.reshape(fd, (test_num, 28, 28, 1))
        fd = np.load(os.path.join(target_path, 'testing_label2.npy'))
        teY = np.reshape(fd, (test_num))

        # rotated
        #fd = np.load(os.path.join(target_path, 'testing_images_rotated_data.npy'))
        #teX = np.reshape(fd, (test_num, 28, 28, 1))
        #fd = np.load(os.path.join(target_path, 'testing_images_rotated_label.npy'))
        #teY = np.reshape(fd, (test_num))

        num_te_batch = test_num // batch_size
        return teX / 255., teY, num_te_batch

def load_quantitative_precipitation_origin(batch_size, is_training=True):
    target_path = os.path.join('data', 'quantitative_precipitation')
    train_num, test_num = 719, 349
    if is_training:
        fd = np.load(os.path.join(target_path, 'training_data_origin.npy'))
        trainX = np.reshape(fd, (train_num, 1000, 538, 1))

        fd = np.load(os.path.join(target_path, 'training_label_origin.npy'))
        trainY = np.reshape(fd, (train_num))

        trX = trainX[:] / 255.
        trY = trainY[:]
        num_tr_batch = train_num // batch_size

        return trX, trY, num_tr_batch
    else:
        # normal
        fd = np.load(os.path.join(target_path, 'testing_data_origin.npy'))
        teX = np.reshape(fd, (test_num, 1000, 538, 1))
        fd = np.load(os.path.join(target_path, 'testing_data_origin.npy'))
        teY = np.reshape(fd, (test_num))

        # rotated
        #fd = np.load(os.path.join(target_path, 'testing_images_rotated_data.npy'))
        #teX = np.reshape(fd, (test_num, 28, 28, 1))
        #fd = np.load(os.path.join(target_path, 'testing_images_rotated_label.npy'))
        #teY = np.reshape(fd, (test_num))

        num_te_batch = test_num // batch_size
        return teX / 255., teY, num_te_batch

def load_data(dataset, batch_size, is_training=True, one_hot=False):
    if dataset == 'mnist':
        return load_mnist(batch_size, is_training)
    elif dataset == 'fashion-mnist':
        return load_fashion_mnist(batch_size, is_training)
    elif dataset == 'myself':
        return load_myself(batch_size, is_training)
    elif dataset == 'quantitative_precipitation':
        return load_quantitative_precipitation(batch_size, is_training)
    elif dataset == 'quantitative_precipitation_origin':
        return load_quantitative_precipitation_origin(batch_size, is_training)
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def get_batch_data(dataset, batch_size, num_threads):
    trX, trY, num_tr_batch = load_data(dataset, batch_size, is_training=True)
    # tf使用了兩個線程分別執行數據讀入和數據計算，創建tf的文件名域名
    data_queues = tf.train.slice_input_producer([trX, trY]) #will be removed in a future version.
    # 隨機打亂張量的順序創建batch，batch_size一次處理的tensors數量，num_thread線程數量，capacity隊列中的最大的元素數需>min_after_dequeue混合程度
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)
    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image
    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)
