import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys

from config import *
from utils import load_data
from capsNet import CapsNet

cfg = config()

def save_to():
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    if cfg.is_training:
        i = 1
        loss = os.path.join(cfg.results, 'loss_{}_v{}.csv'.format(str(cfg.epoch), i))
        train_acc = os.path.join(cfg.results, 'train_acc_{}_v{}.csv'.format(str(cfg.epoch), i))
        while os.path.exists(loss):
            i += 1
            loss = os.path.join(cfg.results, 'loss_{}_v{}.csv'.format(str(cfg.epoch), i))
            train_acc = os.path.join(cfg.results, 'train_acc_{}_v{}.csv'.format(str(cfg.epoch), i))

        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        return(fd_train_acc, fd_loss)
    else:
        i = 1
        test_acc = os.path.join(cfg.results, 'test_acc2_{}_v{}.csv'.format(str(cfg.epoch), i))
        while os.path.exists(test_acc):
            i += 1
            test_acc = os.path.join(cfg.results, 'test_acc2_{}_v{}.csv'.format(str(cfg.epoch), i))

        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return(fd_test_acc)


def train(model, num_label):
    print('innnnnnnnn training')
    trX, trY, num_tr_batch = load_data(cfg.dataset, cfg.batch_size, is_training=True)

    fd_train_acc, fd_loss = save_to()
    print("\nNote: all of results will be saved to directory: " + cfg.results)
    for epoch in range(cfg.epoch):
        print("Training for epoch %d/%d:" % (epoch, cfg.epoch))
        for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
            start = step * cfg.batch_size
            end = start + cfg.batch_size
            global_step = epoch * num_tr_batch + step

            if global_step % cfg.train_sum_freq == 0:
                #_, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary])
                assert not np.isnan(loss), 'Something wrong! loss is nan...'
                print()
                print('train_acc:', train_acc)
                print('loss:', loss)

                fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                fd_loss.flush()
                fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                fd_train_acc.flush()
            else:
                pass
                #sess.run(model.train_op)

            if (epoch + 1) % cfg.save_freq == 0:
                pass
                #supervisor...save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

    fd_train_acc.close()
    fd_loss.close()


def evaluation(model, num_label):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    ###
    try:fd_test_acc = save_to()[0] #(try tuple->except IOstream)
    except TypeError:fd_test_acc = save_to()
    ###
    print('Model restored!')

    test_acc = 0
    for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
        start = i * cfg.batch_size
        end = start + cfg.batch_size
        #acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
        test_acc += acc
        print('test_acc:', test_acc)
    test_acc = test_acc / (cfg.batch_size * num_te_batch)
    print('test_acc last:', test_acc)
    fd_test_acc.write(str(test_acc))
    fd_test_acc.close()
    print('Test accuracy has been saved to ' + cfg.results + '/test_acc.csv')


def main():
    print(' Loading Graph...')
    num_label = 10
    model = CapsNet()
    print(' Graph loaded')

    if cfg.is_training:
        print(' Start training...')
        train(model, num_label)
        print('Training done')
    else:
        evaluation(model, num_label)

if __name__ == "__main__":
    try:
        if sys.argv[1] == 'train':
            cfg.is_training = True
        elif sys.argv[1] == 'test':
            cfg.is_training = False
    except IndexError: pass

    aceptable_dataset = [
        'mnist',
        'fashion-mnist',
        'myself',
        'quantitative_precipitation'
    ]
    try:
        if sys.argv[2] in aceptable_dataset:
            cfg.dataset = sys.argv[2]
    except IndexError:
        pass

    try:
        cfg.epoch = int(sys.argv[3])
    except IndexError:
        pass

    main()