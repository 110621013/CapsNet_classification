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
        loss = cfg.results + '/{}_loss_{}.csv'.format(cfg.dataset, str(cfg.epoch))
        train_acc = cfg.results + '/{}_train_acc_{}.csv'.format(cfg.dataset, str(cfg.epoch))

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
        test_acc = cfg.results + '/{}_test_acc_{}.csv'.format(cfg.dataset, str(cfg.epoch))
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return(fd_test_acc)


def train(model, supervisor, num_label):
    print('innnnnnnnn training')
    trX, trY, num_tr_batch = load_data(cfg.dataset, cfg.batch_size, is_training=True)
    #Y = valY[:num_val_batch * cfg.batch_size].reshape((-1, 1))

    fd_train_acc, fd_loss = save_to()
    config = tf.ConfigProto() #This is used to configure the parameters of the session when creating a session:
    config.gpu_options.allow_growth = True  #Initially allocate a small amount of GPU capacity, and then slowly increase according to demand
                                            #Will not release memory, so it will cause fragmentation
    with supervisor.managed_session(config=config) as sess: #Supervisor have checkpoint, Saver and summary_computed, so we don’t need to manually change
        print("\nNote: all of results will be saved to directory: " + cfg.results)
        for epoch in range(cfg.epoch):
            print("Training for epoch %d/%d:" % (epoch, cfg.epoch))
            if supervisor.should_stop():
                print('supervisor stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc, summary_str = sess.run([model.train_op, model.total_loss, model.accuracy, model.train_summary])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'
                    supervisor.summary_writer.add_summary(summary_str, global_step)
                    print()
                    print('train_acc:', train_acc)
                    print('loss:', loss)

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)

            if (epoch + 1) % cfg.save_freq == 0:
                supervisor.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        fd_train_acc.close()
        fd_loss.close()


def evaluation(model, supervisor, num_label):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    ###
    try:fd_test_acc = save_to()[0] #(try tuple->except IOstream)
    except TypeError:fd_test_acc = save_to()
    ###
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess: ##If the device you specify does not exist, allow TF to automatically allocate the device
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        #tf.logging.info('Model restored!')
        print('Model restored!')

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
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

    sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)

    if cfg.is_training:
        print(' Start training...')
        train(model, sv, num_label)
        print('Training done')
    else:
        evaluation(model, sv, num_label)

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