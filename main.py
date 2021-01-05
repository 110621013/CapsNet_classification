import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys

from config import *
from utils import load_data
from capsNet import CapsNet
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score,roc_auc_score

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
        # normal
        test_ARPFR = cfg.results + '/{}_test_ARPFR_{}.csv'.format(cfg.dataset, str(cfg.epoch))
        test_LandP = cfg.results + '/{}_test_LandP_{}.csv'.format(cfg.dataset, str(cfg.epoch))

        # rotated
        #test_ARPFR = cfg.results + '/{}_test_ARPFR_rotated_{}.csv'.format(cfg.dataset, str(cfg.epoch))
        #test_LandP = cfg.results + '/{}_test_LandP_rotated_{}.csv'.format(cfg.dataset, str(cfg.epoch))

        fd_test_ARPFR = open(test_ARPFR, 'w')
        fd_test_ARPFR.write('accuracy_score,recall_score,precision_score,f1_score,roc_auc_score\n')
        fd_test_LandP = open(test_LandP, 'w')
        fd_test_LandP.write('testing_label,testing_predict\n')
        return(fd_test_ARPFR, fd_test_LandP)


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
    try:fd_test_ARPFR, fd_test_LandP = save_to()[0], save_to()[1] #(try tuple->except IOstream)
    except TypeError:fd_test_ARPFR, fd_test_LandP = save_to()
    ###
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess: ##If the device you specify does not exist, allow TF to automatically allocate the device
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        #tf.logging.info('Model restored!')
        print('Model restored!')

        # get testing_label(349->320)
        fd_testing_label = open(os.path.join(os.getcwd(), 'results', 'testing_label.csv'), 'r')
        testing_label = []
        for i in fd_testing_label:
            zo_str = i[0]
            if zo_str == 't':continue
            if len(testing_label) == 320:break
            testing_label.append(int(zo_str))

        # get testing_predict
        testing_predict = []
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            #acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
            #test_acc += acc
            argmax_idx = sess.run(model.argmax_idx, {model.X: teX[start:end]})
            for zo in argmax_idx:
                zo_str = str(zo)
                testing_predict.append(int(zo_str))

        tn, fp, fn, tp = confusion_matrix(testing_label, testing_predict).ravel()
        print('tn, fp, fn, tp: ', tn, fp, fn, tp)
        testing_accuracy_score = accuracy_score(testing_label, testing_predict)
        testing_recall_score = recall_score(testing_label, testing_predict)
        testing_precision_score = precision_score(testing_label, testing_predict)
        testing_f1_score = f1_score(testing_label, testing_predict)
        testing_roc_auc_score = roc_auc_score(testing_label, testing_predict)
        print('accuracy_score:', testing_accuracy_score)
        print('recall_score:', testing_recall_score)
        print('precision_score:', testing_precision_score)
        print('f1_score:', testing_f1_score)
        print('roc_auc_score:', testing_roc_auc_score)

        fd_test_ARPFR.write(
            '{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(
                testing_accuracy_score,
                testing_recall_score,
                testing_precision_score,
                testing_f1_score,
                testing_roc_auc_score
            )
        )
        fd_test_ARPFR.close()
        print('Testing ARPFR has been saved')

        for i in range(len(testing_predict)):
            fd_test_LandP.write('{},{}\n'.format(str(testing_label[i]), str(testing_predict[i])))
        fd_test_LandP.close()
        print('Testing label and predict has been saved')

def rotated_evaluation(model, supervisor, num_label):
    teX, teY, num_te_batch = load_data(cfg.dataset, cfg.batch_size, is_training=False)
    ###
    try:fd_test_ARPFR, fd_test_LandP = save_to()[0], save_to()[1] #(try tuple->except IOstream)
    except TypeError:fd_test_ARPFR, fd_test_LandP = save_to()
    ###
    with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess: ##If the device you specify does not exist, allow TF to automatically allocate the device
        supervisor.saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        #tf.logging.info('Model restored!')
        print('Model restored!')

        # get testing_label(349->320)
        fd_testing_label = open(os.path.join(os.getcwd(), 'results', 'testing_images_rotated_label.csv'), 'r')
        testing_label = []
        for i in fd_testing_label:
            zo_str = i[0]
            if zo_str == 't':continue
            if len(testing_label) == 320:break
            testing_label.append(int(zo_str))

        # get testing_predict
        testing_predict = []
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            #acc = sess.run(model.accuracy, {model.X: teX[start:end], model.labels: teY[start:end]})
            #test_acc += acc
            argmax_idx = sess.run(model.argmax_idx, {model.X: teX[start:end]})
            for zo in argmax_idx:
                zo_str = str(zo)
                testing_predict.append(int(zo_str))

        tn, fp, fn, tp = confusion_matrix(testing_label, testing_predict).ravel()
        print('tn, fp, fn, tp: ', tn, fp, fn, tp)
        testing_accuracy_score = accuracy_score(testing_label, testing_predict)
        testing_recall_score = recall_score(testing_label, testing_predict)
        testing_precision_score = precision_score(testing_label, testing_predict)
        testing_f1_score = f1_score(testing_label, testing_predict)
        testing_roc_auc_score = roc_auc_score(testing_label, testing_predict)
        print('accuracy_score:', testing_accuracy_score)
        print('recall_score:', testing_recall_score)
        print('precision_score:', testing_precision_score)
        print('f1_score:', testing_f1_score)
        print('roc_auc_score:', testing_roc_auc_score)

        fd_test_ARPFR.write(
            '{:.5f},{:.5f},{:.5f},{:.5f},{:.5f}\n'.format(
                testing_accuracy_score,
                testing_recall_score,
                testing_precision_score,
                testing_f1_score,
                testing_roc_auc_score
            )
        )
        fd_test_ARPFR.close()
        print('Testing ARPFR has been saved')

        for i in range(len(testing_predict)):
            fd_test_LandP.write('{},{}\n'.format(str(testing_label[i]), str(testing_predict[i])))
        fd_test_LandP.close()
        print('Testing label and predict has been saved')

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
        #rotated_evaluation(model, sv, num_label)


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
        'quantitative_precipitation_origin'
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