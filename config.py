class config(object):
    def __init__(self):
        ############################
        #    hyper parameters      #
        ############################

        # For separate margin loss
        self.m_plus = 0.9 #'the parameter of m plus'
        self.m_minus = 0.1 #'the parameter of m minus'
        self.lambda_val = 0.5 #'down weight of the loss for absent digit classes'

        # for training
        self.batch_size = 64 #'batch size' #128->64, 128 will overflow memory
        self.epoch = 500 #'epoch' #origin 50 #500->50
        self.iter_routing = 3 #'number of iterations in routing algorithm'
        self.mask_with_y = True #'use the true label to mask out target capsule or not'

        self.stddev = 0.01 #'stddev for W initializer'
        self.regularization_scale = 0.392 #'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392'
        #if change 28*28 to x*x, regularization_scale need to change too

        ############################
        #   environment setting    #
        ############################
        self.dataset = 'myself' #'The name of dataset [myself, fashion-mnist'
        self.is_training = True #'train or predict phase'
        self.num_threads = 8 #'number of threads of enqueueing examples'
        self.logdir = 'logdir' #'logs directory'
        self.train_sum_freq = 10 #'the frequency of saving train summary(step)' #100
        self.val_sum_freq = 50 #'the frequency of saving valuation summary(step)' #500
        self.save_freq = 5 #'the frequency of saving model(epoch)' #3
        self.results = 'results' #'path for saving results'

        ############################
        #   distributed setting    #
        ############################
        self.num_gpu = 1 #'number of gpus for distributed training' #2
        self.batch_size_per_gpu = 128 #'batch size on 1 gpu'
        self.thread_per_gpu = 4, #'Number of preprocessing threads per tower.'

