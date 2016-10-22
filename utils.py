import tensorflow as tf
import numpy as np
import argparse

GLOBAL = {'var': []}

def batch_norm(x, n_out = None):
    assert len(list(x.get_shape())) == 4
    if n_out == None:
        n_out = int(x.get_shape()[-1])
    phase_train = GLOBAL['phase_train']
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        #normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        #normed = x
        def haha(x):
            return tf.reshape(x, [-1,1,1,n_out])
        normed = (x - haha(mean))/(haha(var) + 1e-3) * haha(gamma) + haha(beta)
        #GLOBAL['var'].append(normed)
    return normed

def decorate(x, nonlinearity = 'relu', batchnorm = False):
    if batchnorm == True:
        x = batch_norm(x)
    if nonlinearity == 'relu':
        return tf.nn.relu(x)
    elif nonlinearity == 'identity':
        return x
    elif nonlinearity == 'tanh':
        return tf.nn.tanh(x)
    else:
        raise NotImplementedError

def conv(x, channel, kernel_shape = 3, padding = 'VALID', stride = 1, nonlinearity = 'identity', use_cudnn_on_gpu = False, batch_norm = False):
    filter_shape = (kernel_shape, kernel_shape, int(x.get_shape()[3]), channel)

    W = tf.Variable(tf.truncated_normal( filter_shape, stddev=0.1), name = 'W')
    b = tf.Variable(tf.constant(0.001, shape=[1, 1, 1, channel]), name = "b")

    oup = tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = padding, data_format = 'NHWC', use_cudnn_on_gpu = use_cudnn_on_gpu, name = 'conv')
    oup = oup + b
    return decorate(oup, nonlinearity, batch_norm)

def pool(x, stride = 2, kernel_shape = 2, padding = 'VALID'):
    return tf.nn.max_pool(
        x, ksize = [1, kernel_shape, kernel_shape, 1],
        strides = [1, stride, stride, 1],
        padding = 'VALID',
        data_format = 'NHWC',
        name = 'pool'
    )

def flatten(x):
    t = np.prod( list(map(int, x.get_shape()[1:])) )
    return tf.reshape(x, [-1, t])

def fc(x, channel, nonlinearity = 'identity'):
    x = flatten(x)
    W = tf.Variable(tf.truncated_normal((int(x.get_shape()[1]), channel), stddev = 0.1))
    b = tf.Variable(tf.zeros([channel]))
    x = tf.matmul(x, W) + b
    return decorate(x, nonlinearity)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--minibatch_num', type = int, default = 500)
    parser.add_argument('-n', '--minibatch_size', type = int, default = 256)
    parser.add_argument('-l', '--learning_rate', type = int, default = 0.01)
    parser.add_argument('-o', '--save_path', default = 'hehe.save')
    parser.add_argument('-c', '--continue_path', default = None)
    return parser

def train(args, make_network, make_trainer):
    phase_train = tf.placeholder(tf.bool, name = 'phase_train')
    GLOBAL['phase_train'] = phase_train
    net = make_network(args)

    #train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(net['loss'])

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(net['loss'])
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)


    def build_inputs(**kwargs):
        return {net['inputs'][i]: kwargs[i] for i in kwargs}

    def train_func(**kwargs):
        keys = sorted(net['train_outputs'].keys())
        dicts = build_inputs(**kwargs)
        dicts[phase_train] = True
        ans = sess.run(
            [train_step] + [net['train_outputs'][i] for i in keys],
            feed_dict = dicts 
        )
        return {i: b for i, b in zip(keys, ans[1:])}

    def valid_func(**kwargs):
        keys = sorted(net['valid_outputs'].keys())
        dicts = build_inputs(**kwargs)
        dicts[phase_train] = False
        ans = sess.run(
            [net['train_outputs'][i] for i in keys],
            feed_dict = dicts
        )
        return {i: b for i, b in zip( keys, ans )}

    worker = make_trainer(args, train_func, valid_func)
    saver = tf.train.Saver()
    if args.continue_path != None:
        saver.restore(sess, args.continue_path)
    while True:
        worker.run()
        saver.save(sess, args.save_path)
