import cv2
import gzip
import pickle
import tensorflow as tf
import numpy as np
import utils
from utils import conv, pool, fc

def load_data():
    def split_dataset(data):
        feature, label = data
        feature = feature.reshape(feature.shape[0], 28, 28, 1)
        return feature.astype('float32'), label.astype('float32')
    dataset = 'mnist.pkl.gz'
    print('... loading data')
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
    test_set_x, test_set_y = split_dataset(test_set)
    valid_set_x, valid_set_y = split_dataset(valid_set)
    train_set_x, train_set_y = split_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

class DataProvider():
    def __init__(self):
        self.train, self.valid, self.test = load_data()

    def gen_epoch(self, args, flag = 'train'):
        if flag == 'train':
            for i in range(args.minibatch_num):
                imgs = []
                labels = []
                for j in range(args.minibatch_size):
                    t = np.random.randint(len(self.train[0]))
                    imgs.append(self.train[0][t].reshape(28, 28, 1))
                    labels.append(self.train[1][t])
                yield {
                    'img': imgs,
                    'label': labels
                }
        else:
            yield {
                'img': self.valid[0],
                'label': self.valid[1],
            }

def extract_feature(img):
    tmp = fc(img, 100, nonlinearity = 'relu')
    return tf.nn.top_k(tmp, 10, sorted = False)[0]
    x = conv(img, 20, nonlinearity = 'relu', batch_norm = True)
    x = pool(x)
    x = conv(x, 40, nonlinearity = 'relu', batch_norm = True)
    x = pool(x)
    return fc(x, 100, nonlinearity = 'relu')

def make_network(args):
    img = tf.placeholder(tf.float32, shape =[None, 28, 28, 1])
    label = tf.placeholder(tf.int64, shape = [None,])
    one_hot = tf.to_float( tf.one_hot(label, depth = 10, axis = -1, on_value = 1, off_value = 0) )

    x = extract_feature(img)
    predict = fc(x, 10)
    y = tf.nn.softmax(predict)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(one_hot * tf.log(y), reduction_indices=[1]))

    return {
        'inputs': {
            'img': img,
            'label': label
        },
        'train_outputs': {
            'y': y,
            'loss': cross_entropy,
#            'var': utils.GLOBAL['var'][0]
        },
        'valid_outputs': {
            'y': y,
            'loss': cross_entropy,
        },
        'loss': cross_entropy
    }

def make_trainer(args, train_func, valid_func):
    data = DataProvider()

    def calc_accuracy(label, pred):
        return np.mean(label == np.argmax(pred, axis = 1))

    class Trainer:
        def __init__(self, data):
            self.data = data
            pass

        def run(self):
            data_provider = self.data.gen_epoch(args, flag = 'train')
            results = []
            for i in range(args.minibatch_num):
                data = next(data_provider)
                ans = train_func(**data)
                accuracy = calc_accuracy(data['label'], ans['y'])
                print('\b\r', i, 'accuracy: ', accuracy, 'loss:', ans['loss'], end = '')
                results.append(accuracy)
            print('\ntrain accuracy', np.mean(results))

            valid = next(self.data.gen_epoch(args, flag = 'valid'))
            pred = valid_func(**valid)
            print('valid accuracy', calc_accuracy(valid['label'], pred['y']))
    return Trainer(data)

def main():
    parser = utils.make_parser()
    args = parser.parse_args()
    utils.train(args, make_network, make_trainer)

if __name__ == '__main__':
    main()
