from lasagne import layers
import semi_vae
import cPickle as pkl
import gzip
from theano import tensor as T
from lasagne import updates
import theano
import numpy as np
from batchiterator import *

def init_configurations():
    params = {}
    params['data_path'] = '../data/mnist.pkl.gz'
    params['batch_size'] = 200
    params['num_classes'] = 10
    params['dim_z'] = 50
    params['num_units_hidden_common'] = 500
    params['num_samples_train_label'] = 1000 # the first n samples in trainset.
    params['epoch'] = 1000
    params['valid_period'] = 20 # temporary exclude validset
    params['test_period'] = 20
    return params


def load_data(params):
    data_path = params['data_path']
    train, dev, test = pkl.load(gzip.open(data_path))
    params['image_shape'] = (28, 28)
    params['num_samples_train'] = train[0].shape[0]
    params['num_samples_dev'] = dev[0].shape[0]
    params['num_samples_test'] = test[0].shape[0]
    params['alpha'] = 0.1
    return train, dev, test


def build(params):
    image_shape = params['image_shape']
    image_layer = layers.InputLayer([params['batch_size'], image_shape[0] * image_shape[1]])
    label_layer = layers.InputLayer([params['batch_size'], params['num_classes']])

    # rewighted alpha
    reweighted_alpha = (params['alpha'] * params['num_samples_train'] / params['num_samples_train_label'])
    semi_vae_layer = semi_vae.SemiVAE(
        [image_layer, label_layer],
        params['num_units_hidden_common'],
        params['dim_z'],
        reweighted_alpha
    )

    sym_images = T.matrix('images')
    sym_labels = T.matrix('labels')

    cost_for_label = semi_vae_layer.get_cost_for_label([sym_images, sym_labels])
    cost_for_unlabel = semi_vae_layer.get_cost_for_unlabel(sym_images)

    cost_test, acc_test = semi_vae_layer.get_cost_test([sym_images, sym_labels])

    network_params = semi_vae_layer.get_params()
    print network_params

    update_for_label = updates.adam(cost_for_label, network_params)
    update_for_unlabel = updates.adam(cost_for_unlabel, network_params)

    fn_for_label = theano.function([sym_images, sym_labels], cost_for_label,
                                      updates = update_for_label)
    fn_for_unlabel = theano.function([sym_images], cost_for_unlabel,
                                      updates = update_for_unlabel)

    fn_for_test = theano.function([sym_images, sym_labels], [cost_test, acc_test])

    return semi_vae, fn_for_label, fn_for_unlabel, fn_for_test


def train():
    params = init_configurations()
    print params

    trainset, devset, testset = load_data(params)
    trainset_label = [trainset[0][:params['num_samples_train_label']],
                      trainset[1][:params['num_samples_train_label']]]
    trainset_unlabel = [trainset[0][params['num_samples_train_label']:],
                        trainset[1][params['num_samples_train_label']:]]

    iter_train_label = BatchIterator(range(params['num_samples_train_label']),
                                     params['batch_size'],
                                     data = trainset_label)

    num_samples_train_unlabel = params['num_samples_train'] - params['num_samples_train_label']
    iter_train_unlabel = BatchIterator(range(num_samples_train_unlabel),
                                           params['batch_size'],
                                           data = trainset_unlabel)

    iter_test = BatchIterator(range(params['num_samples_test']),
                              params['batch_size'],
                              data = testset,
                              testing = True)

    semi_vea_layer, fn_for_label, fn_for_unlabel, fn_for_test = build(params)

    for epoch in xrange(params['epoch']):
        n_batches_train_label = params['num_samples_train_label']/params['batch_size']
        n_batches_train_unlabel = (params['num_samples_train'] - params['num_samples_train_label'])/params['batch_size']

        n_batches_train = n_batches_train_label + n_batches_train_unlabel
        idx_batches = np.random.permutation(n_batches_train) # idx < n_batches_train_label is for label
        for batch in xrange(n_batches_train):
            idx = idx_batches[batch]

            # for label
            if idx < n_batches_train_label:
                images, labels = iter_train_label.next()
                train_batch_cost = fn_for_label(images, labels)
                print('Epoch %d batch %d labelled: %f' % (epoch, batch, train_batch_cost))
            else:
                images = iter_train_unlabel.next()
                train_batch_cost = fn_for_unlabel(images)
                print('Epoch %d batch %d unlabelled: %f' % (epoch, batch, train_batch_cost))

        if epoch % params['test_period'] == 0:
            n_batches_test = params['num_samples_test'] / params['batch_size']
            for batch in xrange(n_batches_test):
                images, labels = iter_test.next()
                test_batch_cost, test_batch_acc = fn_for_test(images, labels)
                print('TEST epoch %d batch %d: %f' % (epoch, batch, test_batch_cost, test_batch_acc))


if __name__ == '__main__':
    train()
