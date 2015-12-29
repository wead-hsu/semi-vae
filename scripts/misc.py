import numpy as np
import theano
def binarize_labels(labels, num_classes):
    labels_oh = np.zeros([labels.shape[0], num_classes], dtype=theano.config.floatX)
    for i in xrange(labels.shape[0]):
        labels_oh[i, labels[i]] = 1
    return labels_oh



