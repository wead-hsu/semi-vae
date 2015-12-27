'''
Author Wead-Hsu, wead-hsu@github
The implementation for paper tilted with 'semi-supervised
learning with deep generative methods'.
'''
import numpy as np
from theano import tensor as T
from lasagne import layers
from lasagne import nonlinearities
from lasagne import objectives
from theano.sandbox.rng_mrg import MRG_RandomStreams


__All__ = ['SemiVAE']

class BasicLayer(layers.Layer):
    def __init__(self, incoming, num_units_hidden, num_units_output,
                 nonlinearity_hidden = T.nnet.softplus, 
                 nonlinearity_output = T.nnet.softplus):

        super(BasicLayer, self).__init__(incoming)

        self.num_units_hidden = num_units_hidden
        self.num_units_output = num_units_output

        # the weight and the nonlinearity is set locally
        self.input_h1_layer = layers.DenseLayer(incoming, num_units_hidden,
                                                nonlinearity = nonlinearity_hidden)

        self.h1_h2_layer = layers.DenseLayer(self.input_h1_layer, num_units_hidden,
                                             nonlinearity = nonlinearity_hidden)

        self.h2_output_layer = layers.DenseLayer(self.h1_h2_layer, num_units_output,
                                                 nonlinearity = nonlinearity_output)


    def get_output_for(self, input):
        h1_activation = self.input_h1_layer.get_output_for(input)
        h2_activation = self.h1_h2_layer.get_output_for(h1_activation)
        output_activation = self.h2_output_layer.get_output_for(h2_activation)
        return output_activation


    def get_output_shape_for(self, input_shape):
        return [input_shape[0], self.num_units_output]


    def get_params(self):
        params = []
        params += self.input_h1_layer.get_params()
        params += self.h1_h2_layer.get_params()
        params += self.h2_output_layer.get_params()
        return params


class SamplerLayer(layers.MergeLayer):
    def __init__(self, incomings):
        super(SamplerLayer, self).__init__(incomings)
        self.mrg_srng = MRG_RandomStreams()
        self.dim_sampling = self.input_shapes[0][1]
        print('dim_sampling: ', self.dim_sampling)
        return


    def get_output_for(self, inputs):
        assert isinstance(inputs, list)
        self.eps = self.mrg_srng.normal((self.dim_sampling,))
        return inputs[0] + T.exp(inputs[1]) * self.eps


    def get_output_shape_for(self, input_shapes): 
        print('samplerlayer shape: ', input_shapes[0]) 
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


    def get_params(self):
        return []



# this class is not lasagne format, since the output is not single and have
# multiple costs functions.
class SemiVAE(layers.MergeLayer):
    def __init__(self,
                 incomings,
                 num_units_hidden_common,
                 dim_z,
                 alpha
                 ):
        '''
         params:
             incomings: input layers, [image, label]
             num_units_hidden_common: num_units_hidden for all BasicLayers.
             dim_z: the dimension of z and num_units_output for encoders BaiscLayer
        '''

        super(SemiVAE, self).__init__(incomings)
        self.mrg_srng = MRG_RandomStreams()

        # random generator
        self.incomings = incomings
        self.num_classes = incomings[1].output_shape[1]
        self.num_units_hidden_common = num_units_hidden_common
        self.dim_z = dim_z
        self.alpha = alpha

        self.concat_xy  = layers.ConcatLayer(self.incomings, axis=1)
        self.encoder_mu = BasicLayer(self.concat_xy,
            num_units_hidden = self.num_units_hidden_common,
            num_units_output = self.dim_z,
            nonlinearity_output = nonlinearities.identity)

        self.encoder_log_sigma = BasicLayer(
            self.incomings[0],
            num_units_hidden = self.num_units_hidden_common,
            num_units_output = self.dim_z,
            nonlinearity_output = nonlinearities.identity)

        [image_input, label_input] = self.incomings
        self.dim_image = image_input.output_shape[1]
        print('dim_image: ', self.dim_image)

        # merge encoder_mu and encoder_log_sigma to get z.
        self.sampler = SamplerLayer((self.encoder_mu, self.encoder_log_sigma))

        self.concat_yz = layers.ConcatLayer([label_input, self.sampler], axis=1)
        self.decoder = BasicLayer(self.concat_yz,
            num_units_hidden = self.num_units_hidden_common,
            num_units_output = self.dim_image,
            nonlinearity_output = nonlinearities.sigmoid)

        self.classifier = BasicLayer(
            self.incomings[0],
            num_units_hidden = self.num_units_hidden_common,
            num_units_output = self.num_classes,
            nonlinearity_output = nonlinearities.softmax)


    def convert_onehot(self, label_input):
        return T.eye(self.num_classes)[label_input].reshape([label_input.shape[0], -1])


    def get_cost_L(self, inputs):
        # make it clear which get_output_for is used
        print('getting_cost_L')

        # inputs must obey the order.
        image_input, label_input = inputs
        label_input_oh = self.convert_onehot(label_input)
        mu_z = self.encoder_mu.get_output_for(self.concat_xy.get_output_for([image_input, label_input_oh]))
        log_sigma_z = self.encoder_log_sigma.get_output_for(image_input)
        z = self.sampler.get_output_for([mu_z, log_sigma_z])
        reconstruct = (self.decoder.get_output_for(self.concat_yz.get_output_for([label_input_oh, z])))

        l_x = objectives.binary_crossentropy(reconstruct, image_input).sum(1)
        l_z = ((mu_z ** 2 + T.exp(log_sigma_z*2) - 1 - 2*log_sigma_z) * 0.5).sum(1)
        
        cost_L = l_x + l_z
        return cost_L


    def get_cost_U(self, image_input):
        print('getting_cost_U')
        '''
        Given unlabel data, whether label should be sampled in theano or numpy?
        Similarly, word drop should be processed in theano or numpy?
        '''
        prob_ys_given_x = (self.classifier.get_output_for(image_input))

        weighted_cost_L = T.zeros([image_input.shape[0],])
	for i in xrange(self.num_classes):
        	label_input = T.zeros([image_input.shape[0]], dtype='int64') + i
        	cost_L = self.get_cost_L([image_input, label_input])
        	prob_y_given_x = prob_ys_given_x[:, i]
                weighted_cost_L += prob_y_given_x * cost_L

        entropy_y_given_x = objectives.categorical_crossentropy(prob_ys_given_x, prob_ys_given_x)
        cost_U = - entropy_y_given_x + weighted_cost_L

        return cost_U


    def get_cost_C(self, inputs):
        print('getting_cost_C')
        image_input, label_input = inputs
        prob_y_given_x = (self.classifier.get_output_for(image_input))[T.arange(label_input.shape[0]), label_input]
        cost_C = -T.log(prob_y_given_x)
        return cost_C


    def get_cost_for_label(self, inputs):
        cost_L = self.get_cost_L(inputs)
        cost_C = self.get_cost_C(inputs)
        return cost_L.mean() + self.alpha * cost_C.mean()


    def get_cost_for_unlabel(self, input):
        cost_U = self.get_cost_U(input)
        return cost_U.mean()


    def get_cost_test(self, inputs):
        image_input, label_input = inputs
        prob_ys_given_x = (self.classifier.get_output_for(image_input))
        cost_test = objectives.categorical_crossentropy(prob_ys_given_x, label_input)
        cost_acc = T.eq(T.argmax(prob_ys_given_x, axis=1), label_input) # dtype?

        return cost_test.mean(), cost_acc.mean()


    def get_params(self):
        params = []
        params += self.encoder_mu.get_params()
        params += self.encoder_log_sigma.get_params()
        params += self.decoder.get_params()
        params += self.classifier.get_params()
        params += self.sampler.get_params()

        return params
