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
from lasagne import init
from lasagne import regularization
from theano.sandbox.rng_mrg import MRG_RandomStreams


__All__ = ['SemiVAE']

def normal(x, mean = 0, sd = 1):
    c = - 0.5 * np.log(2*np.pi)
    return  c - T.log(T.abs_(sd)) - (x - mean) ** 2/ (2 * sd ** 2)


class BasicLayer(layers.Layer):
    def __init__(self, incoming,
            num_units_hidden, #num_units_output,
            W = init.Normal(1e-2),
            nonlinearity_hidden = T.nnet.softplus,
            #nonlinearity_output = T.nnet.softplus,
            ):

        super(BasicLayer, self).__init__(incoming)

        self.num_units_hidden = num_units_hidden
        #self.num_units_output = num_units_output

        # the weight and the nonlinearity and W is set locally
        self.input_h1_layer = layers.DenseLayer(incoming,
                num_units =  num_units_hidden,
                W = W,
                nonlinearity = nonlinearity_hidden
                )

        '''
        self.h1_h2_layer = layers.DenseLayer(self.input_h1_layer,
                num_units = num_units_hidden,
                W = W,
                nonlinearity = nonlinearity_hidden
                )
        '''

        #self.h2_output_layer = layers.DenseLayer(self.h1_h2_layer, num_units_output,
                                                 #nonlinearity = nonlinearity_output)


    def get_output_for(self, input):
        h1_activation = self.input_h1_layer.get_output_for(input)
        #h2_activation = self.h1_h2_layer.get_output_for(h1_activation)
        #output_activation = self.h2_output_layer.get_output_for(h2_activation)
        #return output_activation
        return h1_activation


    def get_output_shape_for(self, input_shape):
        return [input_shape[0], self.num_units_hidden]


    def get_params(self):
        params = []
        params += self.input_h1_layer.get_params()
        #params += self.h1_h2_layer.get_params()
        #params += self.h2_output_layer.get_params()
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
        self.eps = self.mrg_srng.normal((inputs[0].shape[0], self.dim_sampling))
        return inputs[0] + T.exp(0.5 * inputs[1]) * self.eps


    def get_output_shape_for(self, input_shapes):
        print('samplerlayer shape: ', input_shapes[0])
        assert input_shapes[0] == input_shapes[1]
        return input_shapes[0]


    def get_params(self):
        return []


# this class is not lasagne format, since the output is not single and has
# multiple costs functions.
class SemiVAE(layers.MergeLayer):
    def __init__(self,
                 incomings,
                 num_units_hidden_common,
                 dim_z,
                 beta
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
        self.beta = beta

        self.concat_xy  = layers.ConcatLayer(self.incomings, axis=1)

        self.encoder = BasicLayer(self.concat_xy,
            num_units_hidden = self.num_units_hidden_common,
            )

        self.encoder_mu = layers.DenseLayer(
            self.encoder,
            self.dim_z,
            nonlinearity = nonlinearities.identity
            )

        self.encoder_log_var = layers.DenseLayer(
            self.encoder,
            self.dim_z,
            nonlinearity = nonlinearities.identity
            )

        [image_input, label_input] = self.incomings
        self.dim_image = image_input.output_shape[1]
        print('dim_image: ', self.dim_image)

        # merge encoder_mu and encoder_log_var to get z.
        self.sampler = SamplerLayer([self.encoder_mu, self.encoder_log_var])

        self.concat_yz = layers.ConcatLayer([label_input, self.sampler], axis=1)
        self.decoder = BasicLayer(self.concat_yz,
            num_units_hidden = self.num_units_hidden_common
            )

        self.decoder_x = layers.DenseLayer(self.decoder,
            num_units = self.dim_image,
            nonlinearity = nonlinearities.sigmoid
            )

        self.classifier_helper = BasicLayer(
            self.incomings[0],
            num_units_hidden = self.num_units_hidden_common
            )

        self.classifier = layers.DenseLayer(
            self.classifier_helper,
            num_units = self.num_classes,
            nonlinearity = nonlinearities.softmax,
            )


    def convert_onehot(self, label_input_cat):
        return T.eye(self.num_classes)[label_input_cat].reshape([label_input_cat.shape[0], -1])


    def get_cost_L(self, inputs):
        # make it clear which get_output_for is used
        print('getting_cost_L')

        # inputs must obey the order.
        image_input, label_input = inputs
        encoder = self.encoder.get_output_for(self.concat_xy.get_output_for([image_input, label_input]))
        mu_z = self.encoder_mu.get_output_for(encoder)
        log_var_z = self.encoder_log_var.get_output_for(encoder)
        z = self.sampler.get_output_for([mu_z, log_var_z])

        decoder = self.decoder.get_output_for(self.concat_yz.get_output_for([label_input, z]))
        reconstruct = self.decoder_x.get_output_for(decoder)

        l_x = objectives.binary_crossentropy(reconstruct, image_input).sum(1)
        l_z = ((mu_z ** 2 + T.exp(log_var_z) - 1 - log_var_z) * 0.5).sum(1)

        cost_L = l_x + l_z
        return cost_L


    def get_cost_U(self, image_input):
        print('getting_cost_U')
        prob_ys_given_x = self.classifier.get_output_for(self.classifier_helper.get_output_for(image_input))

        '''
        label_input_with = []
	for i in xrange(self.num_classes):
                label_input_with.append(self.convert_onehot(T.zeros([image_input.shape[0]], dtype='int64') + i))

        cost_L_with = []
	for i in xrange(self.num_classes):
                cost_L_with.append(self.get_cost_L([image_input, label_input_with[i]]))

        weighted_cost_L = T.zeros([image_input.shape[0],])
        for i in xrange(self.num_classes):
                weighted_cost_L += prob_ys_given_x[:, i] * cost_L_with[i]
        '''

        weighted_cost_L = T.zeros([image_input.shape[0],])
        for i in xrange(self.num_classes):
            label_input = T.zeros([image_input.shape[0], self.num_classes])
            label_input = T.set_subtensor(label_input[:, i], 1)
            cost_L = self.get_cost_L([image_input, label_input])
            weighted_cost_L += prob_ys_given_x[:,i] * cost_L

        entropy_y_given_x = objectives.categorical_crossentropy(prob_ys_given_x, prob_ys_given_x)
        cost_U = weighted_cost_L - entropy_y_given_x

        return cost_U


    def get_cost_C(self, inputs):
        print('getting_cost_C')
        image_input, label_input = inputs
        prob_ys_given_x = self.classifier.get_output_for(self.classifier_helper.get_output_for(image_input))
        prob_y_given_x = (prob_ys_given_x * label_input).sum(1)
        cost_C = -T.log(prob_y_given_x)
        return cost_C


    def get_cost_for_label(self, inputs):
        cost_L = self.get_cost_L(inputs)
        cost_C = self.get_cost_C(inputs)
        return cost_L.mean() + self.beta * cost_C.mean()


    def get_cost_for_unlabel(self, input):
        cost_U = self.get_cost_U(input)
        return cost_U.mean()


    def get_cost_together(self, inputs):
        label_images, label_labels, unlabel_images = inputs
        cost_for_label = self.get_cost_for_label([label_images, label_labels]) * label_images.shape[0]
        cost_for_unlabel = self.get_cost_for_unlabel(unlabel_images) * unlabel_images.shape[0]
        cost_together = (cost_for_label + cost_for_unlabel) / (label_images.shape[0] + unlabel_images.shape[0])
        cost_together += self.get_cost_prior() / 50000
        return cost_together


    def get_cost_test(self, inputs):
        image_input, label_input = inputs
        prob_ys_given_x = self.classifier.get_output_for(self.classifier_helper.get_output_for(image_input))
        cost_test = objectives.categorical_crossentropy(prob_ys_given_x, label_input)
        cost_acc = T.eq(T.argmax(prob_ys_given_x, axis=1), T.argmax(label_input, axis=1))

        return cost_test.mean(), cost_acc.mean()


    def get_cost_prior(self):
        prior_cost = 0
        params = self.get_params()
        for param in params:
            if param.name == 'W':
                prior_cost += regularization.l2(param).sum()

        return prior_cost


    def get_params(self):
        params = []
        params += self.encoder.get_params()
        params += self.encoder_mu.get_params()
        params += self.encoder_log_var.get_params()
        params += self.decoder.get_params()
        params += self.decoder_x.get_params()
        params += self.classifier_helper.get_params()
        params += self.classifier.get_params()
        params += self.sampler.get_params()

        return params
