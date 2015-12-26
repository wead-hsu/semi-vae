'''
Author Wead-Hsu, wead-hsu@github
The implementation for paper tilted with 'semi-supervised
learning with deep generative methods'.
'''
from theano import tensor as T
from lasagne import layers
from lasagne import nonlinearities
from lasagne import objectives
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams


__All__ = ['SemiVAE']

class BasicLayer(layers.Layer):
    def __init__(self, incoming, num_units_hidden, num_units_output):
        super(BasicLayer, self).__init__(incoming)

        self.num_units_hidden = num_units_hidden
        self.num_units_output = num_units_output

        # the weight and the nonlinearity is set locally
        self.input2hidden_layer = layers.DenseLayer(incoming, num_units_hidden,
                                                    nonlinearity = T.nnet.softplus)

        self.hidden2output_layer = layers.DenseLayer(self.input2hidden_layer, num_units_output,
                                                     nonlinearity = T.nnet.softplus)


    def get_output_for(self, input):
        hidden_activation = self.input2hidden_layer.get_output_for(input)
        output_activation = self.hidden2output_layer.get_output_for(hidden_activation)
        return output_activation


    def get_output_shape_for(self, input_shape):
        return [input_shape[0], self.num_units_output]


    def get_params(self):
        params = []
        params += self.input2hidden_layer.get_params()
        params += self.hidden2output_layer.get_params()
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
        eps = self.mrg_srng.normal((self.dim_sampling,))
        return inputs[0] + inputs[1] * eps


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

        # random generator
        self.srng = RandomStreams()

        self.incomings = incomings
        self.num_classes = incomings[1].output_shape[1]
        self.num_units_hidden_common = num_units_hidden_common
        self.dim_z = dim_z
        self.alpha = alpha

        self.concat_xy  = layers.ConcatLayer(self.incomings, axis=1)
        self.encoder_mu = BasicLayer(self.concat_xy,
            num_units_hidden = self.num_units_hidden_common,
            num_units_output = self.dim_z)

        # softplus function gaurantee that output is nonnegative
        self.encoder_sigma = BasicLayer(
            self.incomings[0],
            num_units_hidden = self.num_units_hidden_common,
            num_units_output = self.dim_z)

        [image_input, label_input] = self.incomings
        dim_image = image_input.output_shape[1]
        print('dim_image: ', dim_image)

        # merge encoder_mu and encoder_sigma to get z.
        self.sampler = SamplerLayer((self.encoder_mu, self.encoder_sigma))

        self.concat_yz = layers.ConcatLayer([label_input, self.sampler], axis=1)
        self.decoder = BasicLayer(self.concat_yz,
            num_units_hidden = self.num_units_hidden_common,
            num_units_output = dim_image
        )

        self.classifier = BasicLayer(
            self.incomings[0],
            num_units_hidden = self.num_units_hidden_common,
            num_units_output = self.num_classes
        )

    def convert_onehot(self, label_input):
        return T.eye(self.num_classes)[label_input].reshape([label_input.shape[0], -1])

    def get_cost_L(self, inputs):
        # make it clear which get_output_for is used
        print('getting_cost_L')

        # inputs must obey the order.
        image_input, label_input = inputs
        label_input_oh = self.convert_onehot(label_input)
        mu_z = self.encoder_mu.get_output_for(self.concat_xy.get_output_for([image_input, label_input_oh]))
        sigma_z = self.encoder_sigma.get_output_for(image_input)
        z = self.sampler.get_output_for([mu_z, sigma_z])
        # use sigmoid function to constrain the value into [0, 1]
        reconstruct = nonlinearities.sigmoid(self.decoder.get_output_for(self.concat_yz.get_output_for([label_input_oh, z])))

        l_x = objectives.binary_crossentropy(reconstruct, image_input).sum(1)
        l_z = ((mu_z ** 2 + sigma_z ** 2 - 1 - 2*T.log(sigma_z)) * 0.5).sum(1)

        cost_L = l_z + l_x
        return cost_L


    def get_cost_U(self, image_input):
        print('getting_cost_U')
        '''
        Given unlabel data, whether label should be sampled in theano or numpy?
        Similarly, word drop should be processed in theano or numpy?
        '''

        label_input = self.srng.random_integers((image_input.shape[0],), 0, self.num_classes-1)
        cost_L = self.get_cost_L([image_input, label_input])
        label_input_oh = self.convert_onehot(label_input)
        prob_ys_given_x = nonlinearities.softmax(self.classifier.get_output_for(image_input))
        prob_y_given_x = prob_ys_given_x[T.arange(label_input.shape[0]), label_input]
        entropy_y_given_x = objectives.categorical_crossentropy(prob_ys_given_x, prob_ys_given_x)
        cost_U = -(- prob_y_given_x * cost_L + entropy_y_given_x)

        return cost_U


    def get_cost_C(self, inputs):
        image_input, label_input = inputs
        prob_y_given_x = T.nnet.softmax(self.classifier.get_output_for(image_input))[T.arange(label_input.shape[0]), label_input]
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
        prob_ys_given_x = T.nnet.softmax(self.classifier.get_output_for(image_input))
        cost_test = objectives.categorical_crossentropy(prob_ys_given_x, label_input)
        cost_acc = T.eq(T.argmax(prob_ys_given_x, axis=1), label_input) # dtype?

        return cost_test.mean(), cost_acc.mean()


    def get_params(self):
        params = []
        params += self.encoder_mu.get_params()
        params += self.encoder_sigma.get_params()
        params += self.classifier.get_params()
        params += self.sampler.get_params()

        return params
