import tensorflow as tf
from distutils.version import LooseVersion

__author__ = 'Aniruddha Mandal'

class Layers(object):
    """Allows to create stack layers for the network"""
    AVAILABLE_PADDINGS = ('SAME', 'VALID')

    def __init__(self, network):
        self.__network = network
    
    @staticmethod
    def __validate_padding(padding):
        if padding not in Layers.AVAILABLE_PADDINGS:
            raise Exception("Padding {} not valid".format(padding))

    @staticmethod
    def __validate_grouping(channels_input: int, channels_output: int, group: int):
        if channels_input % group != 0:
            raise Exception("The number of channels in the input does not match the group")
        
        if channels_output % group != 0:
            raise Exception("The number of channels in the output does not match the group")

    @staticmethod
    def vectorize_input(input_layer):
        input_shape = input_layer.get_shape()

        if input_shape.ndims == 4:
            #spatial input must be vectorized
            dim = 1
            for x in input_shape[1:].as_list():
                dim *= int(x)

            #dim = operator.mul(*(input_shape[1:].as_list()))
            vectorized_input = tf.reshape(input_layer, [-1, dim])
        else:
            vectorized_input, dim = (input_layer, input_shape[-1])

        return vectorized_input, dim

    def __make_var(self, name: str, shape: list):
        """
        Create a tensorflow variable with the given name and shape.
        Args:
            name: variable name
            shape: list defining the shape of the variable
        Returns:
            TF variable
        """
        return tf.compat.v1.get_variable(name, shape, trainable = self.__network.is_trainable(), use_resource = False)

    def new_feed(self, name: str, layer_shape: tuple):
        """
        Create feed layer (Input layer)
        Args: 
            name: name of the layer
        Returns:
            tf.Layer
        """
        feed_data = tf.compat.v1.placeholder(tf.float32, layer_shape, 'input')
        self.__network.add_layer(name, layer_output=feed_data)

    def new_conv(self, name: str, kernel_size: tuple, channels_output: int, stride_size: tuple, padding: str='SAME', group: int=1,
                    biased: bool=True, relu: bool=True, input_layer_name: str=None):
                """
                Creates a convolution layer for the network.
                Args:
                    name: name of the layer
                    kernel_size: kernel size of the convolution layer
                    channels_output: number of output channels
                    stride_size: stride size of the convolution layer
                    padding: Type of padding ('SAME', 'VALID')
                    group: groups of kernel operation
                    biased: to include bias or not
                    relu: whether ReLU should be applied
                    input_layer_name: name of the input layer of the current layer. If None, the last added layer will be taken
                """

                #verify padding
                self.__validate_padding(padding)

                input_layer = self.__network.get_layer(input_layer_name)

                #get number of channels in input
                channels_input = int(input_layer.get_shape()[-1])

                #verify the grouping parameter
                self.__validate_grouping(channels_input, channels_output, group)

                #convolution for given input and kernel
                convo = lambda input_val, kernel: tf.nn.conv2d(input = input_val, filters = kernel, strides = [1,stride_size[1],stride_size[0],1],
                                                                padding = padding)

                with tf.compat.v1.variable_scope(name) as scope:
                    kernel = self.__make_var('weights', shape = [kernel_size[1], kernel_size[0], channels_input // group, channels_output])

                    output = convo(input_layer, kernel)

                    #add bias if required
                    if biased:
                        biases = self.__make_var('biases', [channels_output])
                        output = tf.nn.bias_add(output, biases)

                    #apply relu if required
                    if relu:
                        output = tf.nn.relu(output, name=scope.name)

                self.__network.add_layer(name, layer_output = output)
    
    def new_prelu(self, name: str, input_layer_name: str = None):
        """
        Creates a new prelu layer with the given name and input
        Args:
            name: name of the layer
            input_layer_name: name of the layer that serves as input for the layer
        """

        input_layer = self.__network.get_layer(input_layer_name)

        with tf.compat.v1.variable_scope(name):
            channels_input = int(input_layer.get_shape()[-1])
            alpha = self.__make_var('alpha', shape = [channels_input])
            output = tf.nn.relu(input_layer) + tf.multiply(alpha, -tf.nn.relu(-input_layer))
        
        self.__network.add_layer(name, layer_output = output)

    def new_max_pool(self, name: str, kernel_size: tuple, stride_size: tuple, padding: 'SAME', input_layer_name: str=None):
        """
        Create a new max pooling layer
        Args:
            name: name of the layer
            kernel_size: kernel size of the max pooling layer (width, height)
            stride_size: stride size of the pooling layer (width, height)
            padding: Type of padding ('SAME', 'VALID')
            input_layer_name: name of the input layer of the layer. If None, will take the last added layer name
        """

        self.__validate_padding(padding)

        input_layer = self.__network.get_layer(input_layer_name)

        output = tf.nn.max_pool2d(input = input_layer, ksize = [1, kernel_size[1], kernel_size[0], 1], 
                                    strides=[1, stride_size[1], stride_size[0], 1], padding = padding, name=name)

        self.__network.add_layer(name, layer_output = output)

    def new_fully_connected(self, name: str, output_count: int, relu=True, input_layer_name: str=None):
        """
        Creates a new fully connected layer
        Args:
            name: name of the layer
            output_count: number of outputs of the FC layer
            relu: whether ReLU should be applied at last or not
            input_layer_name: name of the input layer of the FC. If None, the last added layer will be taken.
        """

        with tf.compat.v1.variable_scope(name):
            input_layer = self.__network.get_layer(input_layer_name)
            vectorized_input, dimension = self.vectorize_input(input_layer)
            weights = self.__make_var('weights', shape=[dimension, output_count])
            biases = self.__make_var('biases',  shape=[output_count])
            operation = tf.compat.v1.nn.relu_layer if relu else tf.compat.v1.nn.xw_plus_b

            fc = operation(vectorized_input, weights, biases, name=name)

        self.__network.add_layer(name, layer_output=fc)
    
    def new_softmax(self, name, axis, input_layer_name: str=None):
        """
        Creates a new softmax layer
        Args:
            name: name of the layer
            axis: axis of softmax layer
            input_layer_name: name of the input layer. If None, the last added layer will be taken
        """
        input_layer = self.__network.get_layer(input_layer_name)

        if LooseVersion(tf.__version__) < LooseVersion("1.5.0"):
            max_axis = tf.reduce_max(input_tensor = input_layer, axis=axis, keepdims=True)
            target_exp = tf.exp(input_layer - max_axis)
            normalize = tf.reduce_sum(input_tensor=target_exp, axis=axis, keepdims=True)
        else:
            max_axis = tf.reduce_max(input_tensor=input_layer, axis=axis, keepdims=True)
            target_exp = tf.exp(input_layer - max_axis)
            normalize = tf.reduce_sum(input_tensor = target_exp, axis=axis, keepdims=True)

        softmax = tf.math.divide(target_exp, normalize, name)

        self.__network.add_layer(name, layer_output=softmax)
        