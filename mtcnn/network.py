import tensorflow as tf
__author__ = 'Aniruddha Mandal'

class Network(object):
    def __init__(self, session, trainable: bool = True):
        """
        Initiatializes the network
        Args:
            trainable: network is trainable or not
        """
        self._session = session
        self.__trainable = trainable
        self.__layers = {}
        self.__last_layer_name = None

        with tf.compat.v1.variable_scope(self.__class__.__name__.lower()):
            self._config()

    def _config(self):
        """
        Configures the network layers
        Done using Layers() class
        """
        raise NotImplementedError("This method must be implemented by the network")

    def add_layer(self, name: str, layer_output):
        """
        Adds a layer to the network
        Args: 
            name: name of the layer to add
            layer_output: output layer
        """
        self.__layers[name] = layer_output
        self.__last_layer_name = name

    def get_layer(self, name: str = None):
        """
        Retrieves the layer by its name
        Args:
            name: name of the layer to retrieve. If name is None, the last added layer will be retrieved
        Returns:
            layer output
        """
        if name is None:
            name = self.__last_layer_name
        return self.__layers[name]

    def is_trainable(self):
        """
        Getter for trainable flag
        """
        return self.__trainable

    def set_weights(self, weights_value: dict, ignore_missing = False):
        """
        Set the weights of the network
        Args:
            weights_value: dict of weights for each layer
        """
        network_name = self.__class__.__name__.lower()
        with tf.compat.v1.variable_scope(network_name):
            for layer_name in weights_values:
                with tf.compat.v1.variable_scope(layer_name, reuse=True):
                    for param_name, data in weights_values[layer_name].items():
                        try:
                            var = tf.compat.v1.get_variable(param_name, use_resources=False)
                            self._session.run(var.assign(data))
                        except ValueError:
                            if not ignore_missing:
                                raise
                        
    def feed(self, image):
        """
        Feeds the network with an image
        Args: 
            image: image tensor
        Returns:
            network result
        """
        network_name = self.__class__.__name__.lower()
        
        with tf.compat.v1.variable_scope(network_name):
            return self._feed(image)
    
    def _feed(self, image):
        raise NotImplementedError("Method not implemented")