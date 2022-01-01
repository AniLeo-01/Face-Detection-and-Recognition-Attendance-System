from tensorflow.keras.layers import Input, Dense, Conv2D, PReLU, Flatten, Softmax, Softmax
from tensorflow.keras.models import Model
import numpy as np

class NetworkFactory:
    def pnet(self, input_shape = None):
        if input_shape is None:
            input_shape = (None, None, 3)
        
        #input layer
        p_in = Input(input_shape=input_shape)
        #1st CNN layer
        p_layer = Conv2D(10, kernel_size = (3,3), strides = (1,1), padding = 'valid')(p_in)
        p_layer = PReLU(shared_axes = [1,2])(p_layer)
        p_layer = MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'same')(p_layer)
        #2nd CNN layer
        p_layer = Conv2D(16, kernel_size = (3,3), strides = (1,1), padding = 'valid')(p_layer)
        p_layer = PReLU(shared_axes = [1,2])(p_layer)
        #3rd CNN layer
        p_layer = Conv2D(32, kernel_size = (3,3), strides = (1,1), padding = 'valid')(p_layer)
        p_layer = PReLU(shared_axes = [1,2])(p_layer)
        #1st output layer
        p_layer_out1 = Conv2D(2, kernel_size = (1,1), strides = (1,1))(p_layer)
        p_layer_out1 = Softmax(axis = 3)(p_layer_out1) #output score from bbox regression
        #2nd output layer
        p_layer_out2 = Conv2D(4, kernel_size = (1,1), strides = (1,1))(p_layer) #output the (x,y,w,h) of the bbox

        p_net = Model(p_in, [p_layer_out2, p_layer_out1])

        return p_net

    def rnet(self, input_shape = None):
        if input_shape is None:
            input_shape = (24, 24, 3)
        #input layer
        r_in = Input(input_shape)
        #1st CNN layer
        r_layer = Conv2D(28, kernel_size= (3,3), strides = (1,1), padding='valid')(r_in)
        r_layer = PReLU(shared_axis = [1,2])(r_layer)
        r_layer = MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = 'same')(r_layer)
        #2nd CNN layer
        r_layer = Conv2D(48, kernel_size = (3,3), strides = (1,1), padding = 'valid')(r_layer)
        r_layer = PReLU(shared_axis = [1,2])(r_layer)
        r_layer = MaxPooling2D(pool_size=(3,3), strides = (2,2), padding = 'same')(r_layer)
        #4th CNN layer
        r_layer = Conv2D(64, kernel_size = (2,2), strides = (1,1), padding='valid')(r_layer)
        r_layer = PReLU(shared_axis = [1,2])(r_layer)
        #Flatten layer
        r_layer = Flatten()(r_layer)
        #Dense Layer
        r_layer = Dense(128)(r_layer)
        r_layer = PReLU()(r_layer)
        #Output layer 1
        r_layer_out1 = Dense(2)(r_layer)
        r_layer_out1 = Softmax(axis = 1)(r_layer_out1)
        #output layer 2
        r_layer_out2 = Dense(4)(r_layer)

        r_net = Model(r_in, [r_layer_out2, r_layer_out1])

        return r_net

    def onet(self, input_shape = None):
        if input_shape is None:
            input_shape = (48, 48, 3)

        #input layer
        o_in = Input(input_shape)
        #1st CNN layers
        o_layer = Conv2D(32, kernel_size = (3,3), strides = (1,1), padding= 'valid')(o_in)
        o_layer = PReLU(shared_axis=[1,2])(o_layer)
        o_layer = MaxPooling2D(pool_size=(3,3), strides = (2,2), padding= 'same')(o_layer)
        #2nd CNN layers
        o_layer = Conv2D(64, kernel_size = (3,3), strides = (1,1), padding = 'valid')(o_layer)
        o_layer = PReLU(shared_axes = [1,2])(o_layer)
        o_layer = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding = 'valid')(o_layer)
        #3rd CNN layer
        o_layer = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid')(o_layer)
        o_layer = PReLU(shared_axes=[1,2])(o_layer)
        o_layer = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(o_layer)
        #4th CNN layer
        o_layer = Conv2D(128, kernel_size=(2,2), strides = (1,1), padding='valid')(o_layer)
        o_layer = PReLU(shared_axes = [1,2])(o_layer)
        #Dense layer
        o_layer = Flatten()(o_layer)
        o_layer = Dense(256)(o_layer)
        o_layer = PReLU()(o_layer)
        #1st output layer
        o_layer_out1 = Dense(2)(o_layer)
        o_layer_out1 = Softmax(axis=1)(o_layer_out1)   #2 element face or not classifier
        #2nd output layer
        o_layer_out2 = Dense(4)(o_layer)    #4 element bounding box vector
        o_layer_out3 = Dense(10)(o_layer)   #10 element face localization vector

        o_net = Model(o_in, [o_layer_out2, o_layer_out3, o_layer_out1])
        return o_net

    def load_net(self, weight_file):
        weights = np.load(weight_file, allow_pickle=True).tolist()
        
        p_net = self.pnet()
        r_net = self.rnet()
        o_net = self.onet()

        p_net.set_weights(weights['pnet'])
        r_net.set_weights(weights['rnet'])
        o_net.set_weights(weights['onet'])

        return p_net, r_net, o_net
    
    

        
