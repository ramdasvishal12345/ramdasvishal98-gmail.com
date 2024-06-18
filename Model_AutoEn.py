import keras
from keras import layers
import numpy as np
from keras import backend as K


def Model_AutoEn(Data, Target):
    # This is the size of our encoded representations
    encoding_dim = Target.shape[1]

    # This is our input image
    input_img = keras.Input(shape=(Data.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(Target.shape[1], activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    inp = autoencoder.input  # input placeholder
    outputs = [layer.output for layer in autoencoder.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layerNo = 1
    test = Data[:][np.newaxis, ...]
    test = test[0, :, :]
    test = np.asarray(test).astype(np.float32)
    layer_out = np.asarray(functors[layerNo]([test])).squeeze()
    return layer_out
