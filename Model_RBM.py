import numpy as np
from keras import backend as K


class RBM:

    def __init__(self, visible_size, hidden_size):
        self.layers = None
        self.input = None
        self.visible_size = visible_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.weights = np.random.randn(self.hidden_size, self.visible_size)
        self.biases_v = np.random.randn(self.visible_size)
        self.biases_h = np.random.randn(self.hidden_size)

    def train(self, data, epochs=4):
        for epoch in range(epochs):
            # Sample from the visible layer
            visible_states = data

            # Sample from the hidden layer
            hidden_states = self.sample_hidden(visible_states)

            # Update the weights and biases
            self.weights += self.get_gradient(visible_states, hidden_states)

            self.biases_v += self.get_gradient_v(visible_states)
            self.biases_h += self.get_gradient_h(hidden_states)

    def sample_visible(self, hidden_states):
        # Calculate probabilities for visible states
        probabilities = self.get_probabilities(hidden_states)

        # Sample visible states from probabilities
        visible_states = np.random.binomial(1, probabilities)

        return visible_states

    def sample_hidden(self, visible_states):
        # Calculate probabilities for hidden states
        probabilities = self.get_probabilities(visible_states)

        # Sample hidden states from probabilities
        hidden_states = np.random.binomial(1, probabilities)

        return hidden_states

    def get_probabilities(self, states):
        # Calculate probabilities for visible states given hidden states
        probabilities = 1 / (1 + np.exp(-(states @ self.weights + self.biases_v)))[:, :states.shape[1]]

        # Calculate probabilities for hidden states given visible states
        # probabilities =1 / (1 + np.exp(-(states @ self.weights + self.biases_v)))

        return probabilities

    def get_gradient(self, visible_states, hidden_states):
        # Calculate gradient of weights
        gradient = (visible_states * hidden_states) - (1 - visible_states) * (1 - hidden_states)

        # Calculate gradient of biases_v
        gradient += visible_states - 1

        # Calculate gradient of biases_h
        gradient += hidden_states - 1
        gradient = np.reshape(gradient, [gradient.shape[1], gradient.shape[0]])
        return gradient

    def get_gradient_v(self, visible_states):
        # Calculate gradient of biases_v
        gradient = visible_states - 1
        gradient = gradient[:, 0]

        return gradient

    def get_gradient_h(self, hidden_states):
        # Calculate gradient of biases_h
        gradient = hidden_states - 1
        gradient = gradient[0, :]
        return gradient

    def extract_features(self, data):
        # Train the RBM on the data
        self.train(data)

        # Sample from the hidden layer
        hidden_states = self.sample_hidden(data)

        # Get the features
        features = hidden_states

        return features


# Reference links:
# * [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine)
# * [A Gentle Introduction to Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)
# * [Implementing a Restricted Boltzmann Machine in Python](https://www.oreilly.com/library/view/deep-learning-with/9781492032672/ch04.html)


def Model_RBM(Datas):
    rbm = RBM(Datas.shape[0], Datas.shape[1])
    # Train the RBM
    # rbm.train(Datas)
    # Extract the features
    features = rbm.weights
    features = np.resize(features, [features.shape[1], 200])
    # # Extract the features
    # inp = rbm.input  # input placeholder
    # outputs = [layer.output for layer in rbm.layers]  # all layer outputs
    # functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    #
    # layerNo = 0
    # data = Datas
    # Feats = []
    # for i in range(data.shape[0]):
    #     test = data[i, :, :][np.newaxis, ...]
    #     layer_out = np.asarray(functors[layerNo]([test])).squeeze()  # [func([test]) for func in functors]
    #     Feats.append(layer_out)
    # return Feats
    return features



