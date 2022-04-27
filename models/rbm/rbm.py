import numpy as np
from numpy import load
from models.utils import zeros, append_ones, sigmoid, rand

DIGIT_SIZE = 28
VISIBLE_LAYER_SIZE = DIGIT_SIZE * DIGIT_SIZE
HIDDEN_LAYER_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.5


class Rbm:
    def __init__(self, visible_size=VISIBLE_LAYER_SIZE, hidden_size=HIDDEN_LAYER_SIZE, learning_rate=LEARNING_RATE,
                 momentum=MOMENTUM):
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.W = np.random.normal(scale=0.01, size=(self.visible_size + 1, self.hidden_size + 1)).astype(np.float32)
        self.W[:, -1] = 0.0
        self.W[-1, :] = 0.0
        self.M = zeros(self.visible_size + 1, self.hidden_size + 1)

    def load_weights(self, path='models\\rbm\\rbm.npz'):
        dict_weights = load(path)
        weights = dict_weights['arr_0']
        self.W = weights

    def sample_rbm(self, minibatch, steps=200):
        observations_count = minibatch.shape[0]

        visible = minibatch
        hidden = append_ones(zeros(observations_count, self.hidden_size))

        for cd_i in range(steps):
            hidden[:, :-1] = sigmoid(visible @ self.W[:, :-1])
            hidden[:, :-1] = (hidden[:, :-1] > rand(observations_count, self.hidden_size)).astype(np.float32)

            visible[:, :-1] = sigmoid(hidden @ np.transpose(self.W[:-1, :]))
            if cd_i < (steps - 1):
                visible[:, :-1] = (visible[:, :-1] > rand(observations_count, self.visible_size)).astype(np.float32)

        return visible

    def generate_image(self, real_image):
        h, w = real_image.shape
        flat_image_batch = append_ones(np.reshape(real_image, newshape=(1, h * w)))
        generated_flat_image = self.sample_rbm(flat_image_batch)
        generated_image = np.reshape(generated_flat_image[:, :-1], newshape=(h, w))
        return generated_image
