import numpy as np
from numpy import load

from models.rbm.rbm import Rbm
from models.utils import sigmoid, append_ones

DIGIT_SIZE = 28
VISIBLE_LAYER_SIZE = DIGIT_SIZE * DIGIT_SIZE
HIDDEN_LAYER_SIZE = 200
LEARNING_RATE = 0.1
MOMENTUM = 0.5


class Dbn:
    def __init__(self, visible_size=VISIBLE_LAYER_SIZE, hidden_size=HIDDEN_LAYER_SIZE, learning_rate=LEARNING_RATE,
                 momentum=MOMENTUM):
        self.layers = [
            Rbm(visible_size, hidden_size, learning_rate, momentum),
            Rbm(hidden_size, hidden_size, learning_rate, momentum),
            Rbm(hidden_size, hidden_size, learning_rate, momentum)
        ]

    def load_weights(self, paths=None):
        if paths is None:
            base_path = 'models/dbn/'
            paths = [base_path+'dbn_0.npz', base_path+'dbn_1.npz', base_path+'dbn_2.npz']
        if len(paths) != len(self.layers):
            raise ValueError
        for i, path in enumerate(paths):
            dict_weights = load(path)
            weights = dict_weights['arr_0']
            self.layers[i].W = weights

    def propagate_up(self, layers_count, visible):
        for i in range(layers_count):
            visible = append_ones(sigmoid(visible @ self.layers[i].W[:, :-1]))
        return visible

    def propagate_down(self, layers_count, hidden):
        for i in reversed(range(layers_count)):
            hidden = append_ones(sigmoid(hidden @ np.transpose(self.layers[i].W[:-1, :])))
        return hidden

    def sample_dbn(self, layer_idx, minibatch, steps=200):
        prop_up = self.propagate_up(layer_idx, minibatch)
        sample = self.layers[layer_idx].sample_rbm(prop_up, steps)
        prop_down = self.propagate_down(layer_idx, sample)

        return prop_down

    def generate_image(self, real_image):
        h, w = real_image.shape
        flat_image_batch = append_ones(np.reshape(real_image, newshape=(1, h * w)))
        generated_flat_image = self.sample_dbn(len(self.layers) - 1, flat_image_batch)
        generated_image = np.reshape(generated_flat_image[:, :-1], newshape=(h, w))
        return generated_image
