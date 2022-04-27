import cv2
import mnist
import numpy as np

from models.dbn.dbn import Dbn
from models.rbm.rbm import Rbm
from models.utils import append_ones

rng = np.random.default_rng(1234)

DATASET_SIZE = 20000  # 60000 for whole dataset
DIGIT_SIZE = 28

mnist_train = mnist.train_images().astype(np.float32) / 255.0
rng.shuffle(mnist_train)
dataset = np.reshape(mnist_train[:DATASET_SIZE], newshape=(DATASET_SIZE, DIGIT_SIZE * DIGIT_SIZE))
dataset = append_ones(dataset)

monitoring_indeces = rng.choice(DATASET_SIZE, 256, replace=False)
monitoring_set = dataset[monitoring_indeces]

rbm = Rbm()
rbm.load_weights()

dbn = Dbn()
dbn.load_weights()


def get_real_image():
    real_index = np.random.randint(0, len(monitoring_set))
    real_image = monitoring_set[real_index]
    real_image = np.reshape(real_image[:-1], newshape=(DIGIT_SIZE, DIGIT_SIZE))
    return real_image


def get_image_from_rbm():
    real_image = get_real_image()
    generated_image = rbm.generate_image(real_image)
    return generated_image


def get_image_from_dbn():
    real_image = get_real_image()
    generated_image = dbn.generate_image(real_image)
    return generated_image


def rescale_grayscale_image(img):
    f = 6
    return cv2.cvtColor(cv2.resize(img, dsize=None, fx=f, fy=f), cv2.COLOR_GRAY2BGR)

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(rescale_grayscale_image(get_real_image()))
# ax[1].imshow(rescale_grayscale_image(get_image_from_dbn()))
# plt.show()
