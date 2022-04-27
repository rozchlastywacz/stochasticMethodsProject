# #!/usr/bin/python
# # -*- coding: utf-8 -*-
# import cv2
# import matplotlib.pyplot as plt
# import mnist
# import numpy as np
#
# from models.dbn.dbn import Dbn
# from models.utils import append_ones
#
# # digits = np.reshape(mnist.train_images()[:12*24], newshape=(12, 24, 28, 28))
# # tiles(digits)
#
#
# DATASET_SIZE = 20000  # 60000 for whole dataset
# DIGIT_SIZE = 28
#
# VISIBLE_LAYER_SIZE = DIGIT_SIZE * DIGIT_SIZE
# HIDDEN_LAYER_SIZE = 200
#
# mnist_train = mnist.train_images().astype(np.float32) / 255.0
# np.random.shuffle(mnist_train)
# dataset = np.reshape(mnist_train[:DATASET_SIZE], newshape=(DATASET_SIZE, DIGIT_SIZE * DIGIT_SIZE))
# dataset = append_ones(dataset)
#
# monitoring_indeces = np.random.choice(DATASET_SIZE, 256, replace=False)
# monitoring_set = dataset[monitoring_indeces]
#
# BATCH_SIZE = 128
# EPOCHS_COUNT = 50
#
# LEARNING_RATE = 0.1
# MOMENTUM = 0.5
#
# dbn = Dbn(VISIBLE_LAYER_SIZE, HIDDEN_LAYER_SIZE, LEARNING_RATE, MOMENTUM)
# dbn.load_weights()
#
# real_index = np.random.randint(0, DATASET_SIZE)
# real_image = mnist_train[real_index]
# generated_image = dbn.generate_image(real_image)
#
# real_image = cv2.resize(real_image, dsize=None, fx=4, fy=4)
# generated_image = cv2.resize(generated_image, dsize=None, fx=4, fy=4)
#
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(real_image, cmap='gray')
# ax[1].imshow(generated_image, cmap='gray')
# plt.show()
