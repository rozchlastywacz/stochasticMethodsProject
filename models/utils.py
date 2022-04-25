import numpy as np
import matplotlib.pyplot as plt


def append_ones(matrix, axis=1):
    return np.concatenate((matrix, np.ones((matrix.shape[0], 1), dtype=np.float32)), axis=axis)


def zeros(*dims):
    return np.zeros(shape=tuple(dims), dtype=np.float32)


def ones(*dims):
    return np.ones(shape=tuple(dims), dtype=np.float32)


def rand(*dims):
    return np.random.rand(*dims).astype(np.float32)


def sigmoid(matrix):
    return 1.0 / (1.0 + np.exp(-matrix))


def chunks(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def tiles(examples):
    rows_count = examples.shape[0]
    cols_count = examples.shape[1]
    tile_height = examples.shape[2]
    tile_width = examples.shape[3]

    space_between_tiles = 2
    img_matrix = np.empty(shape=(rows_count * (tile_height + space_between_tiles) - space_between_tiles,
                                 cols_count * (tile_width + space_between_tiles) - space_between_tiles))
    img_matrix.fill(np.nan)

    for r in range(rows_count):
        for c in range(cols_count):
            x_0 = r * (tile_height + space_between_tiles)
            y_0 = c * (tile_width + space_between_tiles)
            ex_min = np.min(examples[r, c])
            ex_max = np.max(examples[r, c])
            img_matrix[x_0:x_0 + tile_height, y_0:y_0 + tile_width] = (examples[r, c] - ex_min) / (ex_max - ex_min)

    plt.matshow(img_matrix, cmap='gray', interpolation='none')
    plt.axis('off')
    plt.show()
