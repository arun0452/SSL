import numpy as np
import random
from tensorflow.keras.utils import to_categorical


def preprocess_data(x_train, y_train, x_test, y_test):
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print(f"Before one-hot encoding: y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    print(f"After one-hot encoding: y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test


def rotate_image(im, iterations=None):
    if iterations is None:
        iterations = random.randint(0, 3)

    rotated_image = np.rot90(im, k=iterations)
    y = [0, 0, 0, 0]
    y[iterations] = 1

    return rotated_image, y


def make_rotated_data(Train, Test, subtract_pixel_mean=True):
    xy_rot_train = [rotate_image(im) for im in Train[0]]
    xy_rot_test = [rotate_image(im) for im in Test[0]]

    x_rot_train, y_rot_train = zip(*xy_rot_train)
    x_rot_test, y_rot_test = zip(*xy_rot_test)

    x_rot_train = np.array(x_rot_train, dtype='float32')
    y_rot_train = np.array(y_rot_train)
    x_rot_test = np.array(x_rot_test, dtype='float32')
    y_rot_test = np.array(y_rot_test)

    if subtract_pixel_mean:
        mean = np.mean(x_rot_train, axis=0)
        x_rot_train -= mean
        x_rot_test -= mean

    return (x_rot_train, y_rot_train), (x_rot_test, y_rot_test)
