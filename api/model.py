# -*- coding: utf-8 -*-

import os
import string
import numpy as np
import tensorflow as tf
from keras.layers import (
    Dense,
    Convolution2D,
    MaxPooling2D,
    Flatten,
)
from keras.models import Sequential
from image_proc import crop_resize, pad_resize, crop_letters

WEIGHTS_BACKUP = "weights/comnist_keras.hdf5"
SIZE = 32

INF = 10**9


def load_model(weight=None, nb_classes=26):
    """Get the convolutional model to be used to read letters

    :param weight: string
        path to the training weigths
    :param nb_classes: int
        number of expected output classes
    :return: mode: keras.model
        the convolutional model
    """
    if weight is None:
        weight = WEIGHTS_BACKUP

    # fix random seed for reproducibility
    np.random.seed(7)

    # number of convolutional filters to use
    nb_filters = 32
    nb_filters2 = 64
    nb_filters3 = 128
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    input_shape = (SIZE, SIZE, 1)

    # create model
    model = Sequential()

    model.add(
        Convolution2D(
            nb_filters,
            (kernel_size[0], kernel_size[1]),
            padding="valid",
            input_shape=input_shape,
            activation="relu",
        )
    )
    model.add(
        Convolution2D(nb_filters2, (kernel_size[0], kernel_size[1]), activation="relu")
    )
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(
        Convolution2D(nb_filters3, (kernel_size[0], kernel_size[1]), activation="relu")
    )
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))

    model.add(Dense(nb_classes))

    if os.path.exists(weight):
        # load weights
        model.load_weights(weight)

    # Compile model (required to make predictions)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("Created model and loaded weights from file")

    return model


def load_letter_predictor(weight=None, nb_classes=26, lang_in="en", nb_output=1):
    """Create a function that will classify images to letters

    :param weight: string
        path to the training weigths
    :param nb_classes: int
        number of expected output classes (letters in alphabet)
    :param lang_in: string
        language in which the letters are written
    :return: function
        a function that convert an image to a letter
    """

    model = load_model(weight, nb_classes)
    if lang_in == "en":
        LETTERS = string.ascii_uppercase
    elif lang_in == "ru":
        LETTERS = "IАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

    def letter_predictor(img, nb_output):
        """Reshape and resize images before classifying

        :param img: PIL.Image
            image of a single letter
        :return: string
            the first and second most probable letters represented by the image
        """
        img = crop_resize(img, -1)
        img = pad_resize(img, SIZE)
        img = np.reshape(img, (1, SIZE, SIZE, 1))

        # Compute probability for each possible letter
        proba_list = model.predict(img, verbose=0)[0]
        probable_letters_list = []
        for _ in range(nb_output):
            # Get index of most probable letter not already identified
            ind = int(np.argmax(proba_list))
            # Add this lette to the output list
            probable_letters_list.append(LETTERS[ind])
            # Remove most probable letter from probability list
            proba_list[ind] = -INF

        return probable_letters_list

    return letter_predictor


def load_word_predictor(weight=None, nb_classes=26, lang_in="en"):
    """Create a function that will convert images to words

    :param weight: string
        path to the training weigths
    :param nb_classes: int
        number of expected output classes (letters in alphabet)
    :param lang_in: string
        language in which the letters are written
    :return: function
        a function that convert an image to a word
    """
    letter_predictor = load_letter_predictor(weight, nb_classes, lang_in)

    def word_predictor(img, nb_output=1):
        """Splits image of word into one image per letter

        :param img: PIL.Image
            image of a word
        :param nb_output: int
            return the n first most probable letters identified on the image
        :return: string
            the word represented by the image
        """
        word = np.empty((100, nb_output), dtype=object)
        nb_letters = 0
        for i, letter in enumerate(crop_letters(img)):
            letters = letter_predictor(letter, nb_output)
            nb_letters += 1
            # Deal with exception of letter Ы
            # which is possibly made of two distinct blocks
            if lang_in == "ru" and letters[0] == "I":
                letter = "Ы"
                try:
                    word = word[:-1, :]
                    nb_letters -= 1
                except Exception as e:
                    pass
            word[i] = letters

        return word[:nb_letters]

    return word_predictor


def load_dataset():
        return tf.keras.utils.image_dataset_from_directory(
            directory='/home/aris/Documents/CoMNIST/images/Cyrillic',
            labels='inferred',
            label_mode='int',
            class_names=None,
            color_mode='rgb',
            batch_size=32,
            image_size=(256, 256),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=False,
            **kwargs
        )
