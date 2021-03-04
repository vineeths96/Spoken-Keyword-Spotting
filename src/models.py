import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU
from parameters import *


def create_model():
    """
    Feature extractor for hotword detection
    :return: Model
    """

    model = Sequential()

    # Normalization layer
    model.add(Reshape(input_shape=INPUT_SHAPE, target_shape=TARGET_SHAPE))
    model.add(BatchNormalization())

    for num_filters in filters:
        # Convolutional layers
        model.add(Conv2D(num_filters, kernel_size=KERNEL_SIZE, padding="same"))
        model.add(BatchNormalization())
        model.add(ReLU())

        # Pooling
        model.add(MaxPooling2D(pool_size=POOL_SIZE))
        model.add(Dropout(DROPOUT))

    # Classification layers
    model.add(Flatten())
    model.add(Dense(DENSE_1, name="features512"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(DROPOUT))
    model.add(Dense(DENSE_2, name="features256"))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(DROPOUT))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.summary()

    return model
