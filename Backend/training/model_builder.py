# Backend/src/model_builder.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape, label_columns):
    """
    Builds a multi-output CNN model.

    Args:
        input_shape (tuple): (height, width, channels)
        label_columns (dict): {label_name: num_classes}

    Returns:
        tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.SeparableConv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.SeparableConv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.SeparableConv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.SeparableConv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Dropout(0.5)(x)

    outputs = {}
    for label_name, num_classes in label_columns.items():
        outputs[label_name] = layers.Dense(num_classes, activation="softmax", name=label_name)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
