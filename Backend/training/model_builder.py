# Backend/src/model_builder.py

import tensorflow as tf

def build_model(input_shape, label_columns):
    """
    Builds a multi-output CNN model.

    Args:
        input_shape (tuple): (height, width, channels)
        label_columns (dict): {label_name: num_classes}

    Returns:
        tf.keras.Model
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.SeparableConv2D(32, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.SeparableConv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.SeparableConv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.SeparableConv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = {}
    for label_name, num_classes in label_columns.items():
        outputs[label_name] = tf.keras.layers.Dense(num_classes, activation="softmax", name=label_name)(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model
