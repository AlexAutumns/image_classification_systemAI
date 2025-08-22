# Backend/src/focal_loss.py

import tensorflow as tf
import tensorflow.keras.backend as K

@tf.keras.utils.register_keras_serializable()
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="sparse_categorical_focal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits


    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        else:
            y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())

        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])

        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        loss = weight * cross_entropy

        # sum over classes, mean over batch
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha, "from_logits": self.from_logits})
        return config