# Backend/src/utils.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def get_label_encodings(df, label_columns):
    """
    Creates label to index and index to label mappings for each label column.
    Returns dict: {label: (label_to_index, index_to_label)}
    """
    label_encodings = {}
    for label in label_columns:
        classes = sorted(df[label].dropna().unique())
        label_to_index = {lab: idx for idx, lab in enumerate(classes)}
        index_to_label = {idx: lab for idx, lab in enumerate(classes)}
        label_encodings[label] = (label_to_index, index_to_label)
    return label_encodings

def encode_labels(df, label_encodings):
    """
    Encode label columns in dataframe using provided encodings.
    """
    for label, (label_to_index, _) in label_encodings.items():
        df[label] = df[label].map(label_to_index)
    return df

def focal_loss(gamma=2., alpha=0.25):
    """
    Focal loss for multi-class classification (expects one-hot labels).

    Args:
        gamma (float): focusing parameter for modulating factor (1-p)
        alpha (float): balancing factor

    Returns:
        Function that computes focal loss given y_true and y_pred.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fixed

def create_label_tensors(labels):
    """
    Convert label dict items to float32 tensors.
    """
    return {key: tf.convert_to_tensor(val, dtype=tf.float32) for key, val in labels.items()}

def display_training_summary(history, label_columns):
    """
    Prints a well-formatted summary table (last 5 epochs) of accuracy and loss.
    """
    metrics = [k for k in history.history.keys() if not k.startswith("val_")]
    results = {}
    for metric in metrics:
        results[metric] = history.history[metric]
        val_metric = f"val_{metric}"
        if val_metric in history.history:
            results[val_metric] = history.history[val_metric]

    df = pd.DataFrame(results)
    df.index.name = "Epoch"

    print("\n📋 Final Training Summary (Last 5 Epochs):")
    try:
        print(df.tail(5).round(4).to_markdown())
    except Exception:
        print(df.tail(5).round(4).to_string())

def plot_training_metrics(history, label_columns, save_dir=None):
    for label in label_columns:
        acc_key = f"{label}_accuracy"
        val_acc_key = f"val_{label}_accuracy"
        print(f"Plotting accuracy for {label}: keys found?", acc_key in history.history, val_acc_key in history.history)
        print("Train accuracy data:", history.history.get(acc_key, []))
        print("Val accuracy data:", history.history.get(val_acc_key, []))

        plt.figure()
        if acc_key in history.history:
            plt.plot(history.history[acc_key], label=f"Train {label}")
        if val_acc_key in history.history:
            plt.plot(history.history[val_acc_key], label=f"Val {label}")
        plt.title(f"Accuracy for {label}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{label}_accuracy.png"))
            plt.close()
        else:
            plt.show()

        loss_key = f"{label}_loss"
        val_loss_key = f"val_{label}_loss"
        print(f"Plotting loss for {label}: keys found?", loss_key in history.history, val_loss_key in history.history)
        print("Train loss data:", history.history.get(loss_key, []))
        print("Val loss data:", history.history.get(val_loss_key, []))

        plt.figure()
        if loss_key in history.history:
            plt.plot(history.history[loss_key], label=f"Train {label}")
        if val_loss_key in history.history:
            plt.plot(history.history[val_loss_key], label=f"Val {label}")
        plt.title(f"Loss for {label}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{label}_loss.png"))
            plt.close()
        else:
            plt.show()
