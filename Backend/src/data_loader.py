import os
import pandas as pd
import tensorflow as tf
import tempfile

def load_dataset(csv_path, images_dir, label_columns, image_size=(60, 80), batch_size=32, val_split=0.2):
    """
    Loads and prepares tf.data.Dataset for multi-output classification.

    Args:
        csv_path (str): Path to CSV file with labels.
        images_dir (str): Directory with images.
        label_columns (list): List of label column names.
        image_size (tuple): Target image size (height, width).
        batch_size (int): Batch size.
        val_split (float): Fraction of dataset to use as validation.

    Returns:
        train_ds: tf.data.Dataset for training with augmentation.
        val_ds: tf.data.Dataset for validation without augmentation.
        label_encodings: dict {label: (label_to_index, unique_labels)}
        train_len: number of training samples
        val_len: number of validation samples
    """

    # Load CSV and clean data
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df.dropna(subset=['id'] + label_columns, inplace=True)
    df['id'] = df['id'].astype(str)

    # Keep only rows with existing images
    available_ids = {f.split('.')[0] for f in os.listdir(images_dir) if f.lower().endswith('.jpg')}
    df = df[df['id'].isin(available_ids)].copy()
    df['image_path'] = df['id'].apply(lambda i: os.path.join(images_dir, f"{i}.jpg"))

    # Encode labels
    label_encodings = {}
    for label in label_columns:
        unique_vals = sorted(df[label].unique())
        label_to_index = {val: idx for idx, val in enumerate(unique_vals)}
        df[f"{label}_index"] = df[label].map(label_to_index)
        label_encodings[label] = (label_to_index, unique_vals)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Train / val split
    split_idx = int(len(df) * (1 - val_split))
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    # Function to load and preprocess images
    def process_path_and_labels(image_path, *labels):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = img / 255.0
        return img, tuple(labels)

    # Lightweight augmentation function
    def augment(image, labels):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        return image, labels

    train_cache_path = os.path.join(tempfile.gettempdir(), "train_cache.tf-data")
    val_cache_path = os.path.join(tempfile.gettempdir(), "val_cache.tf-data")

    train_labels = tuple(df_train[f"{col}_index"].values for col in label_columns)
    val_labels = tuple(df_val[f"{col}_index"].values for col in label_columns)

    # Training dataset pipeline
    train_ds = tf.data.Dataset.from_tensor_slices((df_train["image_path"].values, *train_labels))
    train_ds = train_ds.map(process_path_and_labels, num_parallel_calls=2)
    train_ds = train_ds.map(augment, num_parallel_calls=2)
    train_ds = train_ds.cache(train_cache_path)
    train_ds = train_ds.shuffle(1000)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(1)

    # Reduce threading to save CPU
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 4
    options.experimental_threading.max_intra_op_parallelism = 1
    train_ds = train_ds.with_options(options)

    # Validation dataset pipeline
    val_ds = tf.data.Dataset.from_tensor_slices((df_val["image_path"].values, *val_labels))
    val_ds = val_ds.map(process_path_and_labels, num_parallel_calls=2)
    val_ds = val_ds.cache(val_cache_path)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(1)
    val_ds = val_ds.with_options(options)

    return train_ds, val_ds, label_encodings, len(df_train), len(df_val)


def tuple_to_dict_dataset(dataset, label_columns):
    """
    Converts a tf.data.Dataset yielding (x, y_tuple) to (x, y_dict)
    where y_dict maps label names to their respective elements in the tuple.

    Args:
        dataset: tf.data.Dataset yielding (inputs, tuple of label tensors)
        label_columns: list of label names in order matching tuple

    Returns:
        tf.data.Dataset yielding (inputs, dict of label tensors)
    """
    def map_fn(x, y):
        y_dict = {label_columns[i]: y[i] for i in range(len(label_columns))}
        return x, y_dict

    return dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
