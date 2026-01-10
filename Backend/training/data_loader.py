import os
import hashlib
import pandas as pd
import tensorflow as tf
import tempfile


def _stable_cache_path(prefix: str, csv_path: str, images_dir: str, label_columns, image_size, batch_size, val_split, test_split) -> str:
    """
    Creates a stable cache filename so repeated runs reuse cache safely
    for the same dataset config.
    """
    key = "|".join([
        os.path.abspath(csv_path),
        os.path.abspath(images_dir),
        ",".join(label_columns),
        f"{image_size[0]}x{image_size[1]}",
        str(batch_size),
        str(val_split),
        str(test_split),
    ])
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"{prefix}_cache_{digest}.tf-data")


def load_dataset(
    csv_path,
    images_dir,
    label_columns,
    image_size=(60, 80),
    batch_size=32,
    val_split=0.2,
    test_split=0.1,
    seed=42,
):
    """
    Loads and prepares tf.data.Dataset for (possibly multi-output) classification.

    Returns:
        train_ds: tf.data.Dataset for training with augmentation.
        val_ds: tf.data.Dataset for validation without augmentation.
        test_ds: tf.data.Dataset for testing without augmentation.
        label_encodings: dict {label: (label_to_index, unique_labels)}
        train_len: number of training samples
        val_len: number of validation samples
        test_len: number of testing samples
    """
    if val_split < 0 or test_split < 0 or (val_split + test_split) >= 1.0:
        raise ValueError("val_split and test_split must be >= 0 and val_split + test_split must be < 1.0")

    # Load CSV and clean data
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    df.dropna(subset=["id"] + label_columns, inplace=True)
    df["id"] = df["id"].astype(str)

    # Keep only rows with existing images
    available_ids = {f.split(".")[0] for f in os.listdir(images_dir) if f.lower().endswith(".jpg")}
    df = df[df["id"].isin(available_ids)].copy()
    df["image_path"] = df["id"].apply(lambda i: os.path.join(images_dir, f"{i}.jpg"))

    # Encode labels
    label_encodings = {}
    for label in label_columns:
        unique_vals = sorted(df[label].unique())
        label_to_index = {val: idx for idx, val in enumerate(unique_vals)}
        df[f"{label}_index"] = df[label].map(label_to_index)
        label_encodings[label] = (label_to_index, unique_vals)

    # Shuffle once, deterministically
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split: train / val / test
    n = len(df)
    val_size = int(n * val_split)
    test_size = int(n * test_split)
    train_size = n - val_size - test_size

    if train_size <= 0:
        raise ValueError(
            f"Not enough samples after split. n={n}, train_size={train_size}, val_size={val_size}, test_size={test_size}"
        )

    df_train = df.iloc[:train_size]
    df_val = df.iloc[train_size : train_size + val_size]
    df_test = df.iloc[train_size + val_size :]

    # Image loader
    def process_path_and_labels(image_path, *labels):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = img / 255.0
        return img, tuple(labels)

    # Lightweight augmentation
    def augment(image, labels):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        return image, labels

    # Threading options to save CPU
    options = tf.data.Options()
    options.experimental_threading.private_threadpool_size = 4
    options.experimental_threading.max_intra_op_parallelism = 1

    # ---------- TRAIN ----------
    train_labels = tuple(df_train[f"{col}_index"].values for col in label_columns)
    train_ds = tf.data.Dataset.from_tensor_slices((df_train["image_path"].values, *train_labels))
    train_ds = train_ds.map(process_path_and_labels, num_parallel_calls=2)
    train_ds = train_ds.map(augment, num_parallel_calls=2)

    train_cache_path = _stable_cache_path("train", csv_path, images_dir, label_columns, image_size, batch_size, val_split, test_split)
    train_ds = train_ds.cache(train_cache_path)
    train_ds = train_ds.shuffle(1000, seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(1)
    train_ds = train_ds.with_options(options)

    # ---------- VAL ----------
    val_labels = tuple(df_val[f"{col}_index"].values for col in label_columns)
    val_ds = tf.data.Dataset.from_tensor_slices((df_val["image_path"].values, *val_labels))
    val_ds = val_ds.map(process_path_and_labels, num_parallel_calls=2)

    val_cache_path = _stable_cache_path("val", csv_path, images_dir, label_columns, image_size, batch_size, val_split, test_split)
    val_ds = val_ds.cache(val_cache_path)
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(1)
    val_ds = val_ds.with_options(options)

    # ---------- TEST ----------
    test_labels = tuple(df_test[f"{col}_index"].values for col in label_columns)
    test_ds = tf.data.Dataset.from_tensor_slices((df_test["image_path"].values, *test_labels))
    test_ds = test_ds.map(process_path_and_labels, num_parallel_calls=2)

    test_cache_path = _stable_cache_path("test", csv_path, images_dir, label_columns, image_size, batch_size, val_split, test_split)
    test_ds = test_ds.cache(test_cache_path)
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(1)
    test_ds = test_ds.with_options(options)

    return train_ds, val_ds, test_ds, label_encodings, len(df_train), len(df_val), len(df_test)


def tuple_to_dict_dataset(dataset, label_columns):
    """
    Converts a tf.data.Dataset yielding (x, y_tuple) to (x, y_dict)
    where y_dict maps label names to their respective elements in the tuple.
    """
    def map_fn(x, y):
        y_dict = {label_columns[i]: y[i] for i in range(len(label_columns))}
        return x, y_dict

    return dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
