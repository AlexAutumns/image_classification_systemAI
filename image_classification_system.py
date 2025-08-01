import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt

# ----------------------------------------
# Configuration
# ----------------------------------------

DATA_DIR = "data"
IMAGES_TRAIN_DIR = os.path.join(DATA_DIR, "myntradataset", "images")
CSV_TRAIN_PATH = os.path.join(DATA_DIR, "myntradataset", "styles.csv")

# Uncomment below lines to use original dataset later
# IMAGES_VAL_DIR = os.path.join(DATA_DIR, "images")
# CSV_VAL_PATH = os.path.join(DATA_DIR, "styles.csv")

IMAGE_SIZE = (60, 80)
BATCH_SIZE = 16  # smaller batch to reduce resource use
EPOCHS = 50
LABEL_COLUMN = "articleType"  # change if needed

# ----------------------------------------
# Enable GPU memory growth (optional but recommended)
# ----------------------------------------

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"Warning: Could not set memory growth: {e}")

# ----------------------------------------
# Helper functions to load and filter data
# ----------------------------------------

def load_styles_csv(csv_path):
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    df.dropna(subset=["id", LABEL_COLUMN], inplace=True)
    df["id"] = df["id"].astype(str)
    return df

def filter_existing_images(df, images_dir):
    available_image_ids = {f.split(".")[0] for f in os.listdir(images_dir) if f.endswith(".jpg")}
    df_filtered = df[df["id"].isin(available_image_ids)].copy()
    df_filtered["image_path"] = df_filtered["id"].apply(lambda i: os.path.join(images_dir, f"{i}.jpg"))
    return df_filtered

# ----------------------------------------
# Load Myntra dataset for both training and validation (split later)
# ----------------------------------------

df = load_styles_csv(CSV_TRAIN_PATH)
df = filter_existing_images(df, IMAGES_TRAIN_DIR)

# ----------------------------------------
# Encode labels based on the Myntra dataset labels
# ----------------------------------------

unique_labels = df[LABEL_COLUMN].unique()
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
df["label_index"] = df[LABEL_COLUMN].map(label_to_index)

print(f"✅ Total samples (Myntra): {len(df)}")
print(f"✅ Number of classes: {len(unique_labels)}")

# ----------------------------------------
# Split Myntra dataset into training and validation sets (80% train, 20% val)
# ----------------------------------------

df = df.sample(frac=1).reset_index(drop=True)  # shuffle dataset

split_index = int(0.8 * len(df))
df_train = df.iloc[:split_index]
df_val = df.iloc[split_index:]

# ----------------------------------------
# Prepare TensorFlow datasets
# ----------------------------------------

def process_image(image_path, label_index):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0  # normalize pixels
    return image, label_index

train_dataset = tf.data.Dataset.from_tensor_slices(
    (df_train["image_path"].values, df_train["label_index"].values)
)
train_dataset = train_dataset.map(process_image, num_parallel_calls=2)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (df_val["image_path"].values, df_val["label_index"].values)
)
val_dataset = val_dataset.map(process_image, num_parallel_calls=2)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ----------------------------------------
# Build CNN model with L2 regularization
# ----------------------------------------

weight_decay = 1e-4

model = models.Sequential([
    layers.Input(shape=(*IMAGE_SIZE, 3)),
    layers.Conv2D(32, 3, activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
    layers.Dropout(0.5),
    layers.Dense(len(unique_labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------------------
# Setup callbacks: EarlyStopping and ModelCheckpoint
# ----------------------------------------

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint('best_clothing_classifier_model.h5', save_best_only=True)
]

# ----------------------------------------
# Train model with callbacks
# ----------------------------------------

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks_list
)

# ----------------------------------------
# Plot accuracy curves
# ----------------------------------------

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------
# Save final model to disk
# ----------------------------------------

model.save("clothing_classifier_model_final.h5")
print("✅ Final model saved as clothing_classifier_model_final.h5")
