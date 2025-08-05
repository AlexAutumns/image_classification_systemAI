import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import matplotlib.pyplot as plt

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# ----------------------------------------
# Configuration
# ----------------------------------------

DATA_DIR = "../Dataset"  # relative path to outside Backend folder
IMAGES_TRAIN_DIR = os.path.join(DATA_DIR, "images")
CSV_TRAIN_PATH = os.path.join(DATA_DIR, "styles.csv")

IMAGE_SIZE = (60, 80)
BATCH_SIZE = 16  # smaller batch to reduce resource use
EPOCHS = 300

# Multi-output label columns
LABEL_COLUMNS = ["masterCategory", "subCategory", "articleType"]

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
    df.dropna(subset=["id"] + LABEL_COLUMNS, inplace=True)
    df["id"] = df["id"].astype(str)
    return df

def filter_existing_images(df, images_dir):
    available_image_ids = {f.split(".")[0] for f in os.listdir(images_dir) if f.endswith(".jpg")}
    df_filtered = df[df["id"].isin(available_image_ids)].copy()
    df_filtered["image_path"] = df_filtered["id"].apply(lambda i: os.path.join(images_dir, f"{i}.jpg"))
    return df_filtered

# ----------------------------------------
# Load dataset
# ----------------------------------------

df = load_styles_csv(CSV_TRAIN_PATH)
df = filter_existing_images(df, IMAGES_TRAIN_DIR)

# ----------------------------------------
# Encode labels for each category
# ----------------------------------------

label_encoders = {}
for label_col in LABEL_COLUMNS:
    unique_labels = df[label_col].unique()
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    df[f"{label_col}_index"] = df[label_col].map(label_to_index)
    label_encoders[label_col] = (label_to_index, unique_labels)

print(f"✅ Total samples: {len(df)}")
for label_col in LABEL_COLUMNS:
    print(f"✅ Classes in {label_col}: {len(label_encoders[label_col][0])}")

# ----------------------------------------
# Split into train and validation
# ----------------------------------------

df = df.sample(frac=1).reset_index(drop=True)  # shuffle

split_index = int(0.8 * len(df))
df_train = df.iloc[:split_index]
df_val = df.iloc[split_index:]

# ----------------------------------------
# Prepare datasets with multi-output labels
# ----------------------------------------

def process_image_and_labels(image_path, *labels):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0  # normalize pixels
    # Return image and tuple of labels
    return image, tuple(labels)

train_labels = tuple(df_train[f"{col}_index"].values for col in LABEL_COLUMNS)
val_labels = tuple(df_val[f"{col}_index"].values for col in LABEL_COLUMNS)

train_dataset = tf.data.Dataset.from_tensor_slices((df_train["image_path"].values, *train_labels))
train_dataset = train_dataset.map(process_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((df_val["image_path"].values, *val_labels))
val_dataset = val_dataset.map(process_image_and_labels, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ----------------------------------------
# Build CNN model with multi-output heads
# ----------------------------------------

weight_decay = 1e-4
inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
x = layers.Conv2D(32, 3, activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(x)
x = layers.MaxPooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(weight_decay))(x)
x = layers.Dropout(0.5)(x)

# Separate output layers for each label
outputs = []
for label_col in LABEL_COLUMNS:
    num_classes = len(label_encoders[label_col][0])
    output = layers.Dense(num_classes, activation="softmax", name=label_col)(x)
    outputs.append(output)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="adam",
    loss={label: "sparse_categorical_crossentropy" for label in LABEL_COLUMNS},
    metrics={label: "accuracy" for label in LABEL_COLUMNS}
)

model.summary()

# ----------------------------------------
# Callbacks
# ----------------------------------------

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    callbacks.ModelCheckpoint('models/best_clothing_classifier_model.h5', save_best_only=True)
]

# ----------------------------------------
# Train model
# ----------------------------------------

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks_list
)

# ----------------------------------------
# Plot accuracy curves for each output
# ----------------------------------------

for label_col in LABEL_COLUMNS:
    plt.plot(history.history[f"{label_col}_accuracy"], label=f"Train {label_col}")
    plt.plot(history.history[f"val_{label_col}_accuracy"], label=f"Val {label_col}")

plt.title("Training and Validation Accuracy for Multi-Output Classification")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------
# Save final model
# ----------------------------------------

model.save("models/clothing_classifier_model_final.h5")
print(f"✅ Final model saved at 'models/'")
