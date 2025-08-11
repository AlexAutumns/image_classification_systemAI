# Backend/src/train_model.py

import os
import time
import tensorflow as tf
import pandas as pd
from model_builder import build_model
from data_loader import load_dataset
from utils import display_training_summary, plot_training_metrics
from focal_loss import SparseCategoricalFocalLoss

# ----------------------------------------
# Configuration
# ----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "Dataset")
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
CSV_PATH = os.path.join(DATASET_DIR, "styles.csv")

MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TRAINING_METRICS_DIR = os.path.join(MODELS_DIR, "training_metrics")
os.makedirs(TRAINING_METRICS_DIR, exist_ok=True)

BEST_MODEL_KERAS_PATH = os.path.join(MODELS_DIR, "best_clothing_classifier_model.keras")
BEST_MODEL_H5_PATH = os.path.join(MODELS_DIR, "best_clothing_classifier_model.h5")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "clothing_classifier_model_final.keras")
METRICS_CSV_PATH = os.path.join(MODELS_DIR, "training_metrics.csv")

IMAGE_SIZE = (128, 170)
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 10
LABEL_COLUMNS = ["masterCategory", "subCategory", "articleType"]

# ----------------------------------------
# Enable GPU memory growth
# ----------------------------------------
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"\n⚠️ Warning: Could not set memory growth: {e}")

# ----------------------------------------
# Load dataset
# ----------------------------------------
train_ds, val_ds, label_encodings, train_len, val_len = load_dataset(
    CSV_PATH,
    IMAGE_DIR,
    LABEL_COLUMNS,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    val_split=0.2,
)

# Build label size dict for model
label_sizes = {label: len(label_encodings[label][0]) for label in LABEL_COLUMNS}

# ----------------------------------------
# Load or build model (with .h5 to .keras conversion)
# ----------------------------------------
input_shape = (*IMAGE_SIZE, 3)

def load_or_convert_model():
    if os.path.exists(BEST_MODEL_KERAS_PATH):
        print(f"\n🔄 Loading model from {BEST_MODEL_KERAS_PATH}...")
        model = tf.keras.models.load_model(
            BEST_MODEL_KERAS_PATH,
            custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss}
        )
    elif os.path.exists(BEST_MODEL_H5_PATH):
        print(f"\n🔄 Found .h5 model at {BEST_MODEL_H5_PATH}, converting to .keras format...")
        model = tf.keras.models.load_model(
            BEST_MODEL_H5_PATH,
            custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss}
        )
        model.save(BEST_MODEL_KERAS_PATH)
        print(f"✅ Converted and saved to {BEST_MODEL_KERAS_PATH}")
    else:
        print("\n🚀 Building new model...")
        model = build_model(input_shape=input_shape, label_columns=label_sizes)
    return model

model = load_or_convert_model()
print("\n\nLoaded model input shape:", model.input_shape)
print("Current dataset image size:", (*IMAGE_SIZE, 3))


# ----------------------------------------
# Compile model
# ----------------------------------------
losses = {
    "articleType": SparseCategoricalFocalLoss(gamma=2.0, alpha=0.25),
    "masterCategory": SparseCategoricalFocalLoss(gamma=1.0, alpha=0.9),
    "subCategory": SparseCategoricalFocalLoss(gamma=1.0, alpha=0.9), 
}

loss_weights = {
    "articleType": 1.0,
    "masterCategory": 1.0,
    "subCategory": 1.0,
}

metrics = {label: "accuracy" for label in LABEL_COLUMNS}


from tensorflow.keras.optimizers import AdamW
model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
    loss=losses,
    loss_weights=loss_weights,
    metrics=metrics
)

model.summary()

# ----------------------------------------
# Callbacks
# ----------------------------------------
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=PATIENCE, restore_best_weights=True
)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_KERAS_PATH, save_best_only=True
)

# ----------------------------------------
# Train model
# ----------------------------------------
print("\n⏳ Training started...")
start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

end_time = time.time()
print(f"\n⏱️ Training completed in {end_time - start_time:.2f} seconds.")

# ----------------------------------------
# Save metrics & plots
# ----------------------------------------
display_training_summary(history, LABEL_COLUMNS)
plot_training_metrics(history, LABEL_COLUMNS, save_dir=TRAINING_METRICS_DIR)

pd.DataFrame(history.history).to_csv(METRICS_CSV_PATH, index_label="epoch")
print(f"\n📤 Metrics saved at {METRICS_CSV_PATH}")

# ----------------------------------------
# Save final model
# ----------------------------------------
model.save(FINAL_MODEL_PATH)
print(f"\n✅ Final model saved at {FINAL_MODEL_PATH}")
