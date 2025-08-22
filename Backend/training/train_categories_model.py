import os
import time
import tensorflow as tf
import pandas as pd
from model_builder import build_model
from data_loader import load_dataset, tuple_to_dict_dataset
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

BEST_MULTI_MODEL_PATH = os.path.join(MODELS_DIR, "best_master_sub_model.keras")
FINAL_MULTI_MODEL_PATH = os.path.join(MODELS_DIR, "master_sub_model_final.keras")
METRICS_CSV_PATH = os.path.join(MODELS_DIR, "training_metrics_master_sub.csv")

IMAGE_SIZE = (128, 170)
BATCH_SIZE = 128
EPOCHS = 20
PATIENCE = 5

MULTI_LABEL_COLUMNS = ["masterCategory", "subCategory"]

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
    MULTI_LABEL_COLUMNS + ["articleType"],  # still load articleType to keep dataset consistent, but will not use it in training
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    val_split=0.2,
)

# Convert tuple labels to dict labels for multi-output model
train_ds = tuple_to_dict_dataset(train_ds, MULTI_LABEL_COLUMNS)
val_ds = tuple_to_dict_dataset(val_ds, MULTI_LABEL_COLUMNS)

# Build label size dict for model (only masterCategory and subCategory)
label_sizes = {label: len(label_encodings[label][0]) for label in MULTI_LABEL_COLUMNS}

# ----------------------------------------
# Load or build model
# ----------------------------------------
input_shape = (*IMAGE_SIZE, 3)

def load_or_build_model():
    if os.path.exists(BEST_MULTI_MODEL_PATH):
        print(f"\n🔄 Loading existing masterCategory+subCategory model from {BEST_MULTI_MODEL_PATH}...")
        model = tf.keras.models.load_model(
            BEST_MULTI_MODEL_PATH,
            custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss}
        )
    else:
        print("\n🚀 Building new masterCategory+subCategory model...")
        model = build_model(input_shape=input_shape, label_columns=label_sizes)
    return model

multi_model = load_or_build_model()

# ----------------------------------------
# Compile model
# ----------------------------------------
losses = {
    "masterCategory": SparseCategoricalFocalLoss(gamma=1.0, alpha=0.9),
    "subCategory": SparseCategoricalFocalLoss(gamma=1.0, alpha=0.9),
}
loss_weights = {
    "masterCategory": 1.0,
    "subCategory": 1.0,
}
metrics = {label: "accuracy" for label in MULTI_LABEL_COLUMNS}

from tensorflow.keras.optimizers import AdamW
multi_model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
    loss=losses,
    loss_weights=loss_weights,
    metrics=metrics
)

multi_model.summary()

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
    BEST_MULTI_MODEL_PATH, save_best_only=True
)

# ----------------------------------------
# Train model
# ----------------------------------------
print("\n⏳ Training masterCategory+subCategory model...")
start_time = time.time()

history = multi_model.fit(
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
display_training_summary(history, MULTI_LABEL_COLUMNS)
plot_training_metrics(history, MULTI_LABEL_COLUMNS, save_dir=TRAINING_METRICS_DIR)

pd.DataFrame(history.history).to_csv(METRICS_CSV_PATH, index_label="epoch")
print(f"\n📤 Metrics saved at {METRICS_CSV_PATH}")

# ----------------------------------------
# Save final model
# ----------------------------------------
multi_model.save(FINAL_MULTI_MODEL_PATH)
print(f"\n✅ Final masterCategory+subCategory model saved at {FINAL_MULTI_MODEL_PATH}")
