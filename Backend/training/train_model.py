# Backend/src/train_model.py

import os
import time
import json
import tensorflow as tf
import pandas as pd

from model_builder import build_model
from data_loader import load_dataset, tuple_to_dict_dataset
from utils import display_training_summary, plot_training_metrics
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.utils import plot_model

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
TEST_RESULTS_JSON_PATH = os.path.join(MODELS_DIR, "test_results_all_heads.json")

IMAGE_SIZE = (128, 170)
BATCH_SIZE = 128
EPOCHS = 10
PATIENCE = 10

# We keep ALL heads, but we will prioritize articleType in loss weights + monitoring.
LABEL_COLUMNS = ["masterCategory", "subCategory", "articleType"]

# Splits (CW2 expects a real TEST evaluation)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

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
# Load dataset (train/val/test)
# ----------------------------------------
train_ds, val_ds, test_ds, label_encodings, train_len, val_len, test_len = load_dataset(
    CSV_PATH,
    IMAGE_DIR,
    LABEL_COLUMNS,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    val_split=VAL_SPLIT,
    test_split=TEST_SPLIT,
)

# Convert (x, y_tuple) -> (x, y_dict) for stable multi-output training
train_ds = tuple_to_dict_dataset(train_ds, LABEL_COLUMNS)
val_ds = tuple_to_dict_dataset(val_ds, LABEL_COLUMNS)
test_ds = tuple_to_dict_dataset(test_ds, LABEL_COLUMNS)

print(f"\n📊 Split sizes: train={train_len}, val={val_len}, test={test_len}")

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
            custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss},
        )
    elif os.path.exists(BEST_MODEL_H5_PATH):
        print(f"\n🔄 Found .h5 model at {BEST_MODEL_H5_PATH}, converting to .keras format...")
        model = tf.keras.models.load_model(
            BEST_MODEL_H5_PATH,
            custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss},
        )
        model.save(BEST_MODEL_KERAS_PATH)
        print(f"✅ Converted and saved to {BEST_MODEL_KERAS_PATH}")
    else:
        print("\n🚀 Building new model...")
        model = build_model(input_shape=input_shape, label_columns=label_sizes)
    return model

model = load_or_convert_model()

print("\nLoaded model input shape:", model.input_shape)
print("Current dataset image size:", (*IMAGE_SIZE, 3))

# ----------------------------------------
# Compile model
# ----------------------------------------
losses = {
    "articleType": SparseCategoricalFocalLoss(gamma=2.0, alpha=0.25),
    "masterCategory": SparseCategoricalFocalLoss(gamma=1.0, alpha=0.9),
    "subCategory": SparseCategoricalFocalLoss(gamma=1.0, alpha=0.9),
}

# Focus articleType more (still trains all heads)
loss_weights = {
    "articleType": 2.0,
    "masterCategory": 0.5,
    "subCategory": 0.5,
}

metrics = {label: "accuracy" for label in LABEL_COLUMNS}

from tensorflow.keras.optimizers import AdamW
model.compile(
    optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
    loss=losses,
    loss_weights=loss_weights,
    metrics=metrics,
)

model.summary()

plot_model(model, to_file=os.path.join(MODELS_DIR, "model_architecture.png"),
           show_shapes=True, show_layer_names=True, expand_nested=False)

# ----------------------------------------
# Callbacks (monitor articleType primarily)
# ----------------------------------------
# We want the "best model" for CW2 to be best on articleType.
monitor_metric = "val_articleType_accuracy"

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor=monitor_metric,
    mode="max",
    factor=0.5,
    patience=3,
    verbose=1,
    min_lr=1e-6,
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor=monitor_metric,
    mode="max",
    patience=PATIENCE,
    restore_best_weights=True,
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    BEST_MODEL_KERAS_PATH,
    monitor=monitor_metric,
    mode="max",
    save_best_only=True,
    verbose=1,
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
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
)

end_time = time.time()
print(f"\n⏱️ Training completed in {end_time - start_time:.2f} seconds.")

# ----------------------------------------
# Evaluate on TEST set (CW2 requirement)
# ----------------------------------------
print("\n🧪 Evaluating on TEST set...")
test_results = model.evaluate(test_ds, verbose=1, return_dict=True)

print("\n✅ TEST results:")
for k, v in test_results.items():
    print(f"  - {k}: {v}")

# Highlight the key metric you care about most
if "articleType_accuracy" in test_results:
    print(f"\n⭐ ArticleType TEST accuracy: {test_results['articleType_accuracy']:.4f}")

with open(TEST_RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "split_sizes": {"train": train_len, "val": val_len, "test": test_len},
            "test_results": test_results,
            "notes": {
                "monitor_metric": monitor_metric,
                "loss_weights": loss_weights,
                "val_split": VAL_SPLIT,
                "test_split": TEST_SPLIT,
            },
        },
        f,
        indent=2,
    )

print(f"\n📤 TEST results saved at {TEST_RESULTS_JSON_PATH}")

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
