"""
Backend for Fashion Image Classifier
- Classifies master, subcategory, and article type.
- Uses FastAPI + TensorFlow (Keras) models.
- Loads label lists from /labels/*.txt, optionally generates them from /labels/styles.csv.
- Enforces hierarchy consistency (master -> sub -> article) when picking predictions.
- Returns top-3 suggestions (overall and hierarchy-restricted) for each level.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from focal_loss import SparseCategoricalFocalLoss  # required for loading compiled models
from PIL import Image
import numpy as np
import io
import os
import traceback
import pandas as pd
import time
from typing import List, Dict, Tuple, Optional

app = FastAPI(title="Fashion Image Classifier API")

# ----------------------------- CORS -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set to your FE origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Paths -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
os.makedirs(LABELS_DIR, exist_ok=True)

# Label files (preferred) and optional CSV (for first-time generation)
STYLES_FILE = os.path.join(LABELS_DIR, "styles.csv")
MASTER_LABEL_FILE = os.path.join(LABELS_DIR, "masterLabels.txt")
SUB_LABEL_FILE = os.path.join(LABELS_DIR, "subLabels.txt")
ARTICLE_LABEL_FILE = os.path.join(LABELS_DIR, "articleLabels.txt")

# Models
MASTER_SUB_MODEL_PATH = os.path.join(MODELS_DIR, "best_master_sub_model.keras")
ARTICLE_MODEL_PATH = os.path.join(MODELS_DIR, "best_clothing_classifier_model.keras")

# ----------------------------- Image size -----------------------------
# Both models are trained for (H=128, W=170, C=3). Keep this consistent.
IMG_HEIGHT = 128
IMG_WIDTH = 170

# ----------------------------- Labels -----------------------------
def _read_nonempty_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def create_label_files_from_styles():
    """
    Create label txt files from styles.csv if they are missing.
    Skips bad lines to avoid CSV parser errors.
    """
    if not os.path.exists(STYLES_FILE):
        raise FileNotFoundError(
            "Label files missing and labels/styles.csv not found to generate them."
        )
    df = pd.read_csv(STYLES_FILE, on_bad_lines="skip")
    df = df.dropna(subset=["masterCategory", "subCategory", "articleType"])

    masters = sorted(df["masterCategory"].astype(str).unique().tolist())
    subs = sorted(df["subCategory"].astype(str).unique().tolist())
    articles = sorted(df["articleType"].astype(str).unique().tolist())

    with open(MASTER_LABEL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(masters))
    with open(SUB_LABEL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(subs))
    with open(ARTICLE_LABEL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(articles))

def load_labels() -> Tuple[List[str], List[str], List[str]]:
    """
    Load labels from files. If any file is missing, try to generate from styles.csv.
    """
    have_all = all(os.path.exists(p) for p in [MASTER_LABEL_FILE, SUB_LABEL_FILE, ARTICLE_LABEL_FILE])
    if not have_all:
        create_label_files_from_styles()

    masters = _read_nonempty_lines(MASTER_LABEL_FILE)
    subs = _read_nonempty_lines(SUB_LABEL_FILE)
    articles = _read_nonempty_lines(ARTICLE_LABEL_FILE)

    if not masters or not subs or not articles:
        raise RuntimeError("Label files exist but are empty. Please verify label files or styles.csv.")
    return masters, subs, articles

MASTER_CLASSES, SUB_CLASSES, ARTICLE_CLASSES = load_labels()

# ----------------------------- Hierarchy -----------------------------
def build_hierarchy() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Returns:
      master_to_sub: { masterLabel: [subLabels...] }
      sub_to_article: { subLabel: [articleLabels...] }
    If styles.csv is not present, returns empty dicts (no restriction).
    """
    if not os.path.exists(STYLES_FILE):
        return {}, {}

    df = pd.read_csv(STYLES_FILE, on_bad_lines="skip")
    df = df.dropna(subset=["masterCategory", "subCategory", "articleType"])

    master_to_sub = {}
    sub_to_article = {}

    for _, row in df.iterrows():
        m = str(row["masterCategory"])
        s = str(row["subCategory"])
        a = str(row["articleType"])
        master_to_sub.setdefault(m, set()).add(s)
        sub_to_article.setdefault(s, set()).add(a)

    # convert to lists for JSONability
    return (
        {k: sorted(list(v)) for k, v in master_to_sub.items()},
        {k: sorted(list(v)) for k, v in sub_to_article.items()},
    )

MASTER_TO_SUB, SUB_TO_ARTICLE = build_hierarchy()

# ----------------------------- Model loading -----------------------------
# We keep compile=False to avoid re-creating custom loss instances
# but we still pass the custom_objects in case Keras tries to deserialize.
master_sub_model = load_model(
    MASTER_SUB_MODEL_PATH,
    custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss},
    compile=False,
)
article_model = load_model(
    ARTICLE_MODEL_PATH,
    custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss},
    compile=False,
)

def check_model_label_compatibility() -> Dict[str, str]:
    """
    Logs sanity checks; returns a dict of warnings to also expose via API if desired.
    Compatible with two cases for master_sub_model:
      - Single tensor of length len(MASTER_CLASSES) + len(SUB_CLASSES)
      - Two-head output (list/tuple/dict) of shapes [len(MASTER), len(SUB)]
    """
    warnings = {}

    # Inspect output shapes by running a tiny dummy prediction
    dummy = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    try:
        ms_out = master_sub_model.predict(dummy, verbose=0)
        ar_out = article_model.predict(dummy, verbose=0)
    except Exception as e:
        warnings["prediction"] = f"Model warm-up prediction failed: {repr(e)}"
        return warnings

    # Flatten helper mirrors the one used in classify
    def _flat(out):
        if isinstance(out, (list, tuple)):
            return [np.array(x) for x in out]
        if isinstance(out, dict):
            return [np.array(v) for v in out.values()]
        return [np.array(out)]

    ms_parts = _flat(ms_out)
    ar_parts = _flat(ar_out)

    # Master/Sub: try two-head first, else single concat
    ok_master_sub = False
    if len(ms_parts) == 2:
        m_dim = ms_parts[0].shape[-1]
        s_dim = ms_parts[1].shape[-1]
        if m_dim != len(MASTER_CLASSES):
            warnings["master_labels"] = f"Master head size {m_dim} != len(MASTER_CLASSES) {len(MASTER_CLASSES)}"
        if s_dim != len(SUB_CLASSES):
            warnings["sub_labels"] = f"Sub head size {s_dim} != len(SUB_CLASSES) {len(SUB_CLASSES)}"
        ok_master_sub = True
    else:
        # assume single tensor concat
        total = ms_parts[0].shape[-1]
        expected = len(MASTER_CLASSES) + len(SUB_CLASSES)
        if total != expected:
            warnings["master_sub_labels"] = f"Master-Sub output {total} != expected {expected}"
        ok_master_sub = True  # still usable; we handle at runtime

    # Article
    if len(ar_parts) != 1:
        warnings["article_heads"] = "Article model returned multiple outputs; expected single head."
    else:
        a_dim = ar_parts[0].shape[-1]
        if a_dim != len(ARTICLE_CLASSES):
            warnings["article_labels"] = f"Article head size {a_dim} != len(ARTICLE_CLASSES) {len(ARTICLE_CLASSES)}"

    # Print warnings so you see them server-side
    if warnings:
        print("[Model/Label Sanity Warnings]")
        for k, v in warnings.items():
            print(f" - {k}: {v}")

    return warnings

SANITY_WARNINGS = check_model_label_compatibility()

# ----------------------------- Image helpers -----------------------------
def preprocess_image(file_bytes: bytes, target_size=(IMG_HEIGHT, IMG_WIDTH)) -> np.ndarray:
    """
    Convert uploaded image bytes to normalized NumPy array for model input.
    target_size: (H, W)
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((target_size[1], target_size[0]))  # PIL expects (W, H)
    arr = (np.array(img).astype(np.float32) / 255.0)[None, ...]  # (1, H, W, 3)
    return arr

def flatten_master_sub_output(pred_outputs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accepts outputs from master_sub_model and returns (pred_master, pred_sub).
    Supports:
      - list/tuple of two heads [master_probs, sub_probs]
      - dict with 2 values
      - single tensor with concat [master_probs..., sub_probs...]
    """
    if isinstance(pred_outputs, (list, tuple)):
        if len(pred_outputs) != 2:
            raise ValueError("Expected 2 heads for master_sub_model but got different length.")
        pm = np.array(pred_outputs[0])[0]
        ps = np.array(pred_outputs[1])[0]
        return pm, ps
    elif isinstance(pred_outputs, dict):
        vals = list(pred_outputs.values())
        if len(vals) != 2:
            raise ValueError("Expected dict with 2 outputs for master_sub_model.")
        pm = np.array(vals[0])[0]
        ps = np.array(vals[1])[0]
        return pm, ps
    else:
        flat = np.array(pred_outputs)[0]
        num_master = len(MASTER_CLASSES)
        pm = flat[:num_master]
        ps = flat[num_master:]
        return pm, ps

def flatten_article_output(pred_outputs) -> np.ndarray:
    """Returns (num_classes,) for the article model."""
    if isinstance(pred_outputs, (list, tuple)):
        arr = np.array(pred_outputs[0])[0]
    elif isinstance(pred_outputs, dict):
        arr = np.array(list(pred_outputs.values())[0])[0]
    else:
        arr = np.array(pred_outputs)[0]
    return arr

def safe_argmax(pred: np.ndarray, valid_indices: Optional[List[int]] = None) -> int:
    """
    Returns the index of the maximum probability.
    If valid_indices is provided, chooses the max only among those indices.
    If valid set is empty/None, falls back to global argmax.
    """
    if pred is None or pred.size == 0 or np.isnan(pred).all():
        return 0
    if valid_indices:
        best_idx = max(valid_indices, key=lambda i: pred[i] if i < len(pred) else -np.inf)
        return int(best_idx)
    return int(np.argmax(pred))

def safe_conf(pred: np.ndarray, idx: int) -> float:
    try:
        return float(pred[idx]) if 0 <= idx < len(pred) else 0.0
    except Exception:
        return 0.0

def top_n(pred: np.ndarray, labels: List[str], n=3, valid_indices: Optional[List[int]] = None):
    """
    Returns top-n list of {label, confidence}.
    If valid_indices is provided, ranking is restricted to those indices.
    """
    if valid_indices:
        pairs = [(i, pred[i]) for i in valid_indices if 0 <= i < len(pred)]
    else:
        pairs = list(enumerate(pred))
    if not pairs:
        return []
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:max(1, n)]
    return [{"label": labels[i], "confidence": float(v)} for i, v in pairs]

# ----------------------------- API -----------------------------
@app.post("/api/classify")
async def classify_image(image: UploadFile = File(...)):
    """
    POST /api/classify
    Body: multipart/form-data with "image"
    Response JSON:
    {
      master: { predicted, confidence, top3, top3Restricted },
      sub:    { predicted, confidence, top3, top3Restricted },
      article:{ predicted, confidence, top3, top3Restricted },
      hierarchyValid: bool,
      meta: { modelInputSize: [H, W], labelCounts: {...}, timingsMs: {...}, warnings: {...} }
    }
    """
    try:
        t0 = time.perf_counter()
        img_bytes = await image.read()
        img_arr = preprocess_image(img_bytes)
        t_pre = (time.perf_counter() - t0) * 1000.0

        # --- Predict
        t1 = time.perf_counter()
        raw_ms = master_sub_model.predict(img_arr, verbose=0)
        raw_ar = article_model.predict(img_arr, verbose=0)
        t_pred = (time.perf_counter() - t1) * 1000.0

        # --- Flatten
        pred_master, pred_sub = flatten_master_sub_output(raw_ms)
        pred_article = flatten_article_output(raw_ar)

        # --- Master
        master_idx = safe_argmax(pred_master)
        master_label = MASTER_CLASSES[master_idx]
        master_conf = safe_conf(pred_master, master_idx)

        # --- Sub (restricted to master)
        allowed_sub_labels = set(MASTER_TO_SUB.get(master_label, []))
        valid_sub_indices = [i for i, s in enumerate(SUB_CLASSES) if s in allowed_sub_labels] or None
        sub_idx = safe_argmax(pred_sub, valid_indices=valid_sub_indices)
        sub_label = SUB_CLASSES[sub_idx]
        sub_conf = safe_conf(pred_sub, sub_idx)

        # --- Article (restricted to sub)
        allowed_article_labels = set(SUB_TO_ARTICLE.get(sub_label, []))
        valid_article_indices = [i for i, a in enumerate(ARTICLE_CLASSES) if a in allowed_article_labels] or None
        article_idx = safe_argmax(pred_article, valid_indices=valid_article_indices)
        article_label = ARTICLE_CLASSES[article_idx]
        article_conf = safe_conf(pred_article, article_idx)

        # --- Hierarchy validity (if we have a hierarchy at all)
        hierarchy_valid = True
        if MASTER_TO_SUB:
            hierarchy_valid &= (sub_label in MASTER_TO_SUB.get(master_label, []))
        if SUB_TO_ARTICLE:
            hierarchy_valid &= (article_label in SUB_TO_ARTICLE.get(sub_label, []))

        # --- Build response
        resp = {
            "master": {
                "predicted": master_label,
                "confidence": master_conf,
                "top3": top_n(pred_master, MASTER_CLASSES, n=3),
                "top3Restricted": top_n(pred_master, MASTER_CLASSES, n=3)  # master usually has no restriction
            },
            "sub": {
                "predicted": sub_label,
                "confidence": sub_conf,
                "top3": top_n(pred_sub, SUB_CLASSES, n=3),
                "top3Restricted": top_n(pred_sub, SUB_CLASSES, n=3, valid_indices=valid_sub_indices)
            },
            "article": {
                "predicted": article_label,
                "confidence": article_conf,
                "top3": top_n(pred_article, ARTICLE_CLASSES, n=3),
                "top3Restricted": top_n(pred_article, ARTICLE_CLASSES, n=3, valid_indices=valid_article_indices)
            },
            "hierarchyValid": hierarchy_valid,
            "meta": {
                "modelInputSize": [IMG_HEIGHT, IMG_WIDTH],
                "labelCounts": {
                    "master": len(MASTER_CLASSES),
                    "sub": len(SUB_CLASSES),
                    "article": len(ARTICLE_CLASSES),
                },
                "timingsMs": {
                    "preprocess": round(t_pre, 2),
                    "predict": round(t_pred, 2),
                    "total": round((time.perf_counter() - t0) * 1000.0, 2),
                },
                "warnings": SANITY_WARNINGS,  # model/label compatibility notes
            },
        }

        return resp

    except Exception as e:
        traceback.print_exc()
        return {"error": repr(e)}
