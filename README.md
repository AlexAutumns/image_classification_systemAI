# 👗 Fashion Image Classification AI

This project is a machine learning system that classifies fashion product images into their respective categories using Convolutional Neural Networks (CNNs).  
It uses a **dual-model approach** for maximum accuracy:
- **Article Type Model** — optimized for predicting the `articleType`.
- **Master & Sub Category Model** — optimized for predicting `masterCategory` and `subCategory`.

> 🧠 The models are trained on a real-world dataset and will be deployed as part of an upcoming web interface.

---

## 📦 Dataset

The training data is sourced from the [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset on Kaggle, which includes labeled fashion product images with metadata such as `articleType`, `masterCategory`, `subCategory`, `gender`, `baseColour`, and more.

- **Total images:** ~44,000  
- **Size:** ~500MB  
- **Labels used:**
  - `articleType` (e.g., T-shirts, Shoes, Dresses)
  - `masterCategory` (e.g., Apparel, Footwear, Accessories)
  - `subCategory` (e.g., Topwear, Sandals, Watches)

---

## 🗂️ Project Structure

- `venv/` — Python virtual environment (excluded from GitHub)  

- `Dataset/`  
  - `images/` — all images used for training/inference  
  - `styles.csv` — metadata and labels  

- `Backend/`  
  - `api/` — *(planned for backend endpoints)*  
  - `models/`  
    - `training_metrics/`  
      - `articleType_accuracy.png`  
      - `articleType_loss.png`  
      - `masterCategory_accuracy.png`  
      - `masterCategory_loss.png`  
      - `subCategory_accuracy.png`  
      - `subCategory_loss.png`  
    - `best_clothing_classifier.keras` — best checkpoint for articleType model  
    - `clothing_classifier_model_final.keras` — final articleType model after training  
    - `best_master_sub_model.keras` — best checkpoint for masterCategory & subCategory model  
    - `master_sub_model_final.keras` — final masterCategory & subCategory model  
    - `training_metrics.csv` — metrics for articleType model  
    - `training_metrics_master_sub.csv` — metrics for master/subCategory model  

  - `src/`  
    - `data_loader.py` — loads & prepares `tf.data` pipelines (augmentation, caching, batching)  
    - `model_builder.py` — builds CNN architectures (multi-output)  
    - `focal_loss.py` — custom `SparseCategoricalFocalLoss` implementation  
    - `train_model.py` — trains the **articleType** model (multi-output originally)  
    - `train_categories_model.py` — trains the **masterCategory + subCategory** model (category-focused)  
    - `train_model_noFocal.py` — alternative training without focal loss  
    - `utils.py` — plotting, summary helpers, metric saving  

  - `requirements.txt` — pip dependencies  

- `README.md` — project documentation  

---

## ⚙️ Technologies & Libraries

- Python 3.10+  
- pandas  
- tensorflow / tensorflowjs  
- matplotlib  
- fastapi  
- uvicorn  
- python-multipart  
- Pillow  
- scikit-learn  
- numpy  

> **Note:** GPU support is recommended for faster training.

---

## 🚀 Planned Frontend

A web frontend will:
- Allow users to upload fashion images
- Use the trained models via a backend API
- Display predictions for:
  - **Article Type** (from the articleType model)
  - **Master & Sub Category** (from the category model)
- Show prediction confidence scores

---

## 🧠 How Training & Inference Work

**Training:**
- `train_model.py` → trains the articleType-focused model.  
- `train_categories_model.py` → trains the master/subCategory-focused model using the same dataset.  
- Both save metrics and plots to `Backend/models/training_metrics/`.

**Inference (recommended flow):**
1. Load both models:  
   - `best_clothing_classifier.keras` for `articleType`  
   - `best_master_sub_model.keras` for `masterCategory` & `subCategory`
2. Preprocess the input image identically to training.
3. Run:
   - `articleType` prediction on the first model.
   - `masterCategory` & `subCategory` predictions on the second model.
4. Combine into a single structured JSON response.

> **TODO:** Implement `inference.py` in `Backend/src/` to automate this process.

---

## 📌 How to Run

1. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

2. **Install dependencies:**
    ```bash
    pip install -r Backend/requirements.txt
    ```

3. **Place the dataset in the correct folder:**
    ```
    ./Dataset
    ```

4. **Train models:**
    ```bash
    # Train articleType model
    python Backend/src/train_model.py

    # Train category model
    python Backend/src/train_categories_model.py
    ```

---

## 📈 Improving Category Accuracy

- Stronger data augmentation (rotation, zoom, color jitter).  
- Transfer learning with efficient backbones (MobileNetV2, EfficientNet-lite).  
- Class balancing via `loss_weights` and tuned focal loss `gamma` / `alpha`.  
- Higher input resolution for fine-grained classes.  
- Optimizer tuning & LR scheduling (AdamW, cosine decay).  
- Analyze confusion matrices for class-specific weaknesses.

---

## 📤 Output

- Training graphs → `Backend/models/training_metrics/`  
- Saved models → `Backend/models/`:
  - `best_clothing_classifier.keras`  
  - `clothing_classifier_model_final.keras`  
  - `best_master_sub_model.keras`  
  - `master_sub_model_final.keras`  

---

## 📌 To Do

- [x] Train articleType model  
- [x] Train category-focused model (masterCategory + subCategory)  
- [ ] **Implement `inference.py` to load both models and merge predictions**  
- [ ] Create FastAPI backend for real-time predictions  
- [ ] Build React frontend to display results  

---

## 📜 License

This project is for educational and experimental use.  
Dataset: Publicly available via Kaggle, subject to its license terms.
