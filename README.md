# 👗 Fashion Image Classification AI

This project is a machine learning system that classifies fashion product images into their respective categories using a Convolutional Neural Network (CNN). It aims to automate and improve product categorization, which is essential for e-commerce applications and inventory systems.

> 🧠 The model is trained on a real-world dataset and will be deployed as part of an upcoming React-based web interface.

## 📦 Dataset

The training data is sourced from the [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset on Kaggle, which includes labeled fashion product images with metadata such as `articleType`, `gender`, `baseColour`, and more.

- Total images: ~44,000
- Size: ~500MB
- Label used for classification: `articleType` (e.g., T-shirts, Shoes, Dresses, etc.)

## 🏗️ Project Structure
- `venv/`  
  Python virtual environment (excluded from GitHub)

- `data/`  
  Contains all dataset-related files  
  - `myntradataset/`  
    - `images/` — Training images  
    - `styles.csv` — Metadata file with labels like `articleType`

- `image_classification_system.py`  
  Script for training the CNN model

- `best_clothing_classifier_model.h5`  
  Best model checkpoint saved during training (based on validation accuracy)

- `clothing_classifier_model_final.h5`  
  Final trained model saved after all epochs

## ⚙️ Technologies & Libraries

- Python 3
- TensorFlow / Keras
- Pandas
- Matplotlib
- NumPy

> Optional: GPU support enabled for faster training

## 🚀 Planned Frontend

A **React-based web application** is currently in development, which will:
- Allow users to upload fashion images
- Use the trained `.h5` model for live classification via a backend API
- Display the predicted category and confidence

## 📌 How to Run

1. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Place the dataset in the correct folder (`./data/myntradataset/`)

4. Run the training script:
    ```bash
    python image_classification_system.py
    ```

## 📈 Output

- Accuracy and loss graphs will be displayed after training.
- Two models will be saved:
  - `best_clothing_classifier_model.h5` (best performing checkpoint)
  - `clothing_classifier_model_final.h5` (final model after all epochs)

## ✅ To Do

- [x] Model training and evaluation
- [ ] Create Flask/FastAPI backend for predictions
- [ ] Build React frontend
- [ ] Connect model to frontend via API
- [ ] Deploy full-stack app


## 📜 License

This project is for educational and experimental use. The dataset is publicly available via Kaggle and subject to its licensing terms.
