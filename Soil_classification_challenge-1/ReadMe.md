
# 🌱 Soil Type Classification Using EfficientNet-B3

## 📘 Project Overview
This project focuses on classifying soil types from images using a transfer learning approach based on the EfficientNet-B3 architecture. The dataset comprises labeled images of various Indian soil types, and the goal is to accurately distinguish between these categories using deep learning.

---

## 📂 Directory Structure
```
├── data/
│   └── download.py          # Script to download and unzip dataset from Kaggle
├── docs/
│   └── architecture.png     # Diagram of the model architecture
│   └── cards/
│       └── ml-metrics.json  # Evaluation metrics
│       └── project-card.ipynb
├── model.ipynb              # Unified notebook for training and inference
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## 🧠 Model Architecture
The model uses a pre-trained EfficientNet-B3 backbone from the `timm` library. Only the final two blocks are fine-tuned. The default classifier head is replaced with a custom sequence:

```python
self.backbone.classifier = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)
```

This setup improves performance while retaining the powerful low-level feature representations learned from ImageNet.

---

## 🔄 Model Pipeline (Step-by-Step)
1. **Data Loading**:
   - Dataset is read from CSV files containing image file paths and corresponding soil type labels.
   - Images are loaded using `PIL` and transformed using `torchvision.transforms` for resizing, normalization, and tensor conversion.

2. **Dataset Splitting**:
   - The data is split into training and validation sets using stratified sampling to preserve label distribution.

3. **Data Augmentation & Preprocessing**:
   - Training images are augmented with random horizontal flips and rotations.
   - All images are resized to a fixed shape and normalized to match ImageNet statistics.

4. **Model Preparation**:
   - EfficientNet-B3 is loaded with pretrained ImageNet weights.
   - The final classification head is replaced with a custom MLP (Linear → ReLU → Dropout → Linear).
   - Only the last two blocks of EfficientNet are unfrozen for fine-tuning.

5. **Training**:
   - Model is trained using cross-entropy loss.
   - Optimizer: Adam with weight decay
   - Learning rate scheduler is applied (optional)
   - Validation metrics (accuracy, F1-score) are computed after each epoch.

6. **Evaluation**:
   - After training, the model is evaluated on the validation set.
   - Predictions are compared to ground-truth labels to generate a classification report and confusion matrix.

7. **Inference**:
   - The trained model is used to predict the class of new/unseen soil images by passing them through the same preprocessing and forward pass pipeline.

---

## 🧪 Dataset
- **Source**: Kaggle competition / India-specific soil images.
- **Labels**: Alluvial, Black, Clay, Red soil
- **Format**: Images organized by CSV files for training/testing

> Make sure to run `python data/download.py` to get the dataset and unzip it into the correct folder structure.

---

## 🚀 How to Run the Project
Launch the unified notebook for both training and inference:
```bash
jupyter notebook model.ipynb
```

Follow the cells in order: from dataset preparation to training and evaluation.

---

## 📊 Evaluation Metrics
Results are saved in `docs/cards/ml-metrics.json` and include:
```json
{
  "accuracy": 0.92,
  "precision": 0.91,
  "recall": 0.90,
  "f1_score": 0.905
}
```

---

## 🛠 Dependencies
Install all dependencies via:
```bash
pip install -r requirements.txt
```

Includes:
- numpy, pandas, matplotlib
- scikit-learn
- timm, torch, torchvision
- PIL, seaborn

---

