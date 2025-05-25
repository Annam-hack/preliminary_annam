# 🌱 Soil Type Classification via Positive-Unlabeled Learning with EfficientNet-B0

## 📘 Project Overview

This project tackles the unique challenge of soil type classification when only positive samples are labeled. Traditional supervised learning isn't directly applicable here, so we employ a **Positive-Unlabeled (PU) Learning** strategy. Our final model combines **EfficientNet-B0** for feature extraction with a custom classifier head. To simulate negative examples, we use publicly available datasets such as **CIFAR-10** and **MNIST**.

---

## 📂 Directory Structure

```
├── data/
│   └── download.py          # Script to download and unzip dataset from Kaggle
├── docs/
│   └── arch.png     # Diagram of the model architecture
│   └── cards/
│       └── ml-metrics.json  # Evaluation metrics
├── notebooks/
│   └── soil-classification-task2.ipynb              # Unified notebook for training and inference
├── src/
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## 🧠 Model Architecture

We utilize a pre-trained **EfficientNet-B0** model from `torchvision.models` as the backbone. The classification head is modified as follows:

```python
self.model.classifier = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1),
    nn.Sigmoid()
)
```

* Only the classifier layers are trained, keeping the backbone frozen to retain powerful ImageNet features.
* Output is a binary label: 1 for soil images, 0 for negative (non-soil) examples.

---

## 🔄 Model Pipeline

### 🧾 Data Preparation

* **Positive Class**: Images of soil types from a custom dataset (`/soil_competition-2025/train/`).
* **Negative Class**: Equal-sized subsets from CIFAR-10 and MNIST (converted to RGB).
* All data is resized to 224×224 and normalized to ImageNet standards.

### 🔁 Dataset Composition

Positive and negative samples are balanced and combined into a single binary classification task.

### 🧠 Training

* **Loss**: Binary Cross-Entropy Loss (BCE)
* **Optimizer**: Adam
* **Epochs**: 5
* **Batch Size**: 32

### 🧪 Evaluation

* Threshold tuning is done to maximize F1-score.
* Evaluation metrics include:

  * Accuracy
  * Precision
  * Recall
  * F1 Score

All predictions are evaluated using a validation set split from the full dataset.

---

## 📊 Evaluation Metrics

Metrics are stored in `docs/ml-metrics.json`:

```json
{
    "accuracy": 0.9992615721310264,
    "precision": 0.9927109176826554,
    "recall": 0.9942716857610474,
    "f1_score": 0.9902772354499123
}
```

---

## 🚧 Challenges Faced

* **Label Scarcity**: Only positive soil images were labeled; no direct negative samples.
* **Initial Approaches Failed**:

  * Tried PU learning with autoencoders and Deep SVDD
  * Used multi-patch core modeling
  * These approaches yielded poor generalization and low F1 scores

**Solution**: Synthesized negative class from CIFAR and MNIST, which led to a much more stable and accurate classifier.

---

## 🚀 How to Run

1. Ensure all data is in the correct directory (`data/soil_competition-2025/train/`).
2. Run the notebook:

   ```bash
   jupyter notebook soil_classification.ipynb
   ```
3. View metrics in `docs/ml-metrics.json`.

---

## 🛠 Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

### requirements.txt

```txt
numpy
pandas
matplotlib
scikit-learn
torch
torchvision
tensorflow
Pillow
```

---

## 📚 Acknowledgements

* Soil image dataset (custom)
* Negative samples sourced from [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [MNIST](http://yann.lecun.com/exdb/mnist/)
* Pre-trained EfficientNet via `torchvision`

---

Let me know if you’d like a visual architecture diagram or presentation slides next!

