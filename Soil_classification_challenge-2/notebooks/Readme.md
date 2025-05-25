# 🌱 Soil Image Classification

This repository contains two distinct approaches to tackle the task:
**Classify an input image as either a *soil image* or *not a soil image***.

## 📝 Task Description

The objective of this task is to build a robust classifier that can distinguish between images of soil and those that are not soil. This is critical in domains such as agriculture, remote sensing, and environmental monitoring.

---

## 📁 Notebooks Overview

### 1. `PU_Learning_Ensemble.ipynb`

**Approach**: Positive-Unlabeled (PU) Learning using an ensemble of anomaly detection models
**Techniques Used**:

* **Deep SVDD** (Support Vector Data Description)
* **Autoencoders**
* **Multi-Patch Core** models

This approach treats *soil images* as the **positive class**, and the dataset includes unlabeled data which may contain both soil and non-soil images. By leveraging PU learning techniques, the model tries to learn a decision boundary for the positive class without explicit negative labels.

> ✅ Best suited for imbalanced or weakly labeled datasets, especially where negative samples are hard to collect.

---

### 2. `EfficientNet_Binary_Classification.ipynb`

**Approach**: Supervised binary classification
**Model Used**: EfficientNet (pretrained and fine-tuned)
**Data**:

* **Positive Class**: Soil images
* **Negative Class**: Non-soil images sampled from the **CIFAR** and **MNIST** datasets.

The EfficientNet model was trained directly on labeled positive and negative samples. This supervised setting allowed us to obtain a **higher F1 Score**, thanks to better separation in the learned feature space.

> 🚀 EfficientNet delivered **superior performance** in terms of precision and recall, making it a solid baseline for this task.

---

## 📊 Comparison Summary

| Method                                | Type            | F1 Score | Notes                                     |
| ------------------------------------- | --------------- | -------- | ----------------------------------------- |
| ❌PU Learning (Deep SVDD + AE + MPCore) | Unsupervised/PU | 0.29   | No true negatives needed                  |
| ✅EfficientNet Binary Classifier(Final Selected model)        | Supervised      | 0.99     | Requires labeled negatives (CIFAR, MNIST) |

---

## 📦 Dependencies

Make sure to install the required packages before running the notebooks:

```bash
pip install -r requirements.txt
```

(You may need libraries like `torch`, `torchvision`, `scikit-learn`, `efficientnet_pytorch`, and `matplotlib`.)

---

## 🚧 Future Improvements

* Explore other backbone models like ResNet50 or ConvNeXt.
* Use real-world non-soil images from satellite or drone datasets.
* Apply Grad-CAM or SHAP for interpretability of predictions.

---

## ✨ Acknowledgments

* CIFAR and MNIST datasets for negative samples.
* Open-source implementations of Deep SVDD and Autoencoder frameworks.

---

Let me know if you'd like to generate a fancy table of contents or add badges for Kaggle, Colab, etc.
