# Homework 2: Image Classification with Transfer Learning

## Overview
This repository implements and compares three transfer learning strategies in PyTorch using a pre-trained **MobileNetV3-Small** (`MobileNet_V3_Small_Weights.DEFAULT`, ImageNet-1K) on the **Oxford-IIIT Pet** dataset (37 cat and dog breeds):
1. **Full Fine-Tuning**: All network parameters are updated.
2. **Classifier-Head Training**: Convolutional feature extractor is frozen; only the fully connected classifier layers (`classifier.0` and `classifier.3`) are trained.
3. **Feature Extraction + Traditional Classifiers**: Frozen feature vectors (576-dim after `avgpool`) are standardized using training statistics only and classified with `scikit-learn` (`LogisticRegression` and `LinearSVC`).

## Environment & Dependencies
Experiments were executed on Google Colab (NVIDIA Tesla T4 GPU) with:
* **Python**: 3.13.16
* **PyTorch**: 2.14.1+cu130
* **TorchVision**: 0.29.1+cu130
* **scikit-learn**: 1.6.1
* **NumPy**: 2.4.6
* Additional libraries: `pandas`, `matplotlib`

Install dependencies with:
```bash
pip install torch torchvision scikit-learn numpy pandas matplotlib
