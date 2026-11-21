# Homework 2: Image Classification with Transfer Learning

## Overview
This repository implements and compares three transfer learning strategies in PyTorch using a pre-trained **MobileNetV3-Small** (`MobileNet_V3_Small_Weights.DEFAULT`, pre-trained on ImageNet-1K) on the **Oxford-IIIT Pet** dataset (37 cat and dog breeds):

1. **Full Fine-Tuning**: The final classification layer is replaced (37 outputs) and all network parameters (1,555,781) are trained.
2. **Classifier-Head Training**: The convolutional feature extractor (`model.features`) is frozen; only the fully connected classification layers (`classifier.0` and `classifier.3`, 628,773 parameters) are trained.
3. **Feature Extraction + Traditional Classifiers**: The pre-trained network is completely frozen to extract 576-dimensional feature vectors after `avgpool`. Features are standardized using `StandardScaler` fitted exclusively on the training set, then classified with `scikit-learn` (`LogisticRegression` and `LinearSVC`, 21,349 parameters).

## Environment & Dependencies
All experiments were executed on Google Colab with an **NVIDIA Tesla T4** GPU using the following versions:
* **Python**: `3.13.16`
* **PyTorch**: `2.14.1+cu130`
* **TorchVision**: `0.29.1+cu130`
* **scikit-learn**: `1.6.1`
* **NumPy**: `2.4.6`
* **Additional libraries**: `pandas`, `matplotlib`

Install the required packages via:
```bash
pip install torch torchvision scikit-learn numpy pandas matplotlib
```

## Dataset & Split Protocol
* **Dataset**: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (Parkhi et al., *Cats and Dogs*, CVPR 2012).
* **License**: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
* **Download**: Automatically handled by `torchvision.datasets.OxfordIIITPet(..., download=True)`.
* **Splits**:
  * Official `trainval` (3,680 images) is split 80/20 stratified by breed into **Train (2,944 images)** and **Validation (736 images)** using `seed=42`.
  * Official `test` (**3,669 images**) is strictly held out and evaluated only once per selected model.
  * Exact train/validation split indices are saved in `results/split_indices.json`.

## Reproducibility
Random seeds are fixed to `42` for `random`, `numpy`, and `torch` (CPU and CUDA), with `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`. Minor residual non-determinism may still occur in certain GPU operations.

Model selection (best epoch checkpoint for neural models; regularization parameter `C` for `LogisticRegression` and `LinearSVC`) is performed strictly on the validation set.

## How to Run
1. **Google Colab (Recommended)**: Open `HW2_ImageClassification.ipynb` in Google Colab, enable a GPU runtime (`Runtime -> Change runtime type -> T4 GPU`), and run all cells sequentially.
2. **Local Python Execution**: Convert the notebook or run the script directly:
```bash
python HW2_ImageClassification.py
```

Running the pipeline reproduces all training runs, feature extraction, evaluation metrics, confusion matrices, prediction samples, and model checkpoints inside the `results/` directory.

## Repository Structure
* `HW2_ImageClassification.ipynb`: Complete pipeline for data loading, stratified splitting, training, feature extraction, evaluation, and plotting.
* `results/metrics_summary.csv`: Summary table of accuracy, Macro-F1, trainable parameters, and training time for all methods.
* `results/split_indices.json`: Saved indices for the 80/20 stratified train/validation split (`seed=42`).
* `results/cm_full_finetuning.png`: Confusion matrix for Full Fine-Tuning.
* `results/cm_classifier_head.png`: Confusion matrix for Classifier-Head Training.
* `results/cm_featext_logreg.png`: Confusion matrix for Feature Extraction + Logistic Regression.
* `results/sample_predictions.png`: Representative correct and incorrect test set predictions.
* `results/mobilenet_full.pth` & `results/mobilenet_head.pth`: Best validation checkpoints for the PyTorch models.
