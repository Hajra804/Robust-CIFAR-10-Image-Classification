# Robust-CIFAR-10-Image-Classification


A deep learning project exploring robust image classification on CIFAR-10 using only **500 labeled training images**. The model is trained from scratch and evaluated on clean and distribution-shifted test data to study how data augmentation, regularization, and test-time augmentation affect classification performance.

## Project Overview

Training deep learning models with limited labeled data presents significant challenges, particularly when images contain transformations or corruptions that differ from the training distribution.

This project investigates these challenges using a compact convolutional neural network inspired by LeNet/AlexNet-style feature extraction and Inception modules. The architecture combines multiple convolutional feature-extraction paths with global average pooling to produce a lightweight model with approximately **86K trainable parameters**.

The model is trained using a small, class-balanced CIFAR-10 subset, with a dedicated validation split for model selection and hyperparameter tuning. Final evaluation is performed on clean and shifted test sets that remain separate from the training and validation process.

## Key Features

- **Limited-data learning:** Uses only 500 labeled CIFAR-10 training images, with 400 for training and 100 for validation.
- **Compact CNN architecture:** Combines a convolutional stem, Inception-style modules, and global average pooling.
- **Training from scratch:** Uses no pretrained weights.
- **Data augmentation:** Applies transformations on the fly to training images only.
- **Regularization experiments:** Studies techniques for reducing overfitting in low-data settings.
- **Robustness evaluation:** Compares performance on clean and distribution-shifted CIFAR-10 test images.
- **Ablation study:** Evaluates experimental configurations across three random seeds.
- **Test-time augmentation (TTA):** Investigates whether predictions can be improved by averaging predictions from transformed test inputs.
- **Error analysis:** Examines misclassifications and potential weaknesses under distribution shift.

## Model Architecture

The network uses a lightweight, multi-stage CNN architecture:

1. **Convolutional stem:** Extracts low-level visual features such as edges, textures, and simple patterns.
2. **Inception modules:** Combine convolutional feature extraction at different scales.
3. **Global average pooling:** Aggregates spatial feature maps while reducing the number of classifier parameters.
4. **Classification head:** Produces logits for the 10 CIFAR-10 classes.

The model contains approximately 86K trainable parameters and is designed to investigate the trade-off between model capacity, limited training data, and robustness.

## Dataset and Experimental Setup

### CIFAR-10

CIFAR-10 contains 60,000 color images of size 32 × 32 pixels across 10 classes.

| Dataset split | Number of images | Purpose |
|---|---:|---|
| Small training subset | 500 | Fixed, class-balanced sample |
| Training set | 400 | Model optimization |
| Validation set | 100 | Model selection and tuning |
| Clean test set | 10,000 | Final evaluation |
| Shifted test set | 10,000 | Robustness evaluation |

The 500-image subset is selected using a fixed random seed of `1234`. The training/validation split preserves 40 training images and 10 validation images per class.

The original CIFAR-10 test set is kept separate from training and validation.

### Distribution Shift

The shifted test set introduces image transformations intended to evaluate robustness, including:

- Random rotations within approximately ±25 degrees.
- Brightness and contrast changes.
- Rectangular occlusion covering approximately 25% of an image.
- An additional hidden corruption defined by the official assignment specification.

The shifted set is generated using seed `999`. The hidden corruption function is currently a placeholder and must be replaced with the official implementation before reporting final robustness results.

## Experimental Plan

The project is organized into six parts:

- **Part A — Baseline:** Establishes the initial performance of the CNN.
- **Part B — Data Augmentation:** Investigates the effect of training-time augmentation.
- **Part C — Regularization:** Evaluates methods for controlling overfitting.
- **Part D — Ablation Study:** Compares experimental configurations across three random seeds.
- **Part E — Error Analysis:** Examines classification errors and robustness limitations.
- **Part F — Test-Time Augmentation:** Evaluates inference-time augmentation on test images.

Results are summarized using the mean and standard deviation across training seeds `0`, `1`, and `2`.

## Reproducibility

The experiments use fixed seeds for subset selection, dataset splitting, and repeated training runs. The split indices are saved in `split_indices.npz` to support consistent experiments.

All model selection and hyperparameter tuning are restricted to the validation set. Neither the clean nor the shifted test set is used for training or tuning.

Training is limited to a maximum of 60 epochs. Augmentation is applied on the fly to training data only, and no pretrained weights are used.

## Requirements

- Python 3
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

Install the dependencies:

```bash
pip install torch torchvision matplotlib scikit-learn numpy jupyter
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/cifar10-robust-classification.git
   cd cifar10-robust-classification
   ```

2. Install the required dependencies.

3. Launch Jupyter Notebook:

   ```bash
   jupyter notebook Section_B_CIFAR10_Robust.ipynb
   ```

4. Open the notebook and select **Run All**.

5. CIFAR-10 will download automatically when required. A GPU is recommended to reduce training time.

### Official Shifted Test Set

To use the official shifted test dataset, set the following configuration in the notebook:

```python
OFFICIAL_SHIFTED = "shifted_test.npz"
```

The file must contain NumPy arrays named `x` and `y`, corresponding to images and labels.

Ensure that the official hidden corruption implementation is supplied before running the final experiments.

## Repository Structure

```text
cifar10-robust-classification/
├──_CIFAR10_Robust.ipynb
├── split_indices.npz
├── shifted_test.npz          # Optional official test data
├── README.md
└── requirements.txt          # Optional dependency file
```

The shifted test file is optional if the notebook generates a non-official shifted set for development. Follow the assignment instructions when performing official evaluation.

## Results

Record the final results after executing all experiments.

Recommended metrics include:

- Clean test accuracy.
- Shifted test accuracy.
- Validation accuracy.
- Mean and standard deviation across three seeds.
- Per-class performance and error-analysis findings.
- Performance changes resulting from augmentation, regularization, and TTA.

**Note:** No numerical performance results are claimed here. Populate the results section using the actual experimental outputs.

## Limitations

- The training dataset is intentionally small and may limit generalization.
- The shifted test set measures robustness against specified transformations, not every possible real-world corruption.
- Final robustness conclusions depend on implementing the official hidden corruption.
- Results may vary slightly across hardware and nondeterministic operations.


