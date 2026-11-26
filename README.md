# Machine Learning — Assignment 3

## MLPs, CNNs, Autoencoders, SVMs and Clustering on CIFAR-10

**Author:** Aarish Quazi  
**Roll Number:** M26MER008  
**Course:** Machine Learning  
**Assignment:** 3 (Individual)  
**Dataset:** CIFAR-10  
**Random seed:** 26008

## 1. Overview

This assignment investigates how model architecture, optimization, spatial structure, and learned representations affect image classification on CIFAR-10. The experiments compare multilayer perceptrons (MLPs) and convolutional neural networks (CNNs), study learning versus memorization under label noise, evaluate convolutional and denoising autoencoders, and apply support vector machines (SVMs) and K-means clustering to learned features.

All experiment implementations, training loops, evaluation procedures and figure-generation code are contained in [ML_ASSIGNMENT3.ipynb](ML_ASSIGNMENT3.ipynb). The repository also includes the saved dataset split, experiment configurations, training histories, result tables, figures and model checkpoints.

## 2. Repository Structure

```text
ML_ASSIGNMENT3/
├── ML_ASSIGNMENT3.ipynb              # Main implementation (Q1–Q4)
├── ML_Assignment_3_Shortened_Final.pdf
├── cifar10_split.npz                # Saved stratified training/validation indices
├── checkpoints/                     # PyTorch model weights, SVM and K-means models
├── figures/                         # Generated plots and visualizations
├── results/                         # CSV histories/results, JSON configs, saved arrays
├── .gitignore                       # Excludes downloaded CIFAR-10 data
└── README.md
```

The `data/` directory is created locally when CIFAR-10 is downloaded. The dataset archive and extracted dataset are intentionally excluded from Git; no manual dataset upload is necessary.

## 3. Dataset and Experimental Setup

CIFAR-10 consists of 60,000 RGB images of size 32 × 32, distributed across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

| Split | Images | Images per class |
| --- | ---: | ---: |
| Training | 45,000 | 4,500 |
| Validation | 5,000 | 500 |
| Official test | 10,000 | 1,000 |

- The original 50,000 training samples are divided into a single stratified training/validation split using `train_test_split` with `random_state=26008`.
- The seed is obtained by taking all numeric characters in roll number `M26MER008` in order.
- The indices and seed are stored in [cifar10_split.npz](cifar10_split.npz).
- Images are converted to tensors and normalized to the range [-1, 1] with mean and standard deviation of 0.5 in each RGB channel.
- The common batch size is 128. The standard neural-network training schedule is 30 epochs unless otherwise stated.
- Model, checkpoint and hyperparameter selection are based on validation metrics. The official test set is used for reporting and error analysis, not for choosing hyperparameters.
- The notebook seeds Python, NumPy and PyTorch and enables deterministic cuDNN settings where applicable.

## 4. Experiments

| Question | Experiment | Method |
| --- | --- | --- |
| Q1(a) | Optimization in MLPs | Same 1,052,426-parameter MLP and initial weights; compare SGD (LR 0.01, momentum 0.9) with Adam (LR 0.001). |
| Q1(b) | Learning vs. memorization | Train a wider MLP for 100 epochs with 30% of training labels replaced by incorrect labels. Track clean-label, noisy-label and validation accuracy. |
| Q2(a) | CNN vs. MLP | Compare the 963,466-parameter CNN with the SGD MLP under a matched 30-epoch training budget. Separately extend CNN training with a learning-rate scheduler and early stopping. |
| Q2(b) | Pixel shuffling | Use a fixed permutation of 1,024 pixel positions, keeping RGB channels together. Retrain and compare MLP and CNN. |
| Q2(c) | Patch shuffling | Rearrange sixteen 8 × 8 patches while preserving local structure within each patch. Retrain and compare MLP and CNN. |
| Q3(a) | Convolutional autoencoders | Compare 32-, 128- and 512-dimensional bottlenecks using MSE reconstruction loss. |
| Q3(b) | Denoising autoencoder | Train with Gaussian noise (standard deviation 0.4); evaluate reconstruction and classification using the fixed extended CNN. |
| Q3(c) | Linear probing | Freeze each autoencoder encoder, standardize features using training statistics and train a linear classifier on each bottleneck. |
| Q4(a) | SVM classification | Select the representation by validation linear-probe accuracy. Compare linear, polynomial and RBF kernels and C values of 0.001, 0.01, 0.1, 1 and 10. |
| Q4(b) | SVM error and boundary analysis | Identify difficult class pairs from the confusion matrix and examine support vectors and near-boundary examples. |
| Q4(c) | K-means clustering | Use the same representation and 5,000-sample stratified training subset as the SVM. Compare K = 5, 10, 15 and 20 using validation silhouette score. |

## 5. Installation

Clone the repository and enter its root directory:

```bash
git clone https://github.com/aarishquazi/ML_ASSIGNMENT3.git
cd ML_ASSIGNMENT3
```

Create and activate a Python virtual environment (Linux/macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

The notebook imports PyTorch, torchvision, NumPy, pandas, Matplotlib, scikit-learn and joblib. Install these packages together with Jupyter and nbconvert:

```bash
python -m pip install torch torchvision numpy pandas matplotlib scikit-learn joblib jupyterlab nbconvert ipykernel
```

For CUDA acceleration, install a compatible PyTorch and torchvision build using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The notebook automatically selects CUDA when available and otherwise uses the CPU.

**Environment note:** The committed notebook metadata records Python 3.14.4. The repository does not currently contain a pinned requirements or environment file. The installation command above lists the dependencies identifiable from the notebook, but does not recreate exact package versions from the original experiment. A version-pinned environment from the original training machine is needed for stricter reproducibility.

## 6. How to Reproduce the Experiments

### Option A — Jupyter Notebook

```bash
python -m jupyter lab ML_ASSIGNMENT3.ipynb
```

Select the Python environment created above, then restart the kernel and run all cells in order, from Q1(a) to Q4(c). Later questions reuse models, loaders and representations defined in earlier sections, so the sections should not be executed independently in an empty kernel.

## 7. Recorded Results

The following values are taken from the saved experimental notebook outputs.

### Q1 — MLP Optimization and Memorization

| Optimizer | Best validation accuracy | Test accuracy |
| --- | ---: | ---: |
| SGD with momentum | 55.08% | 54.03% |
| Adam | 52.88% | 53.03% |

In the noisy-label experiment, the wider MLP reached 94.71% accuracy on incorrectly labeled training samples, while final validation accuracy was 40.02%. This demonstrates that the model can memorize corrupted labels without improving generalization.

### Q2 — Effect of Spatial Structure

| Model | Clean test accuracy | Pixel-shuffled test accuracy | Patch-shuffled test accuracy |
| --- | ---: | ---: | ---: |
| MLP (SGD) | 54.03% | 53.82% | 53.04% |
| CNN | 77.65% | 46.49% | 67.33% |

The CNN performs better on normal images but loses more accuracy when spatial relationships are destroyed. Preserving local information within shuffled patches improves its performance relative to pixel shuffling.

The separately extended CNN achieved 81.56% best validation accuracy and 81.41% clean test accuracy in the fixed-classifier experiment. It is not used in the matched 30-epoch comparison above.

### Q3 — Reconstruction and Learned Representations

| Bottleneck | Validation reconstruction MSE | Test reconstruction MSE | Linear-probe test accuracy |
| --- | ---: | ---: | ---: |
| 32 | 0.045630 | 0.045806 | 37.27% |
| 128 | 0.018372 | 0.018364 | 39.81% |
| 512 | 0.006910 | 0.006949 | 47.48% |

| Fixed CNN input | Test accuracy |
| --- | ---: |
| Clean images | 81.41% |
| Gaussian-corrupted images | 30.90% |
| Clean autoencoder reconstruction of corrupted images | 48.04% |
| Denoising autoencoder reconstruction | 53.18% |

The 512-dimensional bottleneck achieved the best validation reconstruction MSE and validation linear-probe accuracy among the tested representations.

### Q4 — SVM and Clustering

- Selected representation: AE-512 (selected by linear-probe validation accuracy).
- Selected SVM: RBF kernel, C = 10.
- SVM validation accuracy: 46.46%.
- SVM test accuracy: 45.88%.
- Selected K-means setting: K = 5.
- Validation silhouette score: 0.0016.
- Test adjusted Rand index (ARI): 0.0530.
- Test normalized mutual information (NMI): 0.0915.

The difficult SVM class pairs include cat/dog, automobile/truck and bird/deer.

## 8. Results, Figures and Checkpoints

| Experiment | Main result files | Representative figures |
| --- | --- | --- |
| Q1 | [Optimizer comparison](results/q1a_final_results.csv), [memorization history](results/q1b_history.csv) | [MLP curves](figures/q1a_curves.png), [memorization curves](figures/q1b_memorisation_curves.png) |
| Q2 | [Spatial comparison](results/q2_final_accuracy_table.csv), [CNN errors](results/q2_cnn_error_analysis.csv) | [Spatial comparison](figures/q2_final_comparison.png), [confusion matrices](figures/q2_confusion_matrices.png) |
| Q3 | [Autoencoder comparison](results/q3a_comparison.csv), [denoising](results/q3b_final_results.csv), [linear probing](results/q3c_final_results.csv) | [Reconstructions](figures/q3a_reconstructions.png), [denoising comparison](figures/q3b_denoising_comparison.png) |
| Q4 | [SVM sweep](results/q4a_svm_sweep.csv), [SVM results](results/q4a_final_results.csv), [clustering results](results/q4c_final_clustering_results.csv) | [SVM comparison](figures/q4a_svm_comparison.png), [clustering heatmap](figures/q4c_contingency_heatmap.png) |

The `results/` directory additionally includes the experiment JSON configurations, saved noisy labels, pixel and patch permutations, latent-feature scaling statistics, SVM/clustering training-subset indices and test predictions.

The `checkpoints/` directory stores PyTorch `.pth` state dictionaries as well as the selected SVM and K-means models in `.joblib` format. The associated model architectures and checkpoint-loading code are defined in the notebook.

The notebook contains the plotting code that generates the figures; separate figure scripts are not currently present in this repository.

## 9. Report and References

- [Assignment PDF](ML_Assignment_3_Shortened_Final.pdf)
- [CIFAR-10 dataset — official source](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [scikit-learn documentation](https://scikit-learn.org/stable/)

To obtain the commit hash identifying the exact repository version used for submission:

```bash
git rev-parse HEAD
```

This repository was prepared as part of the Machine Learning course's individual Assignment 3.
