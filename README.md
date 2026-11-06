# investigation-of-different-activation-functions-and-architectures
The goal of this expanded experiment was to investigate the combined impact of different neural network architectures and activation functions on the prediction quality in a regression task across two distinct datasets (Iris and Wine Quality).

# Motivation & Expaction
While learning machine learning and AI, tutorials often present architectures and activation functions in isolation, while systematic, empirical comparisons across a wide grid search are rare. The underlying theory of dense layers, gradient flow, and non-linear transformations is well-documented. My expectation for this deep-dive experiment was twofold:

1.) Architecture matters: Structural shapes like Bottlenecks (08_Sanduhr) or Pyramids (06_Pyramidal) should handle feature compression and abstraction differently than flat or minimal networks (02_Klein).

2.) Activation dependency: Performance is not universal; modern activations (ELU, GELU, Swish) should consistently outperform legacy functions (Sigmoid, Tanh) depending on network depth and dataset complexity.

This repository documents the complete hands-on grid search comparing 64 model configurations per dataset under identical training conditions.

# Test environment
## Hardware
  * CPU: Intel Core i3-12100F (4 cores / 8 threads)
  * GPU: Nvidia RTX 3050 - 6GB VRAM
  * RAM: 16 GB (dual-channel)
## Software
  * OS: Ubuntu 22.04.05 LTS (64bit)
  * Visual Studio Code
  * Python 3.11.15
  * TensorFlow incl. CUDA
  * Scikit-Learn
  * Pandas
  * Seaborn & Matplotlib

# Dataset

## 1. Iris Dataset
  * Ingestion: Loaded via .csv-file (../Data/iris.csv).
  * Features: sepal length (removed sepal width due to low correlation), petal length, petal width + One-Hot-Encoded species column.
  * Target: sepal length (Regression task).
  * Scaling: Standard scaling applied to continuous features (petal.length, petal.width).
  * Split: 80% training, 20% testing.

## 2. Wine Quality Dataset
  * Ingestion: Loaded via .csv-file (../Data/winequality-red-2.csv, delimiter ;).
  * Features: 11 physicochemical properties (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol).
  * Target: quality (Regression task, values ranging from 3 to 9).
  * Scaling: Standard scaling applied to all features.
  * Split: 80% training, 20% testing.

# Architectures & Grid Search Design

To systematically test structural variations, 8 distinct architectures were paired with 8 activation functions; yielding (8x8 = 64) unique runs per dataset:

## Architectual Configurations:

  * 01_Standard_Ref: 32, 64
  * 02_Klein: 32
  * 03_Breit: 64,128
  * 04_Tief: 32, 64, 32
  * 05_Symmetrisch: 64,64
  * 06_Pyramidal: 64, 32, 16
  * 07_Inverser_Trichter: 16, 32, 64
  * 08_Sanduhr_Bottleneck: 64, 16, 64

## Activation Functions Tested

  * ReLU
  * Leaky ReLU
  * ELU
  * SeLU
  * GELU
  * Swish
  * Tanh
  * Sigmoid

## Training Hyperparameters
  * Optimizer: Adam
  * Loss Function: Mean Absolute Error (MAE)
  * Epochs: 70
  * Batch Size: 32
  * Validation Split: 20% during training

# Results & Evalutation
To evaluate model performance objectively, both Mean Absolute Error (MAE) and R²-Score were analyzed, supported by diagnostic plots (Loss progression, Prediction vs. Reality with Jitter, Error distribution, and Residual plots).

## Top 3 Models Overview
### Iris-Dataset (Top-Performers)
| Model (Architecture + Activation) | MAE | R²-Score |
| --- | --- | --- |
| 08_Sanduhr_Bottleneck_GELU | 0.2554 | 0.8512 |
| 06_Pyramidal_ELU | 0.2574 | 0.8561 |
| 01_Standard_Ref_ELU | 0.2598 | 0.8474 |

### Wine-Dataset (Top-Performers)
| Model (Architecture + Activation) | MAE | R²-Score |
| --- | --- | --- |
| 08_Sanduhr_Bottleneck_Sigmoid | 0.2892 | 0.0267 |
| 06_Pyramidal_Sigmoid | 0.2910 | 0.0449 |
| 04_Tief_GELU | 0.2915 | 0.1706 |

# Comprehensive Interpretation & Insights
**1.) The power of information bottlenecks (08_Sanduhr & 06_Pyramidal)**

Across both datasets, architectures featuring structural compression (08_Sanduhr & 06_Pyramidal) frequently secured top positions. For Iris, 08_Sanduhr_Bottleneck_GELU achieved the lowest MAE (0.2554). For Wine, Bottleneck structures combined with various activations dominated the top tier. This suggests that forcing the network through a narrow intermediate layer acts as an effective regularizer, filtering out noise and retaining salient feature representations.

**2.) Activation Function Behavior**
  * The best performers: modern exponential an smooth activations like ELU, GELU, and SeLU consistently provided stable gradients, avoiding dead zones and achieving superior convergence.
  * The failure cases (02_klein): minimal networks paired with legacy or ill-suited activations (Sigmoid,  ReLU variants on tiny widhts) suffered from catastrophic underfitting or exploding/vanishing gradients on Iris, resuling in heavily negative R²-scores (down to -9.13 for 02_klein_leaky_ReLU). Due to that fact, the 'wine-dataset' has been analyzed on top.
  * The Wine Dataset Anomaly (R² vs. MAE): On the Wine-dataset, models utilizing Sigmoid or Tanh in specific deep structures archieved low MAE values (around 0.289), yet their R²-score hovered near zero. Visual inspection of the Jitter and Residual plots revealed why: these models collapsed their predictions tightly around the global target mean (~5.6 to 5.9), producing small absolute deviations on average while completely failing to capture the variance of minority classes (e.g. exceptional quality 8 or 3 wines).

**3.) Diagnostic Plot Observations**
  * Loss Progression: Top-performing models exhibted smooth, closely aligned training and validation loss curves. Failing models displayed flat horizontal lines (zero learning) or erratic divergence.
  * Prediction vs. Reality: Successful models tracked the diagonal identity line closely. Weaker models formed horizontal bands, proving they had regressed to predicting a constant mean.
  * Residual plots:
      - good models: displayed random scatter (Gaussian noise) centered around zero, indicating no remaining systematic bias.
      - bad models: Exhibted distinct non-linear patterns, curves, or funnel shapes, conforming that the network failed to map the underlying data manifold.
   
# Next steps
This comprehensive grid search opened up several promising avenues for future research:
  * Hyperparameter Tuning: Investigate learning rate schedules and alternative optimizers.
  * Regularization Studies: introducing Drop-Out and L2-weight-decay to examine their impact on generalization in wider networks (03_Breit)
  * Task Transition: apply the identical architecture and activation matrix to classification taks to observe, if activation rankings shift.


The complete raw data tables, all 512 generated evaluation plots, input files and my Python-script are available in the repository.
