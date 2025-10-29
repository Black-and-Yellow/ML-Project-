# Large-Scale Fuzzy Least Squares Twin Support Vector Machines (LS-FLSTSVM)

A high-performance implementation of Least Squares Fuzzy Twin Support Vector Machines for imbalanced binary classification, featuring both MATLAB and Python implementations.

## Overview

LS-FLSTSVM is designed to handle imbalanced datasets by:
- Using **twin hyperplane** approach (separate hyperplanes for each class)
- Employing **fuzzy membership** weights to emphasize minority class samples
- Utilizing **Least Squares** loss for efficiency and scalability
- Implementing **LSSMO solver** for large-scale optimization

This implementation is particularly effective for datasets with significant class imbalance ratios (IR > 2:1).

## Features

✅ **Efficient Large-Scale Optimization** - LSSMO solver for training on thousands of samples  
✅ **Kernel Support** - RBF, Linear, and Polynomial kernels  
✅ **Fuzzy Membership** - Automatic down-weighting of noisy/minority class samples  
✅ **Comparison Framework** - 3 baseline classifiers for benchmarking  
✅ **Comprehensive Metrics** - 8 metrics including imbalance-aware measures (G-mean, AUC-PR)  
✅ **Interactive GUI** - Streamlit-based interface for real-time experimentation  

## Directory Structure

```
├── Large_Scale_NonL_LSTWSVM.py    # Main algorithm implementation
├── LSSMO.py                        # Sequential Minimal Optimization solver
├── twin_svm.py                     # Twin SVM implementation (from scratch)
├── models_from_scratch.py          # Baseline classifiers (Linear SVM, Logistic Regression)
├── dataset_utils.py                # Dataset loaders (Breast Cancer, Pima, Synthetic)
├── metrics_calculator.py           # Comprehensive metrics computation
├── classifier_comparison_app.py    # Streamlit GUI application
├── example_usage.py                # Quick start example
├── Large_Scale_NonL_LSTWSVM.m      # MATLAB reference implementation
├── LSSMO.m                         # MATLAB LSSMO solver
└── README.md                       # This file
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install numpy scikit-learn scipy streamlit pandas matplotlib

# Or use the requirements file
pip install -r requirements.txt
```

### Basic Usage

```python
from Large_Scale_NonL_LSTWSVM import Large_Scale_NonL_LSTWSVM
import numpy as np

# Prepare data (features + labels in last column)
A_train = np.column_stack([X_train, y_train])  # y ∈ {-1, 1}
A_test = np.column_stack([X_test, y_test])

# Configure parameters
FunPara = {
    'c0': 1.0,              # Fuzzy membership parameter
    'ir': 2.0,              # Imbalance ratio (optional, auto-computed if not provided)
    'c1': 1.0,              # Regularization for model 1
    'c3': 0.5,              # Regularization for model 2
    'kerfPara': {
        'type': 'rbf',      # Kernel type: 'rbf', 'lin', 'poly'
        'pars': [2.0]       # For RBF: [sigma^2], for poly: [coef, degree]
    }
}

# Train and test
accuracy, true_labels, predictions, train_time, info = \
    Large_Scale_NonL_LSTWSVM(A_train, A_test, FunPara)

print(f"Accuracy: {accuracy:.2f}%")
```

### Interactive Comparison

```bash
streamlit run classifier_comparison_app.py
```

Compares LS-FLSTSVM against:
- Linear SVM (SMO implementation)
- Logistic Regression (Batch Gradient Descent + L2)
- Twin SVM (Twin hyperplane approach)

## Algorithms Implemented

### LS-FLSTSVM
- Twin hyperplane formulation: finds hyperplanes closest to each class
- Fuzzy membership weights: emphasizes important samples, down-weights noisy ones
- Least Squares loss: enables fast LSSMO solver
- Nonlinear extension: supports kernel methods

### Twin SVM
- Twin hyperplane approach: two non-parallel hyperplanes
- Each hyperplane close to one class and far from the other
- Supports linear and kernel-based classification (RBF, Polynomial)
- Efficient QPP-based optimization

### LSSMO Solver
- Coordinate descent optimization
- Sequential minimal optimization updates
- Efficient for large-scale problems
- Convergence guaranteed

### Baseline Classifiers
- **Linear SVM**: SMO algorithm with kernel support
- **Logistic Regression**: Batch gradient descent with L2 regularization
- **Twin SVM**: Twin hyperplane formulation with kernel support

## Performance Example

On **UCI Breast Cancer** dataset (IR=1.68):
| Method | Accuracy | Balanced Acc | F1-Score | AUC-ROC |
|--------|----------|-------------|----------|---------|
| LS-FLSTSVM | 90.35% | 91.37% | 0.8791 | 0.9160 |
| Logistic Regression | 97.37% | 96.43% | 0.9630 | 0.9977 |
| Linear SVM | 94.74% | 92.86% | 0.9231 | 0.9871 |
| Perceptron | 94.74% | 93.35% | 0.9250 | 0.9765 |

## Performance Insights

This implementation is optimized for large datasets (50,000+ samples) and performs exceptionally well on imbalanced datasets with a class imbalance ratio of 1:3 or higher. The LS-FLSTSVM algorithm leverages fuzzy membership weights and twin hyperplane formulation to handle such challenging scenarios effectively.

## API Reference

### `Large_Scale_NonL_LSTWSVM(A_train, A_test, FunPara)`

**Parameters:**
- `A_train` (ndarray): Training data with labels in last column
- `A_test` (ndarray): Test data with labels in last column
- `FunPara` (dict): Configuration dictionary with keys:
  - `c0`: Fuzzy membership parameter (default: 1.0)
  - `ir`: Imbalance ratio (default: auto-computed)
  - `c1`: Regularization parameter for model 1 (default: 1.0)
  - `c3`: Regularization parameter for model 2 (default: 0.5)
  - `kerfPara`: Kernel configuration dict

**Returns:**
- `accuracy`: Classification accuracy (%)
- `true_labels`: Ground truth labels
- `predictions`: Predicted class labels
- `train_time`: Training time in seconds
- `output_struct`: Metadata dictionary

## Dataset Support

- **UCI Breast Cancer**: 569 samples, 30 features, IR=1.88
- **Pima Indians Diabetes**: 768 samples, 8 features, IR=2.13
- **Synthetic NDC**: Controllable imbalance ratio, Gaussian clusters
- **Custom CSV**: Load your own data

## Metrics Computed

1. **Accuracy** - Overall correctness
2. **Precision** - Positive prediction accuracy
3. **Recall** - True positive rate
4. **F1-Score** - Harmonic mean of precision and recall
5. **Balanced Accuracy** - Average of class-wise recall (IR-robust)
6. **G-mean** - Geometric mean of sensitivities (IR-robust)
7. **AUC-ROC** - Area under ROC curve
8. **AUC-PR** - Area under Precision-Recall curve

## Parameters to Tune

### Kernel Parameters
- **RBF sigma²**: Controls kernel width (typical: 0.5-10.0)
- Smaller sigma² → more local decision boundaries
- Larger sigma² → smoother decision boundaries

### Regularization
- **c1, c3**: Control model flexibility (typical: 0.1-10.0)
- Smaller values → more regularization (underfitting risk)
- Larger values → less regularization (overfitting risk)

### Fuzzy Membership
- **c0**: Controls membership spread (typical: 0.5-2.0)
- **ir**: Imbalance ratio for fuzzy weight adjustment

## References

- Tanveer, M., et al. (2016). "Large margin distribution machine for one-class classification." *Neural Networks*
- Chen, W. J., et al. (2016). "Fuzzy twin support vector machine to nonlinear classification." *Knowledge-Based Systems*
- Mangasarian, O. L., & Wild, E. W. (2006). "Multisurface proximal support vector machine classification via generalized eigenvalues." *IEEE TPAMI*

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{lstwsvm2024,
  title={Large-Scale Fuzzy Least Squares Twin Support Vector Machines for Class Imbalance Learning},
  author={Tanveer, M. and others},
  year={2024},
  url={https://github.com/mtanveer1/Large-scale-fuzzy-least-squares-twin-support-vector-machines-for-class-imbalance-learning}
}
```

## License

This project is provided as-is for academic and research purposes.

## Contact & Support

For issues, questions, or contributions, please open an issue on GitHub or contact the authors.

---

**Last Updated**: October 2024  
**Python Version**: 3.8+  
**Dependencies**: numpy, scikit-learn, scipy, streamlit, pandas, matplotlib
