# Example usage

import numpy as np
from Large_Scale_NonL_LSTWSVM import Large_Scale_NonL_LSTWSVM

# Example 1: Load data from file
# If you have .mat file, use scipy
from scipy.io import loadmat

# Load MATLAB .mat file
# data = loadmat('your_data.mat')
# A_train = data['A']  # Training data with labels in last column
# A_test = data['A_test']  # Test data with labels in last column

# Example 2: Create synthetic data for demonstration
np.random.seed(42)

# Create training data (100 samples, 10 features)
# Last column is label: 1 for positive class, -1 for negative class
n_samples_train = 100
n_features = 10

X_train_pos = np.random.randn(50, n_features) + 1  # Positive class
X_train_neg = np.random.randn(50, n_features) - 1  # Negative class
y_train = np.concatenate([np.ones(50), -np.ones(50)])

# Combine features and labels
A_train = np.column_stack([np.vstack([X_train_pos, X_train_neg]), y_train])

# Create test data (50 samples)
n_samples_test = 50
X_test_pos = np.random.randn(25, n_features) + 1
X_test_neg = np.random.randn(25, n_features) - 1
y_test = np.concatenate([np.ones(25), -np.ones(25)])

A_test = np.column_stack([np.vstack([X_test_pos, X_test_neg]), y_test])

# Set parameters
FunPara = {
    'c0': 1.0,        # Fuzzy membership parameter
    'ir': 1.0,        # Imbalance ratio
    'c1': 1.0,        # Regularization parameter for model 1
    'c3': 0.5,        # Regularization parameter (c3=c4)
    'kerfPara': {
        'type': 'rbf',     # Kernel type: 'rbf', 'lin', or 'poly'
        'pars': [2.0]      # For RBF: [sigma^2], for poly: [coef, degree]
    }
}

# For polynomial kernel, use:
# 'kerfPara': {
#     'type': 'poly',
#     'pars': [1, 3]    # [coef0, degree]
# }

# For linear kernel, use:
# 'kerfPara': {
#     'type': 'lin',
#     'pars': []        # No parameters needed
# }

print("Training and testing LSTWSVM...")
print(f"Training samples: {A_train.shape[0]}")
print(f"Test samples: {A_test.shape[0]}")
print(f"Features: {n_features}")
print("-" * 50)

# Run the classifier
try:
    accuracy, true_labels, predicted_labels, train_time, output_info = \
        Large_Scale_NonL_LSTWSVM(A_train, A_test, FunPara)
    
    print(f"Training Time: {train_time:.4f} seconds")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Function: {output_info['function_name']}")
    print("-" * 50)
    
    # Show some predictions
    print("\nSample predictions (first 10):")
    print("True Label | Predicted")
    print("-" * 25)
    for i in range(min(10, len(true_labels))):
        print(f"{true_labels[i]:10.0f} | {predicted_labels[i]:10.0f}")
    
    # Calculate confusion matrix
    true_pos = np.sum((true_labels == 1) & (predicted_labels == 1))
    true_neg = np.sum((true_labels == -1) & (predicted_labels == -1))
    false_pos = np.sum((true_labels == -1) & (predicted_labels == 1))
    false_neg = np.sum((true_labels == 1) & (predicted_labels == -1))
    
    print("\n" + "-" * 50)
    print("Confusion Matrix:")
    print(f"True Positives:  {true_pos}")
    print(f"True Negatives:  {true_neg}")
    print(f"False Positives: {false_pos}")
    print(f"False Negatives: {false_neg}")
    
except NotImplementedError as e:
    print(f"Error: {e}")
    print("\nYou need to implement the nufuzz2() function.")
    print("This function should compute fuzzy membership values for your data.")
    
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

# CSV example removed

# Custom data example removed