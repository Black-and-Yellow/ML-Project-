# Kernel functions

import math
import numpy as np


def kernel_function(kernel_type, u, v, param):
    # Compute kernel value
    if kernel_type == 0 or kernel_type == 1:
        # Linear kernel: u · v
        return np.dot(u, v)
    
    elif kernel_type == 2:
        # Polynomial kernel: (u · v + 1)^param
        return pow(np.dot(u, v) + 1, param)
    
    elif kernel_type == 3:
        # RBF kernel: exp(-||u-v||^2 / param^2)
        diff = u - v
        return pow(math.e, (-np.dot(diff, diff) / (param ** 2)))
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Must be 0, 1, 2, or 3.")


def center_train_kernel(K):
    # Center training kernel
    m, n = K.shape
    if m != n:
        raise ValueError("Invalid Kernel matrix: must be square")
    
    In = np.ones((m, m)) / m
    K_centered = K + np.dot(In, np.dot(K, In)) - (np.dot(K, In) + np.dot(In, K))
    return K_centered


def center_test_kernel(K):
    # Center test kernel
    l, m = K.shape
    In = np.ones((m, m)) / m
    Im = np.ones((l, m)) / m
    K_centered = K + np.dot(Im, np.dot(K, In)) - (np.dot(K, In) + np.dot(Im, K))
    return K_centered
