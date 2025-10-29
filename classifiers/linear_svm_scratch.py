# Linear SVM (SMO)

import numpy as np
from typing import Optional


class LinearSVMScratch:
    # Linear SVM class
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, tol: float = 1e-3, kernel: str = 'linear'):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        
        self.w = None
        self.b = 0.0
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None
        
    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        # Linear kernel
        return np.dot(x1, x2)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearSVMScratch':
        # Fit SVM
        n_samples, n_features = X.shape
        
        # Initialize alphas and bias
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        
        # Precompute kernel matrix for efficiency
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
        
        # SMO algorithm (simplified version)
        for iteration in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)
            
            for i in range(n_samples):
                # Calculate error for i-th sample
                E_i = np.sum(self.alpha * y * K[:, i]) + self.b - y[i]
                
                # Check KKT conditions
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    
                    # Select j != i randomly
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # Calculate error for j-th sample
                    E_j = np.sum(self.alpha * y * K[:, j]) + self.b - y[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute bounds L and H
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Update bias
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
            
            # Check convergence
            if np.linalg.norm(self.alpha - alpha_prev) < self.tol:
                break
        
        # Extract support vectors
        sv_indices = self.alpha > 1e-5
        self.support_vectors = X[sv_indices]
        self.support_labels = y[sv_indices]
        self.alpha = self.alpha[sv_indices]
        
        # Compute weight vector for linear kernel
        self.w = np.sum((self.alpha * self.support_labels)[:, np.newaxis] * self.support_vectors, axis=0)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict labels
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # Decision function
        return np.dot(X, self.w) + self.b
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Accuracy
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
