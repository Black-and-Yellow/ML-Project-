# Twin SVM implementation

import numpy as np
from scipy.linalg import solve
import time


class TwinSVM:
    # TwinSVM class
    
    def __init__(self, c1=1.0, c2=1.0, kernel='linear', gamma=1.0, degree=3, coef0=0.0):
        self.c1 = c1
        self.c2 = c2
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None
        self.u1 = None
        self.u2 = None
        
        self.A = None  # Training data for class +1
        self.B = None  # Training data for class -1
        self.train_time = 0.0
        
    def _kernel_matrix(self, X1, X2):
        # Kernel matrix
        if self.kernel == 'linear':
            return X1 @ X2.T
        
        elif self.kernel == 'rbf':
            # RBF kernel: exp(-gamma * ||x - y||^2)
            X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            distances = X1_norm + X2_norm - 2 * (X1 @ X2.T)
            return np.exp(-self.gamma * distances)
        
        elif self.kernel == 'poly':
            # Polynomial kernel: (gamma * x^T y + coef0)^degree
            return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        # Train model
        start_time = time.time()
        
        # Separate data by class
        self.A = X[y == 1]  # Class +1
        self.B = X[y == -1]  # Class -1
        
        m1 = self.A.shape[0]  # Number of samples in class +1
        m2 = self.B.shape[0]  # Number of samples in class -1
        
        # Add bias term (column of ones)
        A_aug = np.column_stack([self.A, np.ones(m1)])
        B_aug = np.column_stack([self.B, np.ones(m2)])
        
        if self.kernel == 'linear':
            # Linear TWSVM
            # Solve for hyperplane 1 (close to A, far from B)
            # min ||A*u1||^2 + c1 * ||xi||^2
            # s.t. -(B*u1) + xi >= e, xi >= 0
            
            # QPP formulation: (A^T A + 1/c1 * B^T B) u1 = A^T e_A
            G = A_aug.T @ A_aug
            H = B_aug.T @ B_aug
            
            try:
                # Regularized solution
                P1 = G + (1.0 / self.c1) * H + 1e-8 * np.eye(A_aug.shape[1])
                q1 = np.sum(A_aug, axis=0)
                self.u1 = solve(P1, q1, assume_a='pos')
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                P1 = G + (1.0 / self.c1) * H + 1e-6 * np.eye(A_aug.shape[1])
                self.u1 = np.linalg.lstsq(P1, q1, rcond=None)[0]
            
            # Solve for hyperplane 2 (close to B, far from A)
            # min ||B*u2||^2 + c2 * ||eta||^2
            # s.t. (A*u2) + eta >= e, eta >= 0
            
            try:
                P2 = H + (1.0 / self.c2) * G + 1e-8 * np.eye(B_aug.shape[1])
                q2 = np.sum(B_aug, axis=0)
                self.u2 = solve(P2, q2, assume_a='pos')
            except np.linalg.LinAlgError:
                P2 = H + (1.0 / self.c2) * G + 1e-6 * np.eye(B_aug.shape[1])
                self.u2 = np.linalg.lstsq(P2, q2, rcond=None)[0]
            
            # Extract weights and biases
            self.w1 = self.u1[:-1]
            self.b1 = self.u1[-1]
            self.w2 = self.u2[:-1]
            self.b2 = self.u2[-1]
            
        else:
            # Nonlinear TWSVM with kernel
            # Compute kernel matrices
            K_AA = self._kernel_matrix(self.A, self.A)
            K_BB = self._kernel_matrix(self.B, self.B)
            K_BA = self._kernel_matrix(self.B, self.A)
            K_AB = self._kernel_matrix(self.A, self.B)
            
            # Augment with bias column
            K_AA_aug = np.column_stack([K_AA, np.ones(m1)])
            K_BB_aug = np.column_stack([K_BB, np.ones(m2)])
            K_BA_aug = np.column_stack([K_BA, np.ones(m2)])
            K_AB_aug = np.column_stack([K_AB, np.ones(m1)])
            
            # Solve for hyperplane 1
            try:
                P1 = K_AA_aug.T @ K_AA_aug + (1.0 / self.c1) * K_BA_aug.T @ K_BA_aug + 1e-8 * np.eye(m1 + 1)
                q1 = K_AA_aug.T @ np.ones(m1)
                self.u1 = solve(P1, q1, assume_a='pos')
            except np.linalg.LinAlgError:
                P1 = K_AA_aug.T @ K_AA_aug + (1.0 / self.c1) * K_BA_aug.T @ K_BA_aug + 1e-6 * np.eye(m1 + 1)
                self.u1 = np.linalg.lstsq(P1, q1, rcond=None)[0]
            
            # Solve for hyperplane 2
            try:
                P2 = K_BB_aug.T @ K_BB_aug + (1.0 / self.c2) * K_AB_aug.T @ K_AB_aug + 1e-8 * np.eye(m2 + 1)
                q2 = K_BB_aug.T @ np.ones(m2)
                self.u2 = solve(P2, q2, assume_a='pos')
            except np.linalg.LinAlgError:
                P2 = K_BB_aug.T @ K_BB_aug + (1.0 / self.c2) * K_AB_aug.T @ K_AB_aug + 1e-6 * np.eye(m2 + 1)
                self.u2 = np.linalg.lstsq(P2, q2, rcond=None)[0]
        
        self.train_time = time.time() - start_time
        return self
    
    def decision_function(self, X):
        # Decision function
        if self.kernel == 'linear':
            # Distance to hyperplane 1 (close to class +1)
            dist1 = np.abs(X @ self.w1 + self.b1) / (np.linalg.norm(self.w1) + 1e-10)
            
            # Distance to hyperplane 2 (close to class -1)
            dist2 = np.abs(X @ self.w2 + self.b2) / (np.linalg.norm(self.w2) + 1e-10)
            
        else:
            # Kernel distances
            K_XA = self._kernel_matrix(X, self.A)
            K_XB = self._kernel_matrix(X, self.B)
            
            K_XA_aug = np.column_stack([K_XA, np.ones(X.shape[0])])
            K_XB_aug = np.column_stack([K_XB, np.ones(X.shape[0])])
            
            dist1 = np.abs(K_XA_aug @ self.u1)
            dist2 = np.abs(K_XB_aug @ self.u2)
        
        # Return signed distance: positive for class +1, negative for class -1
        return dist2 - dist1
    
    def predict(self, X):
        # Predict labels
        distances = self.decision_function(X)
        
        # Assign to closer hyperplane
        y_pred = np.where(distances >= 0, -1, 1)
        
        return y_pred
    
    def score(self, X, y):
        # Calculate accuracy
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
