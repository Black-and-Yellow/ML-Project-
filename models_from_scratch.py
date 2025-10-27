"""
From-Scratch Classifiers for Comparison with LS-FLSTSVM
Implements Linear SVM, Logistic Regression, and Perceptron with Pocket Algorithm
All classifiers avoid using sklearn's fit() methods
"""

import numpy as np
from typing import Optional, Tuple
import time


class LinearSVMScratch:
    """
    Linear SVM implemented from scratch using Sequential Minimal Optimization (SMO)
    Solves the dual problem: max Σα_i - 0.5 Σ Σ α_i α_j y_i y_j (x_i·x_j)
    Subject to: 0 ≤ α_i ≤ C and Σ α_i y_i = 0
    """
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, tol: float = 1e-3, kernel: str = 'linear'):
        """
        Parameters:
        -----------
        C : float
            Regularization parameter (higher C = less regularization)
        max_iter : int
            Maximum number of iterations for SMO
        tol : float
            Tolerance for stopping criterion
        kernel : str
            Kernel type ('linear' only for now)
        """
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
        """Linear kernel: x1 · x2"""
        return np.dot(x1, x2)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearSVMScratch':
        """
        Train the SVM using simplified SMO algorithm
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Labels {1, -1}
        """
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
        """
        Predict class labels
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted labels {1, -1}
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function (signed distance to hyperplane)
        
        Returns:
        --------
        scores : np.ndarray, shape (n_samples,)
            Decision scores
        """
        return np.dot(X, self.w) + self.b
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy
        
        Returns:
        --------
        accuracy : float
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class LogisticRegressionScratch:
    """
    Logistic Regression implemented from scratch using Batch Gradient Descent
    Minimizes: J(w,b) = (1/n) Σ log(1 + exp(-y_i(w·x_i + b))) + λ||w||²
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 reg_lambda: float = 0.01, tol: float = 1e-4, 
                 learning_rate_decay: float = 0.99):
        """
        Parameters:
        -----------
        learning_rate : float
            Initial learning rate for gradient descent
        max_iter : int
            Maximum number of iterations
        reg_lambda : float
            L2 regularization parameter
        tol : float
            Tolerance for early stopping
        learning_rate_decay : float
            Learning rate decay factor per epoch
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.learning_rate_decay = learning_rate_decay
        
        self.w = None
        self.b = 0.0
        self.loss_history = []
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function with numerical stability"""
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionScratch':
        """
        Train logistic regression using batch gradient descent
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Labels {1, -1}
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Learning rate schedule
        lr = self.learning_rate
        
        for iteration in range(self.max_iter):
            # Compute predictions (logistic function)
            z = np.dot(X, self.w) + self.b
            # For labels {1,-1}, we use y*z in logistic loss
            # loss = log(1 + exp(-y*z))
            
            # Compute gradients
            # d/dw log(1 + exp(-y*z)) = -y*x*sigmoid(-y*z)
            # For numerical stability, use: -y*x / (1 + exp(y*z))
            margins = y * z
            exp_margins = np.exp(np.clip(margins, -500, 500))  # Clip for stability
            grad_factor = -y / (1 + exp_margins)
            
            grad_w = np.dot(X.T, grad_factor) / n_samples + 2 * self.reg_lambda * self.w
            grad_b = np.sum(grad_factor) / n_samples
            
            # Update weights
            self.w -= lr * grad_w
            self.b -= lr * grad_b
            
            # Compute loss for monitoring
            loss = np.mean(np.log(1 + np.exp(-margins))) + self.reg_lambda * np.sum(self.w ** 2)
            self.loss_history.append(loss)
            
            # Early stopping check
            if iteration > 10:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    break
            
            # Learning rate decay
            lr *= self.learning_rate_decay
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Returns:
        --------
        proba : np.ndarray, shape (n_samples,)
            P(y=1|X) for each sample
        """
        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Returns:
        --------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted labels {1, -1}
        """
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, -1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function (w·x + b)
        
        Returns:
        --------
        scores : np.ndarray, shape (n_samples,)
        """
        return np.dot(X, self.w) + self.b
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy
        
        Returns:
        --------
        accuracy : float
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class PerceptronPocket:
    """
    Perceptron with Pocket Algorithm
    Classic online learning algorithm that keeps the best weight vector found
    Handles non-separable data by remembering the weights with lowest training error
    """
    
    def __init__(self, max_iter: int = 1000, learning_rate: float = 1.0):
        """
        Parameters:
        -----------
        max_iter : int
            Maximum number of epochs (passes through data)
        learning_rate : float
            Learning rate for weight updates (typically 1.0 for perceptron)
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
        self.w = None
        self.b = 0.0
        self.w_pocket = None  # Best weights found
        self.b_pocket = 0.0   # Best bias found
        self.best_error = float('inf')
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PerceptronPocket':
        """
        Train perceptron with pocket algorithm
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y : np.ndarray, shape (n_samples,)
            Labels {1, -1}
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Initialize pocket
        self.w_pocket = np.copy(self.w)
        self.b_pocket = self.b
        self.best_error = n_samples  # Start with worst possible error
        
        for epoch in range(self.max_iter):
            errors = 0
            
            # Shuffle data for stochastic updates
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                xi = X[idx]
                yi = y[idx]
                
                # Compute prediction
                activation = np.dot(self.w, xi) + self.b
                prediction = 1 if activation >= 0 else -1
                
                # Update if misclassified
                if prediction != yi:
                    self.w += self.learning_rate * yi * xi
                    self.b += self.learning_rate * yi
                    errors += 1
            
            # Check if current weights are better than pocket
            current_error = self._count_errors(X, y)
            if current_error < self.best_error:
                self.best_error = current_error
                self.w_pocket = np.copy(self.w)
                self.b_pocket = self.b
            
            # Early stopping if perfect classification
            if current_error == 0:
                break
        
        # Use pocket weights (best found during training)
        self.w = self.w_pocket
        self.b = self.b_pocket
        
        return self
    
    def _count_errors(self, X: np.ndarray, y: np.ndarray) -> int:
        """Count number of misclassifications"""
        predictions = self.predict(X)
        return np.sum(predictions != y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Returns:
        --------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted labels {1, -1}
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function (w·x + b)
        
        Returns:
        --------
        scores : np.ndarray, shape (n_samples,)
        """
        return np.dot(X, self.w) + self.b
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy
        
        Returns:
        --------
        accuracy : float
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


if __name__ == "__main__":
    # Demo: Test all three classifiers on synthetic data
    print("=" * 80)
    print("FROM-SCRATCH CLASSIFIERS DEMO")
    print("=" * 80)
    
    # Generate synthetic binary classification data
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X_pos = np.random.randn(100, n_features) + 1.0
    X_neg = np.random.randn(100, n_features) - 1.0
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(100), -np.ones(100)])
    
    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    
    # Split train/test
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test, {n_features} features")
    
    # Test each classifier
    classifiers = [
        ("Linear SVM (SMO)", LinearSVMScratch(C=1.0, max_iter=1000)),
        ("Logistic Regression", LogisticRegressionScratch(learning_rate=0.1, max_iter=500)),
        ("Perceptron (Pocket)", PerceptronPocket(max_iter=100))
    ]
    
    for name, clf in classifiers:
        print(f"\n{name}")
        print("-" * 40)
        
        # Train
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Predict
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        print(f"  Training time:   {train_time:.4f}s")
        print(f"  Train accuracy:  {train_acc:.4f}")
        print(f"  Test accuracy:   {test_acc:.4f}")
    
    print("\n" + "=" * 80)
