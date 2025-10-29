# Logistic regression (from scratch)

import numpy as np


class LogisticRegressionScratch:
    # Logistic regression class
    
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                 reg_lambda: float = 0.01, tol: float = 1e-4, 
                 learning_rate_decay: float = 0.99):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.learning_rate_decay = learning_rate_decay
        
        self.w = None
        self.b = 0.0
        self.loss_history = []
        
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Sigmoid
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionScratch':
        # Fit model
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
        # Predict probabilities
        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict labels
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, -1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        # Decision function
        return np.dot(X, self.w) + self.b
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        # Accuracy
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
