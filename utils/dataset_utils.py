# Dataset utilities

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DatasetLoader:
    # Dataset loader
    
    @staticmethod
    def list_available_datasets() -> Dict[str, str]:
        # List datasets
        return {
            'breast_cancer': 'UCI Breast Cancer Wisconsin (Diagnostic) - 569 samples, 30 features, binary classification',
            'pima_diabetes': 'Pima Indians Diabetes - 768 samples, 8 features, moderate imbalance (~35% positive)',
            'synthetic_ndc': 'Synthetic NDC (Normal Distribution Clusters) - controllable imbalance, Gaussian clusters',
            'synthetic_imbalanced': 'Synthetic Imbalanced Dataset - highly imbalanced (customizable ratio)',
        }
    
    @staticmethod
    def get_dataset_info() -> str:
        # Short dataset info
        return "See README for dataset info."
    
    @staticmethod
    def load_breast_cancer_data(test_size: float = 0.2, 
                                 random_state: int = 42,
                                 normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Load breast cancer data
        data = load_breast_cancer()
        X, y = data.data, data.target
        
        # Convert labels to {1, -1} (1=malignant, -1=benign)
        y = np.where(y == 1, -1, 1)  # sklearn: 1=malignant, 0=benign -> our convention: 1=positive, -1=negative
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalize
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def load_pima_diabetes(test_size: float = 0.2,
                          random_state: int = 42,
                          normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Load or synthesize Pima diabetes
        try:
            # Try to load from common locations
            for path in ['pima-indians-diabetes.csv', 'data/pima-indians-diabetes.csv', 
                        'datasets/pima-indians-diabetes.csv']:
                try:
                    df = pd.read_csv(path, header=None)
                    X = df.iloc[:, :-1].values
                    y = df.iloc[:, -1].values
                    # Convert to {1, -1}
                    y = np.where(y == 1, 1, -1)
                    break
                except FileNotFoundError:
                    continue
            else:
                raise FileNotFoundError("Pima diabetes CSV not found")
                
        except FileNotFoundError:
            print("Warning: Pima diabetes CSV not found. Generating similar synthetic data...")
            # Generate synthetic data with similar characteristics
            np.random.seed(random_state)
            n_samples = 768
            n_features = 8
            
            # Approximate Pima characteristics (IR ~1.87)
            n_positive = 268
            n_negative = 500
            
            # Generate two classes with different means
            X_pos = np.random.randn(n_positive, n_features) * np.array([30, 30, 15, 20, 100, 10, 0.5, 15]) + \
                    np.array([5, 120, 70, 25, 150, 32, 0.5, 35])
            X_neg = np.random.randn(n_negative, n_features) * np.array([25, 25, 12, 18, 80, 8, 0.4, 12]) + \
                    np.array([3, 110, 68, 22, 120, 30, 0.3, 30])
            
            X = np.vstack([X_pos, X_neg])
            y = np.concatenate([np.ones(n_positive), -np.ones(n_negative)])
            
            # Shuffle
            idx = np.random.permutation(len(y))
            X, y = X[idx], y[idx]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalize
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def generate_synthetic_ndc(n_samples: int = 1000,
                               n_features: int = 10,
                               imbalance_ratio: float = 3.0,
                               cluster_sep: float = 2.0,
                               random_state: int = 42,
                               test_size: float = 0.2,
                               normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Generate synthetic NDC dataset
        np.random.seed(random_state)
        
        # Calculate samples per class
        n_minority = int(n_samples / (imbalance_ratio + 1))
        n_majority = n_samples - n_minority
        
        # Generate minority class (label = 1) centered at origin
        X_minority = np.random.randn(n_minority, n_features)
        y_minority = np.ones(n_minority)
        
        # Generate majority class (label = -1) centered at cluster_sep
        center_offset = np.ones(n_features) * cluster_sep
        X_majority = np.random.randn(n_majority, n_features) + center_offset
        y_majority = -np.ones(n_majority)
        
        # Combine
        X = np.vstack([X_minority, X_majority])
        y = np.concatenate([y_minority, y_majority])
        
        # Shuffle
        idx = np.random.permutation(len(y))
        X, y = X[idx], y[idx]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalize
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def load_from_csv(filepath: str,
                     label_column: int = -1,
                     test_size: float = 0.2,
                     random_state: int = 42,
                     normalize: bool = True,
                     header: Optional[int] = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Load dataset from CSV
        df = pd.read_csv(filepath, header=header)
        
        # Extract features and labels
        if label_column == -1:
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        else:
            feature_cols = [i for i in range(df.shape[1]) if i != label_column]
            X = df.iloc[:, feature_cols].values
            y = df.iloc[:, label_column].values
        
        # Convert labels to {1, -1} if they're not already
        unique_labels = np.unique(y)
        if len(unique_labels) == 2:
            if not (set(unique_labels) == {1, -1} or set(unique_labels) == {1.0, -1.0}):
                # Map to {1, -1}: larger value -> 1, smaller -> -1
                y = np.where(y == unique_labels[1], 1, -1)
        else:
            raise ValueError(f"Expected binary classification, found {len(unique_labels)} classes")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalize
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test


# Convenience functions
def get_dataset_recommendations():
    # Print recommendations
    loader = DatasetLoader()
    print(loader.get_dataset_info())


if __name__ == "__main__":
    # Simple demo
    print("Dataset loader module. Run functions from code.")
