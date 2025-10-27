"""
Dataset Utilities for Classifier Comparison Framework
Provides loaders for standard datasets and synthetic data generation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DatasetLoader:
    """
    Comprehensive dataset loader for classifier comparison experiments
    """
    
    @staticmethod
    def list_available_datasets() -> Dict[str, str]:
        """Return dictionary of available datasets with descriptions"""
        return {
            'breast_cancer': 'UCI Breast Cancer Wisconsin (Diagnostic) - 569 samples, 30 features, binary classification',
            'pima_diabetes': 'Pima Indians Diabetes - 768 samples, 8 features, moderate imbalance (~35% positive)',
            'synthetic_ndc': 'Synthetic NDC (Normal Distribution Clusters) - controllable imbalance, Gaussian clusters',
            'synthetic_imbalanced': 'Synthetic Imbalanced Dataset - highly imbalanced (customizable ratio)',
        }
    
    @staticmethod
    def get_dataset_info() -> str:
        """Return detailed information about recommended datasets"""
        info = """
### Recommended Datasets for Class Imbalance Learning

#### 1. UCI Breast Cancer Wisconsin (Diagnostic)
- **Samples**: 569 (357 benign, 212 malignant)
- **Features**: 30 (real-valued, computed from digitized images)
- **Imbalance ratio**: ~1.68:1 (moderate)
- **Why suitable**: Medical diagnosis task, well-studied benchmark, balanced enough to show clear performance differences
- **Normalization**: Standardization (zero mean, unit variance) recommended
- **Typical split**: 80/20 or stratified 5-fold CV

#### 2. Pima Indians Diabetes
- **Samples**: 768 (500 negative, 268 positive)
- **Features**: 8 (medical measurements: glucose, BMI, age, etc.)
- **Imbalance ratio**: ~1.87:1 (moderate imbalance)
- **Why suitable**: Real-world medical data, established benchmark for imbalance learning, low-dimensional
- **Normalization**: StandardScaler (some features have different scales)
- **Typical split**: 80/20 or 10-fold CV

#### 3. Synthetic NDC (Normal Distribution Clusters)
- **Samples**: Customizable (default: 1000)
- **Features**: Customizable (default: 10)
- **Imbalance ratio**: Fully controllable (1:1 to 100:1)
- **Why suitable**: Controlled experiments, can test extreme imbalance, reproducible with seeds
- **Generation**: Two Gaussian clusters with controllable separation and variance
- **Use case**: Ablation studies, parameter sensitivity analysis

#### 4. Credit Card Fraud (Large-scale, optional)
- **Samples**: ~284,807 transactions
- **Features**: 30 (PCA-transformed, anonymized)
- **Imbalance ratio**: ~577:1 (highly imbalanced, 0.172% fraud)
- **Why suitable**: Large-scale, extreme imbalance, real-world financial application
- **Note**: Requires download from Kaggle; too large for quick testing
- **Recommendation**: Use for final large-scale validation only

#### 5. KEEL Imbalanced Repository
- **Datasets**: 40+ benchmark imbalanced datasets
- **Range**: IR from 1.5:1 to 130:1
- **Why suitable**: Standard benchmarks for imbalance learning research
- **Access**: Available at https://sci2s.ugr.es/keel/imbalanced.php
- **Recommendation**: Use for comprehensive comparison across IR ranges

### Dataset Selection Strategy
1. **Quick validation**: Use Breast Cancer or synthetic (fast, built-in)
2. **Standard benchmark**: Use Pima Diabetes (moderate imbalance, real-world)
3. **Ablation studies**: Use Synthetic NDC with varying IR
4. **Publication-ready**: Use KEEL datasets + one medical dataset
5. **Large-scale test**: Use Credit Card Fraud (if computational resources allow)
"""
        return info
    
    @staticmethod
    def load_breast_cancer_data(test_size: float = 0.2, 
                                 random_state: int = 42,
                                 normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess UCI Breast Cancer Wisconsin dataset
        
        Returns:
            X_train, X_test, y_train, y_test
            Labels are {1, -1} for consistency with LSTWSVM
        """
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
        """
        Load Pima Indians Diabetes dataset from CSV (if available) or generate similar synthetic
        
        Note: This requires pima-indians-diabetes.csv in the working directory
        If not available, generates similar synthetic data
        """
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
        """
        Generate synthetic Normal Distribution Cluster (NDC) dataset
        
        Parameters:
        -----------
        n_samples : int
            Total number of samples
        n_features : int
            Number of features
        imbalance_ratio : float
            Ratio of majority to minority class (e.g., 3.0 means 3:1)
        cluster_sep : float
            Separation between cluster centers (larger = easier to separate)
        random_state : int
            Random seed for reproducibility
        test_size : float
            Fraction of data for testing
        normalize : bool
            Whether to standardize features
            
        Returns:
        --------
        X_train, X_test, y_train, y_test with labels {1, -1}
        """
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
        """
        Load custom dataset from CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
        label_column : int
            Column index for labels (default: -1 for last column)
        test_size : float
            Fraction for test set
        random_state : int
            Random seed
        normalize : bool
            Whether to standardize features
        header : int or None
            Row number to use as column names (None if no header)
            
        Returns:
        --------
        X_train, X_test, y_train, y_test
        """
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
    """Print dataset recommendations"""
    loader = DatasetLoader()
    print(loader.get_dataset_info())


if __name__ == "__main__":
    # Demo: Load and display basic statistics for each dataset
    print("=" * 80)
    print("DATASET LOADER DEMO")
    print("=" * 80)
    
    # 1. Breast Cancer
    print("\n1. UCI Breast Cancer Wisconsin Dataset")
    print("-" * 40)
    X_tr, X_te, y_tr, y_te = DatasetLoader.load_breast_cancer_data()
    print(f"Training: {X_tr.shape[0]} samples, {X_tr.shape[1]} features")
    print(f"Test: {X_te.shape[0]} samples")
    unique, counts = np.unique(y_tr, return_counts=True)
    print(f"Training class distribution: {dict(zip(unique, counts))}")
    print(f"Imbalance ratio: {max(counts)/min(counts):.2f}:1")
    
    # 2. Pima Diabetes
    print("\n2. Pima Indians Diabetes Dataset")
    print("-" * 40)
    X_tr, X_te, y_tr, y_te = DatasetLoader.load_pima_diabetes()
    print(f"Training: {X_tr.shape[0]} samples, {X_tr.shape[1]} features")
    print(f"Test: {X_te.shape[0]} samples")
    unique, counts = np.unique(y_tr, return_counts=True)
    print(f"Training class distribution: {dict(zip(unique, counts))}")
    print(f"Imbalance ratio: {max(counts)/min(counts):.2f}:1")
    
    # 3. Synthetic NDC
    print("\n3. Synthetic NDC Dataset (IR=5.0)")
    print("-" * 40)
    X_tr, X_te, y_tr, y_te = DatasetLoader.generate_synthetic_ndc(
        n_samples=1000, n_features=10, imbalance_ratio=5.0, cluster_sep=2.5
    )
    print(f"Training: {X_tr.shape[0]} samples, {X_tr.shape[1]} features")
    print(f"Test: {X_te.shape[0]} samples")
    unique, counts = np.unique(y_tr, return_counts=True)
    print(f"Training class distribution: {dict(zip(unique, counts))}")
    print(f"Imbalance ratio: {max(counts)/min(counts):.2f}:1")
    
    print("\n" + "=" * 80)
    print("All datasets loaded successfully!")
    print("=" * 80)
