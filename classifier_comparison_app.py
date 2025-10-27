"""
Streamlit GUI Application for Classifier Comparison
Compare LS-FLSTSVM with Linear SVM, Logistic Regression, and Perceptron
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models_from_scratch import LinearSVMScratch, LogisticRegressionScratch, PerceptronPocket
from metrics_calculator import MetricsCalculator
from dataset_utils import DatasetLoader
from sklearn.model_selection import StratifiedKFold

# Try to import LS-FLSTSVM if available
try:
    from Large_Scale_NonL_LSTWSVM import Large_Scale_NonL_LSTWSVM
    LSTWSVM_AVAILABLE = True
except ImportError:
    LSTWSVM_AVAILABLE = False


# Set page config
st.set_page_config(
    page_title="Classifier Comparison Framework",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: var(--streamlit-sidebar-background-color);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


class LSFLTSVMWrapper:
    """Wrapper for LS-FLSTSVM to match scikit-learn-like interface"""
    
    def __init__(self, c0=1.0, ir=1.0, c1=1.0, c3=0.5, kernel_type='rbf', kernel_params=None):
        self.c0 = c0
        self.ir = ir
        self.c1 = c1
        self.c3 = c3
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params if kernel_params else [2.0]
        
        self.trained = False
        self.train_time = 0.0
        
    def fit(self, X, y):
        """Train the model"""
        # Combine X and y for LSTWSVM format
        A_train = np.column_stack([X, y])
        
        # Create dummy test set (we'll use real test set in predict)
        A_test = A_train[:10]  # Small dummy test set
        
        FunPara = {
            'c0': self.c0,
            'ir': self.ir,
            'c1': self.c1,
            'c3': self.c3,
            'kerfPara': {
                'type': self.kernel_type,
                'pars': self.kernel_params
            }
        }
        
        # Train (we only care about training time here)
        start = time.time()
        try:
            _, _, _, train_time, _ = Large_Scale_NonL_LSTWSVM(A_train, A_test, FunPara)
            self.train_time = train_time
        except Exception as e:
            st.error(f"Error training LS-FLSTSVM: {str(e)}")
            raise
        
        self.trained = True
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X_test):
        """Predict on test data"""
        if not self.trained:
            raise ValueError("Model must be fitted before prediction")
        
        # Combine train and test for LSTWSVM
        A_train = np.column_stack([self.X_train, self.y_train])
        
        # Create dummy test labels (will be ignored by LSTWSVM in prediction)
        dummy_y_test = np.ones(len(X_test))
        A_test = np.column_stack([X_test, dummy_y_test])
        
        FunPara = {
            'c0': self.c0,
            'ir': self.ir,
            'c1': self.c1,
            'c3': self.c3,
            'kerfPara': {
                'type': self.kernel_type,
                'pars': self.kernel_params
            }
        }
        
        _, _, y_pred, _, _ = Large_Scale_NonL_LSTWSVM(A_train, A_test, FunPara)
        return y_pred
    
    def decision_function(self, X_test):
        """Return decision scores (using predictions as proxy)"""
        return self.predict(X_test).astype(float)
    
    def score(self, X, y):
        """Calculate accuracy"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def perform_cross_validation(clf, X, y, cv=5):
    """
    Perform stratified k-fold cross-validation
    
    Returns:
    --------
    results : dict
        Dictionary with mean and std for each metric
    fold_results : list
        List of results for each fold
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train
        clf_fold = type(clf)(**clf.__dict__) if not isinstance(clf, LSFLTSVMWrapper) else \
                   LSFLTSVMWrapper(clf.c0, clf.ir, clf.c1, clf.c3, clf.kernel_type, clf.kernel_params)
        
        try:
            clf_fold.fit(X_train_fold, y_train_fold)
        except Exception as e:
            st.error(f"Error in fold {fold_idx+1}: {str(e)}")
            continue
        
        # Predict
        y_pred = clf_fold.predict(X_val_fold)
        
        try:
            y_scores = clf_fold.decision_function(X_val_fold)
        except:
            y_scores = None
        
        # Calculate metrics
        metrics = MetricsCalculator.calculate_all_metrics(y_val_fold, y_pred, y_scores)
        fold_results.append(metrics)
    
    # Aggregate results
    if not fold_results:
        return {}, []
    
    results = {}
    metric_names = fold_results[0].keys()
    
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_results if not np.isnan(fold[metric_name])]
        if values:
            results[f"{metric_name}_mean"] = np.mean(values)
            results[f"{metric_name}_std"] = np.std(values)
        else:
            results[f"{metric_name}_mean"] = np.nan
            results[f"{metric_name}_std"] = np.nan
    
    return results, fold_results


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plot confusion matrix"""
    tn, fp, fn, tp = MetricsCalculator.confusion_matrix(y_true, y_pred)
    
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative (-1)', 'Positive (1)'],
                yticklabels=['Negative (-1)', 'Positive (1)'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    return fig


def plot_roc_curves(results_dict):
    """Plot ROC curves for all models"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, data in results_dict.items():
        if 'fpr' in data and 'tpr' in data:
            auc = data.get('auc_roc', 0.0)
            ax.plot(data['fpr'], data['tpr'], label=f"{model_name} (AUC={auc:.3f})", linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_pr_curves(results_dict):
    """Plot Precision-Recall curves for all models"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for model_name, data in results_dict.items():
        if 'precision' in data and 'recall' in data:
            auc = data.get('auc_pr', 0.0)
            ax.plot(data['recall'], data['precision'], label=f"{model_name} (AUC={auc:.3f})", linewidth=2)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_metrics_comparison(comparison_df, metric_names):
    """Plot bar chart comparing metrics across models"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metric_names):
        if idx < len(axes):
            ax = axes[idx]
            data = comparison_df[metric].values
            models = comparison_df.index.values
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            ax.bar(models, data, color=colors)
            ax.set_title(metric)
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(len(metric_names), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    st.title("Classifier Comparison Framework")
    st.markdown("### Compare LS-FLSTSVM with Baseline Classifiers")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    st.sidebar.subheader("1. Dataset Selection")
    dataset_choice = st.sidebar.selectbox(
        "Choose dataset:",
        ["Breast Cancer (UCI)", "Pima Diabetes", "Synthetic NDC", "Upload CSV"]
    )
    
    # Load dataset
    X_train, X_test, y_train, y_test = None, None, None, None
    
    if dataset_choice == "Breast Cancer (UCI)":
        X_train, X_test, y_train, y_test = DatasetLoader.load_breast_cancer_data()
        st.sidebar.success(f"✓ Loaded: {len(y_train)} train, {len(y_test)} test samples")
        
    elif dataset_choice == "Pima Diabetes":
        X_train, X_test, y_train, y_test = DatasetLoader.load_pima_diabetes()
        st.sidebar.success(f"✓ Loaded: {len(y_train)} train, {len(y_test)} test samples")
        
    elif dataset_choice == "Synthetic NDC":
        n_samples = st.sidebar.slider("Total samples:", 500, 5000, 1000, 100)
        n_features = st.sidebar.slider("Features:", 5, 50, 10, 5)
        imb_ratio = st.sidebar.slider("Imbalance ratio:", 1.0, 10.0, 3.0, 0.5)
        cluster_sep = st.sidebar.slider("Cluster separation:", 0.5, 5.0, 2.0, 0.5)
        
        X_train, X_test, y_train, y_test = DatasetLoader.generate_synthetic_ndc(
            n_samples=n_samples, n_features=n_features,
            imbalance_ratio=imb_ratio, cluster_sep=cluster_sep
        )
        st.sidebar.success(f"Generated: {len(y_train)} train, {len(y_test)} test samples")
        
    elif dataset_choice == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.write(f"Loaded {len(df)} rows, {len(df.columns)} columns")
                
                # Assume last column is label
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                
                # Convert labels to {1, -1}
                unique_labels = np.unique(y)
                if len(unique_labels) == 2:
                    y = np.where(y == unique_labels[1], 1, -1)
                
                # Split
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Normalize
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                
                st.sidebar.success(f"✓ Processed: {len(y_train)} train, {len(y_test)} test samples")
                
            except Exception as e:
                st.sidebar.error(f"Error loading file: {str(e)}")
    
    # Model selection
    st.sidebar.subheader("2. Select Classifiers")
    use_svm = st.sidebar.checkbox("Linear SVM (SMO)", value=True)
    use_logreg = st.sidebar.checkbox("Logistic Regression", value=True)
    use_perceptron = st.sidebar.checkbox("Perceptron (Pocket)", value=True)
    use_lstwsvm = st.sidebar.checkbox("LS-FLSTSVM", value=LSTWSVM_AVAILABLE, disabled=not LSTWSVM_AVAILABLE)
    
    if not LSTWSVM_AVAILABLE and use_lstwsvm:
        st.sidebar.warning("LS-FLSTSVM not available. Check if Large_Scale_NonL_LSTWSVM.py is in the directory.")
    
    # Cross-validation settings
    st.sidebar.subheader("3. Evaluation Settings")
    use_cv = st.sidebar.checkbox("Use Cross-Validation", value=False)
    cv_folds = st.sidebar.slider("CV Folds:", 3, 10, 5) if use_cv else 5
    
    # Run button
    run_comparison = st.sidebar.button("Run Comparison", type="primary")
    
    # Main content area
    if X_train is not None:
        # Display dataset info
        st.subheader("Dataset Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training Samples", len(y_train))
        with col2:
            st.metric("Test Samples", len(y_test))
        with col3:
            st.metric("Features", X_train.shape[1])
        with col4:
            unique, counts = np.unique(y_train, return_counts=True)
            imb_ratio = max(counts) / min(counts)
            st.metric("Imbalance Ratio", f"{imb_ratio:.2f}:1")
        
        # Class distribution
        st.write("**Class Distribution:**")
        col_a, col_b = st.columns(2)
        with col_a:
            unique, counts = np.unique(y_train, return_counts=True)
            st.write(f"Training: {dict(zip(unique.astype(int), counts))}")
        with col_b:
            unique, counts = np.unique(y_test, return_counts=True)
            st.write(f"Test: {dict(zip(unique.astype(int), counts))}")
    
    # Run comparison
    if run_comparison and X_train is not None:
        st.subheader("Running Comparison...")
        
        # Initialize models
        models = {}
        
        if use_svm:
            models['Linear SVM'] = LinearSVMScratch(C=1.0, max_iter=1000)
        
        if use_logreg:
            models['Logistic Regression'] = LogisticRegressionScratch(
                learning_rate=0.1, max_iter=1000, reg_lambda=0.01
            )
        
        if use_perceptron:
            models['Perceptron'] = PerceptronPocket(max_iter=200)
        
        if use_lstwsvm and LSTWSVM_AVAILABLE:
            models['LS-FLSTSVM'] = LSFLTSVMWrapper(
                c0=1.0, ir=1.0, c1=1.0, c3=0.5,
                kernel_type='rbf', kernel_params=[2.0]
            )
        
        # Results storage
        results = {}
        all_metrics = []
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (name, clf) in enumerate(models.items()):
            status_text.text(f"Training {name}...")
            
            try:
                if use_cv:
                    # Cross-validation
                    cv_results, fold_results = perform_cross_validation(clf, X_train, y_train, cv=cv_folds)
                    results[name] = cv_results
                    
                    # Store for comparison
                    metrics_dict = {
                        'Model': name,
                        'Accuracy': cv_results.get('Accuracy_mean', np.nan),
                        'Precision': cv_results.get('Precision_mean', np.nan),
                        'Recall': cv_results.get('Recall_mean', np.nan),
                        'F1': cv_results.get('F1_mean', np.nan),
                        'Balanced_Acc': cv_results.get('Balanced_Accuracy_mean', np.nan),
                        'G_mean': cv_results.get('G_mean_mean', np.nan),
                        'AUC_ROC': cv_results.get('AUC_ROC_mean', np.nan),
                        'AUC_PR': cv_results.get('AUC_PR_mean', np.nan),
                    }
                    all_metrics.append(metrics_dict)
                    
                else:
                    # Simple train/test split
                    start_time = time.time()
                    clf.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    
                    y_pred = clf.predict(X_test)
                    
                    try:
                        y_scores = clf.decision_function(X_test)
                    except:
                        y_scores = None
                    
                    metrics = MetricsCalculator.calculate_all_metrics(y_test, y_pred, y_scores)
                    metrics['train_time'] = train_time
                    results[name] = metrics
                    
                    # Store ROC/PR data
                    if y_scores is not None:
                        fpr, tpr, _ = MetricsCalculator.get_roc_curve(y_test, y_scores)
                        prec, rec, _ = MetricsCalculator.get_pr_curve(y_test, y_scores)
                        results[name]['fpr'] = fpr
                        results[name]['tpr'] = tpr
                        results[name]['precision'] = prec
                        results[name]['recall'] = rec
                        results[name]['auc_roc'] = metrics['AUC_ROC']
                        results[name]['auc_pr'] = metrics['AUC_PR']
                    
                    # Store for comparison
                    metrics_dict = {
                        'Model': name,
                        'Accuracy': metrics['Accuracy'],
                        'Precision': metrics['Precision'],
                        'Recall': metrics['Recall'],
                        'F1': metrics['F1'],
                        'Balanced_Acc': metrics['Balanced_Accuracy'],
                        'G_mean': metrics['G_mean'],
                        'AUC_ROC': metrics['AUC_ROC'],
                        'AUC_PR': metrics['AUC_PR'],
                        'Train_Time': train_time
                    }
                    all_metrics.append(metrics_dict)
                
                st.success(f"{name} completed")
                
            except Exception as e:
                st.error(f"{name} failed: {str(e)}")
            
            progress_bar.progress((idx + 1) / len(models))
        
        status_text.text("All models completed!")
        
        # Display results
        st.subheader("Results")
        
        # Metrics table
        st.write("**Performance Metrics:**")
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df = metrics_df.set_index('Model')
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        # Download metrics as CSV
        csv = metrics_df.to_csv()
        st.download_button(
            label="Download Metrics as CSV",
            data=csv,
            file_name="classifier_comparison_metrics.csv",
            mime="text/csv"
        )
        
        # Visualizations (only for non-CV mode)
        if not use_cv:
            st.subheader("Visualizations")
            
            # Metrics comparison bar chart
            st.write("**Metrics Comparison:**")
            metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'Balanced_Acc', 'G_mean', 'AUC_ROC', 'AUC_PR']
            fig_metrics = plot_metrics_comparison(metrics_df, metric_cols)
            st.pyplot(fig_metrics)
            
            # ROC curves
            st.write("**ROC Curves:**")
            fig_roc = plot_roc_curves(results)
            st.pyplot(fig_roc)
            
            # PR curves
            st.write("**Precision-Recall Curves:**")
            fig_pr = plot_pr_curves(results)
            st.pyplot(fig_pr)
            
            # Confusion matrices
            st.write("**Confusion Matrices:**")
            n_models = len([m for m in models.keys() if m in results])
            cols = st.columns(min(3, n_models))
            
            col_idx = 0
            for name, clf in models.items():
                if name in results:
                    with cols[col_idx % 3]:
                        y_pred = clf.predict(X_test)
                        fig_cm = plot_confusion_matrix(y_test, y_pred, title=name)
                        st.pyplot(fig_cm)
                        plt.close(fig_cm)
                    col_idx += 1
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About:**
    This framework compares LS-FLSTSVM against three baseline classifiers
    (Linear SVM, Logistic Regression, Perceptron) for class imbalance learning.
    
    **Reference:**
    M. A. Ganaie, M. Tanveer, and C.-T. Lin,
    "Large-Scale Fuzzy Least Squares Twin SVMs for Class Imbalance Learning,"
    IEEE Trans. Fuzzy Systems, 2022.
    """)


if __name__ == "__main__":
    main()
