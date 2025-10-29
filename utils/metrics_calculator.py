# Metrics calculator

import numpy as np
from typing import Dict, Tuple, Optional


class MetricsCalculator:
    # Metrics helper class
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        # Calculate all metrics
        metrics = {}
        
        # Basic confusion matrix metrics
        tn, fp, fn, tp = MetricsCalculator.confusion_matrix(y_true, y_pred)
        metrics['TP'] = int(tp)
        metrics['TN'] = int(tn)
        metrics['FP'] = int(fp)
        metrics['FN'] = int(fn)
        
        # Accuracy
        metrics['Accuracy'] = MetricsCalculator.accuracy(y_true, y_pred)
        
        # Precision, Recall, F1
        metrics['Precision'] = MetricsCalculator.precision(y_true, y_pred)
        metrics['Recall'] = MetricsCalculator.recall(y_true, y_pred)
        metrics['F1'] = MetricsCalculator.f1_score(y_true, y_pred)
        
        # Specificity
        metrics['Specificity'] = MetricsCalculator.specificity(y_true, y_pred)
        
        # Balanced Accuracy
        metrics['Balanced_Accuracy'] = MetricsCalculator.balanced_accuracy(y_true, y_pred)
        
        # G-mean
        metrics['G_mean'] = MetricsCalculator.g_mean(y_true, y_pred)
        
        # AUC metrics (if scores provided)
        if y_scores is not None:
            try:
                metrics['AUC_ROC'] = MetricsCalculator.auc_roc(y_true, y_scores)
            except:
                metrics['AUC_ROC'] = np.nan
                
            try:
                metrics['AUC_PR'] = MetricsCalculator.auc_pr(y_true, y_scores)
            except:
                metrics['AUC_PR'] = np.nan
        else:
            metrics['AUC_ROC'] = np.nan
            metrics['AUC_PR'] = np.nan
        
        return metrics
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
        # Confusion matrix components
        # Convert to {1, -1} if needed
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == -1) & (y_pred == -1))
        fp = np.sum((y_true == -1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == -1))
        
        return tn, fp, fn, tp
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Calculate accuracy
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Calculate precision
        tn, fp, fn, tp = MetricsCalculator.confusion_matrix(y_true, y_pred)
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Calculate recall
        tn, fp, fn, tp = MetricsCalculator.confusion_matrix(y_true, y_pred)
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    @staticmethod
    def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Calculate specificity
        tn, fp, fn, tp = MetricsCalculator.confusion_matrix(y_true, y_pred)
        if tn + fp == 0:
            return 0.0
        return tn / (tn + fp)
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Calculate F1 score
        prec = MetricsCalculator.precision(y_true, y_pred)
        rec = MetricsCalculator.recall(y_true, y_pred)
        if prec + rec == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)
    
    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Balanced accuracy
        rec = MetricsCalculator.recall(y_true, y_pred)
        spec = MetricsCalculator.specificity(y_true, y_pred)
        return (rec + spec) / 2.0
    
    @staticmethod
    def g_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # G-mean
        rec = MetricsCalculator.recall(y_true, y_pred)
        spec = MetricsCalculator.specificity(y_true, y_pred)
        return np.sqrt(rec * spec)
    
    @staticmethod
    def auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        # AUC-ROC calculation
        # Sort by scores (descending)
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_scores = y_scores[desc_score_indices]
        y_true = y_true[desc_score_indices]
        
        # Count positives and negatives
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == -1)
        
        if n_pos == 0 or n_neg == 0:
            return np.nan
        
        # Calculate TPR and FPR at each threshold
        tps = np.cumsum(y_true == 1)
        fps = np.cumsum(y_true == -1)
        
        tpr = tps / n_pos
        fpr = fps / n_neg
        
        # Add (0,0) point
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        
        return auc
    
    @staticmethod
    def auc_pr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        # AUC-PR calculation
        # Sort by scores (descending)
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_scores = y_scores[desc_score_indices]
        y_true = y_true[desc_score_indices]
        
        # Count positives
        n_pos = np.sum(y_true == 1)
        
        if n_pos == 0:
            return np.nan
        
        # Calculate precision and recall at each threshold
        tps = np.cumsum(y_true == 1)
        fps = np.cumsum(y_true == -1)
        
        precision = tps / (tps + fps)
        recall = tps / n_pos
        
        # Add point at (recall=0, precision=n_pos/n_total)
        precision = np.concatenate([[n_pos / len(y_true)], precision])
        recall = np.concatenate([[0], recall])
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(precision, recall)
        
        return auc
    
    @staticmethod
    def get_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # ROC curve
        # Sort by scores (descending)
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_scores = y_scores[desc_score_indices]
        y_true = y_true[desc_score_indices]
        
        # Count positives and negatives
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == -1)
        
        # Calculate TPR and FPR at each threshold
        tps = np.cumsum(y_true == 1)
        fps = np.cumsum(y_true == -1)
        
        tpr = tps / n_pos if n_pos > 0 else tps
        fpr = fps / n_neg if n_neg > 0 else fps
        
        # Add (0,0) point
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])
        thresholds = np.concatenate([[np.inf], y_scores])
        
        return fpr, tpr, thresholds
    
    @staticmethod
    def get_pr_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # PR curve
        # Sort by scores (descending)
        desc_score_indices = np.argsort(y_scores)[::-1]
        y_scores = y_scores[desc_score_indices]
        y_true = y_true[desc_score_indices]
        
        n_pos = np.sum(y_true == 1)
        
        # Calculate precision and recall at each threshold
        tps = np.cumsum(y_true == 1)
        fps = np.cumsum(y_true == -1)
        
        precision = tps / (tps + fps)
        recall = tps / n_pos if n_pos > 0 else tps
        
        # Add starting point
        precision = np.concatenate([[n_pos / len(y_true) if len(y_true) > 0 else 1.0], precision])
        recall = np.concatenate([[0], recall])
        thresholds = np.concatenate([[np.inf], y_scores])
        
        return precision, recall, thresholds


if __name__ == "__main__":
    # Demo: Calculate metrics on synthetic predictions
    print("=" * 80)
    print("METRICS CALCULATOR DEMO")
    print("=" * 80)
    
    # Generate synthetic predictions
    np.random.seed(42)
    n = 100
    y_true = np.concatenate([np.ones(30), -np.ones(70)])  # Imbalanced: 30 pos, 70 neg
    y_scores = np.random.randn(n) + (y_true == 1) * 1.5  # Scores biased toward true labels
    y_pred = np.where(y_scores > 0.5, 1, -1)
    
    # Calculate all metrics
    metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred, y_scores)
    
    print("\nConfusion Matrix:")
    print(f"  TP: {metrics['TP']}, TN: {metrics['TN']}")
    print(f"  FP: {metrics['FP']}, FN: {metrics['FN']}")
    
    print("\nClassification Metrics:")
    print(f"  Accuracy:          {metrics['Accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['Balanced_Accuracy']:.4f}")
    print(f"  Precision:         {metrics['Precision']:.4f}")
    print(f"  Recall:            {metrics['Recall']:.4f}")
    print(f"  Specificity:       {metrics['Specificity']:.4f}")
    print(f"  F1 Score:          {metrics['F1']:.4f}")
    print(f"  G-mean:            {metrics['G_mean']:.4f}")
    print(f"  AUC-ROC:           {metrics['AUC_ROC']:.4f}")
    print(f"  AUC-PR:            {metrics['AUC_PR']:.4f}")
    
    print("\n" + "=" * 80)
