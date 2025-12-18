"""Performance metrics for ELR classification models."""
import numpy as np
from typing import Set, Any
from sklearn.metrics import confusion_matrix


class PerformanceMetrics:
    """Calculate specialized classification metrics for ELR models."""
    
    @staticmethod
    def positive_likelihood_ratio(y_true, y_pred) -> float:
        """
        Calculate Positive Likelihood Ratio (PLR).
        
        PLR = True Positive Rate / False Positive Rate
        PLR = Sensitivity / (1 - Specificity)
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_pred : array-like of shape (n_samples,)
            Predicted binary labels (0 or 1).
        
        Returns
        -------
        float
            Positive Likelihood Ratio. Returns np.inf if FPR is 0.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Calculate TPR (True Positive Rate / Sensitivity)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Calculate FPR (False Positive Rate)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Handle edge case where FPR is 0
        if fpr == 0:
            return np.inf if tpr > 0 else 0.0
        
        return tpr / fpr
    
    @staticmethod
    def false_negative_rate(y_true, y_pred) -> float:
        """
        Calculate False Negative Rate (FNR).
        
        FNR = FN / (FN + TP)
        FNR = 1 - Sensitivity
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_pred : array-like of shape (n_samples,)
            Predicted binary labels (0 or 1).
        
        Returns
        -------
        float
            False Negative Rate.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Calculate FNR
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return fnr

    @staticmethod
    def decile_ranked_performance(y_true, y_score, d: int) -> float:
        """
        Calculate Decile Ranked Performance (DRP).
        
        DRP = TPR in top d deciles / TPR in entire test set
        
        Steps:
        1. Sort predictions by score (descending)
        2. Divide into 10 deciles
        3. Calculate TPR in top d deciles
        4. Calculate TPR in full dataset
        5. Return ratio
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_score : array-like of shape (n_samples,)
            Predicted scores (probabilities or decision function values).
        d : int
            Number of top deciles to consider (1-10).
        
        Returns
        -------
        float
            Decile Ranked Performance ratio.
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        
        # Validate d parameter
        if not 1 <= d <= 10:
            raise ValueError(f"Parameter 'd' must be between 1 and 10, got {d}")
        
        n_samples = len(y_true)
        
        # Sort by score descending
        sorted_indices = np.argsort(y_score)[::-1]
        sorted_y_true = y_true[sorted_indices]
        
        # Calculate decile size
        decile_size = n_samples / 10.0
        top_d_size = int(np.ceil(d * decile_size))
        
        # Get top d deciles
        top_d_labels = sorted_y_true[:top_d_size]
        
        # Calculate TPR in top d deciles
        tp_top_d = np.sum(top_d_labels == 1)
        total_positives_top_d = len(top_d_labels)
        tpr_top_d = tp_top_d / total_positives_top_d if total_positives_top_d > 0 else 0.0
        
        # Calculate TPR in entire dataset
        total_positives = np.sum(y_true == 1)
        tpr_full = total_positives / n_samples if n_samples > 0 else 0.0
        
        # Return ratio
        if tpr_full == 0:
            return 0.0
        
        return tpr_top_d / tpr_full
    
    @staticmethod
    def extract_decile_ranked_set(ids, y_score, d: int) -> Set[Any]:
        """
        Extract IDs from top d deciles.
        
        Parameters
        ----------
        ids : array-like of shape (n_samples,)
            ID values for each sample.
        y_score : array-like of shape (n_samples,)
            Predicted scores (probabilities or decision function values).
        d : int
            Number of top deciles to consider (1-10).
        
        Returns
        -------
        Set[Any]
            Set of IDs from top d deciles.
        """
        ids = np.asarray(ids)
        y_score = np.asarray(y_score)
        
        # Validate d parameter
        if not 1 <= d <= 10:
            raise ValueError(f"Parameter 'd' must be between 1 and 10, got {d}")
        
        n_samples = len(ids)
        
        # Sort by score descending
        sorted_indices = np.argsort(y_score)[::-1]
        
        # Calculate decile size
        decile_size = n_samples / 10.0
        top_d_size = int(np.ceil(d * decile_size))
        
        # Get IDs from top d deciles
        top_d_indices = sorted_indices[:top_d_size]
        top_d_ids = ids[top_d_indices]
        
        return set(top_d_ids)
    
    @staticmethod
    def extract_decile_positive_set(ids, y_true, y_score, d: int) -> Set[Any]:
        """
        Extract true positive IDs from top d deciles.
        
        Parameters
        ----------
        ids : array-like of shape (n_samples,)
            ID values for each sample.
        y_true : array-like of shape (n_samples,)
            True binary labels (0 or 1).
        y_score : array-like of shape (n_samples,)
            Predicted scores (probabilities or decision function values).
        d : int
            Number of top deciles to consider (1-10).
        
        Returns
        -------
        Set[Any]
            Set of true positive IDs from top d deciles.
        """
        ids = np.asarray(ids)
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        
        # Validate d parameter
        if not 1 <= d <= 10:
            raise ValueError(f"Parameter 'd' must be between 1 and 10, got {d}")
        
        n_samples = len(ids)
        
        # Sort by score descending
        sorted_indices = np.argsort(y_score)[::-1]
        
        # Calculate decile size
        decile_size = n_samples / 10.0
        top_d_size = int(np.ceil(d * decile_size))
        
        # Get top d deciles
        top_d_indices = sorted_indices[:top_d_size]
        
        # Filter for true positives only
        true_positive_mask = y_true[top_d_indices] == 1
        true_positive_ids = ids[top_d_indices][true_positive_mask]
        
        return set(true_positive_ids)
