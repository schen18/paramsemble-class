"""Baseline model using Random Forest for ELR comparison."""
import numpy as np
from typing import Dict, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from paramsemble_class.metrics.performance import PerformanceMetrics


class BaselineModel:
    """
    Baseline Random Forest model for establishing performance benchmarks.
    
    The baseline model trains a Random Forest classifier on all features
    and calculates specialized metrics (PLR, FNR, DRP, DRS, DPS) for
    comparison with constituent logistic regression models.
    
    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility.
    
    Attributes
    ----------
    model_ : RandomForestClassifier
        The trained Random Forest classifier.
    metrics_ : Dict
        Dictionary containing baseline metrics after evaluation.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize baseline Random Forest model.
        
        Parameters
        ----------
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.random_state = random_state
        self.model_ = None
        self.metrics_ = None
    
    def fit(self, X_train, y_train):
        """
        Train Random Forest classifier on all features.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training target labels.
        
        Returns
        -------
        self : BaselineModel
            Fitted baseline model.
        """
        self.model_ = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        self.model_.fit(X_train, y_train)
        return self
    
    def evaluate(self, X_test, y_test, ids, d: int) -> Dict[str, Any]:
        """
        Evaluate baseline model on test data.
        
        Calculates all specialized metrics:
        - PLR: Positive Likelihood Ratio
        - FNR: False Negative Rate
        - DRP: Decile Ranked Performance
        - DRS: Decile Ranked Set (IDs in top d deciles)
        - DPS: Decile Positive Set (true positive IDs in top d deciles)
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test feature matrix.
        y_test : array-like of shape (n_samples,)
            Test target labels.
        ids : array-like of shape (n_samples,)
            ID values for each test sample.
        d : int
            Number of top deciles to consider (1-10).
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'plr': Positive Likelihood Ratio
            - 'fnr': False Negative Rate
            - 'drp': Decile Ranked Performance
            - 'drs': Decile Ranked Set (set of IDs)
            - 'dps': Decile Positive Set (set of true positive IDs)
        
        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before evaluation. Call fit() first.")
        
        # Generate predictions
        y_pred = self.model_.predict(X_test)
        
        # Generate probability scores for ranking
        y_score = self.model_.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        plr = PerformanceMetrics.positive_likelihood_ratio(y_test, y_pred)
        fnr = PerformanceMetrics.false_negative_rate(y_test, y_pred)
        drp = PerformanceMetrics.decile_ranked_performance(y_test, y_score, d)
        drs = PerformanceMetrics.extract_decile_ranked_set(ids, y_score, d)
        dps = PerformanceMetrics.extract_decile_positive_set(ids, y_test, y_score, d)
        
        # Store metrics
        self.metrics_ = {
            'plr': plr,
            'fnr': fnr,
            'drp': drp,
            'drs': drs,
            'dps': dps
        }
        
        return self.metrics_
    
    def predict(self, X):
        """
        Generate predictions for new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        
        Returns
        -------
        array-like of shape (n_samples,)
            Predicted labels.
        
        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        """
        Generate probability predictions for new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        
        Returns
        -------
        array-like of shape (n_samples, 2)
            Predicted probabilities for each class.
        
        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.model_ is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        return self.model_.predict_proba(X)
