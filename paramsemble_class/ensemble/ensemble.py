"""Ensemble ensemble method for ELR."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression


class EnsembleMethod:
    """
    Ensemble ensemble method for creating a meta-model from top performers.
    
    The ensemble method combines predictions from top-performing constituent
    models by training a meta-model (logistic regression) that uses their
    predicted probabilities as features.
    
    The method:
    1. Ranks models by PLR (higher better), FNR (lower better), DRP (higher better)
    2. Selects top n models based on spread parameter
    3. Filters models that outperform baseline
    4. Scores test set with each selected model to generate predicted probabilities
    5. Trains meta-model using constituent predictions as features
    6. Generates final predictions from meta-model
    
    Returns a DataFrame with IDs and predicted probabilities, plus meta-model equation.
    """
    
    @staticmethod
    def _rank_models(
        constituent_results: List[Dict[str, Any]],
        baseline_results: Dict[str, Any]
    ) -> List[int]:
        """
        Rank constituent models by performance metrics.
        
        Ranking criteria (in order of priority):
        1. PLR: Higher is better
        2. FNR: Lower is better
        3. DRP: Higher is better
        
        Only models that outperform baseline are considered.
        
        Parameters
        ----------
        constituent_results : List[Dict[str, Any]]
            List of dictionaries containing metrics for each constituent model.
            Each dict must have keys: 'plr', 'fnr', 'drp', 'drs', 'dps'
        baseline_results : Dict[str, Any]
            Dictionary containing baseline model metrics.
            Must have keys: 'plr', 'fnr', 'drp', 'drs', 'dps'
        
        Returns
        -------
        List[int]
            Indices of models sorted by rank (best first).
        """
        # Filter models that outperform baseline
        # A model outperforms baseline if:
        # - PLR is higher OR
        # - FNR is lower OR
        # - DRP is higher
        outperforming_indices = []
        
        baseline_plr = baseline_results['plr']
        baseline_fnr = baseline_results['fnr']
        baseline_drp = baseline_results['drp']
        
        for i, result in enumerate(constituent_results):
            # Check if model outperforms baseline on any metric
            if (result['plr'] > baseline_plr or
                result['fnr'] < baseline_fnr or
                result['drp'] > baseline_drp):
                outperforming_indices.append(i)
        
        # If no models outperform baseline, return empty list
        if not outperforming_indices:
            return []
        
        # Create ranking scores for outperforming models
        # We'll use a composite score: higher PLR, lower FNR, higher DRP
        scores = []
        for idx in outperforming_indices:
            result = constituent_results[idx]
            # Composite score: PLR (positive) - FNR (negative) + DRP (positive)
            # Handle inf values in PLR
            plr_score = result['plr'] if not np.isinf(result['plr']) else 1000.0
            score = plr_score - result['fnr'] + result['drp']
            scores.append((idx, score))
        
        # Sort by score descending (higher is better)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return sorted indices
        return [idx for idx, _ in scores]
    
    @staticmethod
    def select_and_combine(
        constituent_models: List[Any],  # List of ConstituentModel objects
        X_test,
        y_test,
        ids,
        constituent_results: List[Dict[str, Any]],
        baseline_results: Dict[str, Any],
        spread: int,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float], List[int]]:
        """
        Select top models and train meta-model for ensemble predictions.
        
        This method selects the top n constituent models, generates predicted
        probabilities from each on the test set, then trains a meta-model
        logistic regression using these probabilities as features.
        
        Parameters
        ----------
        constituent_models : List[ConstituentModel]
            List of trained ConstituentModel objects.
        X_test : array-like of shape (n_samples, n_features)
            Test feature matrix.
        y_test : array-like of shape (n_samples,)
            Test target labels.
        ids : array-like of shape (n_samples,)
            ID values for each test sample.
        constituent_results : List[Dict[str, Any]]
            List of dictionaries containing metrics for each constituent model.
            Each dict must have keys: 'plr', 'fnr', 'drp', 'drs', 'dps'
        baseline_results : Dict[str, Any]
            Dictionary containing baseline model metrics.
            Must have keys: 'plr', 'fnr', 'drp', 'drs', 'dps'
        spread : int
            Number of top models to select.
        random_state : int, optional
            Random seed for meta-model training.
        
        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, float], List[int]]
            - DataFrame with columns:
                - 'id': Test set IDs
                - 'predicted': Predicted probabilities from meta-model
            - Meta-model equation dictionary:
                - Keys: "model_0_prob", "model_1_prob", ..., "constant"
                - Values: Coefficients and intercept
            - List of selected model indices
        """
        # Rank models
        ranked_indices = EnsembleMethod._rank_models(
            constituent_results,
            baseline_results
        )
        
        # Select top n models (or all if fewer than n outperform baseline)
        n_to_select = min(spread, len(ranked_indices))
        selected_indices = ranked_indices[:n_to_select]
        
        # If no models selected, return empty results
        if not selected_indices:
            empty_df = pd.DataFrame({
                'id': ids,
                'predicted': np.zeros(len(ids))
            })
            return empty_df, {"constant": 0.0}, []
        
        # Generate predicted probabilities from each selected model
        # Shape: (n_samples, n_selected_models)
        meta_features = []
        
        for idx in selected_indices:
            model = constituent_models[idx]
            # Get probability of positive class
            proba = model.predict_proba(X_test)[:, 1]
            meta_features.append(proba)
        
        # Stack into matrix: rows are samples, columns are model probabilities
        X_meta = np.column_stack(meta_features)
        
        # Train meta-model with lbfgs solver
        meta_model = LogisticRegression(
            solver='lbfgs',
            random_state=random_state,
            max_iter=1000
        )
        meta_model.fit(X_meta, y_test)
        
        # Generate final predictions (probabilities of positive class)
        final_predictions = meta_model.predict_proba(X_meta)[:, 1]
        
        # Create output DataFrame
        result_df = pd.DataFrame({
            'id': ids,
            'predicted': final_predictions
        })
        
        # Extract meta-model equation dictionary
        meta_equation = {}
        
        # Add coefficients for each constituent model's probability
        for i, coef in enumerate(meta_model.coef_[0]):
            meta_equation[f"model_{i}_prob"] = float(coef)
        
        # Add intercept as "constant"
        meta_equation["constant"] = float(meta_model.intercept_[0])
        
        return result_df, meta_equation, selected_indices
