"""Model scoring module for ELR."""
import pandas as pd
import numpy as np
from typing import Any
from ..utils.model_io import ModelIO


class ModelScorer:
    """
    Scores new datasets using previously trained ELR models.
    
    The ModelScorer loads model configurations from JSON files (modeljson format)
    and applies the saved equations to score new data. It supports all three
    ensemble methods (intersect, venn, ensemble) and produces output in the
    same format as the training workflow.
    
    For intersect/venn methods:
    - Applies each constituent model equation
    - Ranks predictions and filters to top d deciles
    - Compiles results across models
    - Returns DataFrame with IDs and occurrence counts
    
    For ensemble method:
    - Applies constituent model equations to generate probabilities
    - Applies meta-model equation to constituent probabilities
    - Returns DataFrame with IDs and predicted probabilities
    
    Parameters
    ----------
    modeljson_path : str
        Path to the model configuration JSON file.
    
    Attributes
    ----------
    config : Dict[str, Any]
        Loaded model configuration containing method, d, models, and meta_model.
    method : str
        Ensemble method: "intersect", "venn", or "ensemble".
    d : int
        Number of top deciles to consider (for intersect/venn methods).
    models : List[Dict[str, float]]
        List of equation dictionaries for constituent models.
    meta_model : Dict[str, float], optional
        Meta-model equation dictionary (only for ensemble method).
    
    Examples
    --------
    >>> scorer = ModelScorer('model_config.json')
    >>> predictions = scorer.score(X_new, ids_new)
    >>> print(predictions.head())
    """
    
    def __init__(self, modeljson_path: str):
        """
        Initialize ModelScorer by loading model configuration.
        
        Parameters
        ----------
        modeljson_path : str
            Path to the model configuration JSON file.
        
        Raises
        ------
        IOError
            If file cannot be read.
        ValueError
            If JSON structure is invalid.
        """
        # Load model configuration
        self.config = ModelIO.load_model(modeljson_path)
        
        # Extract configuration components
        self.method = self.config['method']
        self.d = self.config['d']
        self.models = self.config['models']
        self.meta_model = self.config.get('meta_model', None)
    
    def score(self, X, ids) -> pd.DataFrame:
        """
        Score dataset using loaded model configuration.
        
        The scoring process depends on the ensemble method:
        
        - For intersect/venn: Applies each constituent model equation,
          ranks predictions, filters to top d deciles, and compiles results
          with ID occurrence counts.
        
        - For ensemble: Applies constituent model equations to generate
          probabilities, then applies meta-model equation to produce
          final predictions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pd.DataFrame
            Feature matrix to score. If DataFrame, column names must match
            feature names in equation dictionaries.
        ids : array-like of shape (n_samples,)
            ID values for each sample.
        
        Returns
        -------
        pd.DataFrame
            For intersect/venn methods:
                - 'id': Deduplicated IDs from top d deciles
                - 'sets': Count of how many model DRS contain each ID
            
            For ensemble method:
                - 'id': All input IDs
                - 'predicted': Predicted probabilities from meta-model
        
        Raises
        ------
        ValueError
            If X is missing required features or has invalid shape.
        
        Examples
        --------
        >>> scorer = ModelScorer('intersect_model.json')
        >>> X_new = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        >>> ids_new = [101, 102]
        >>> results = scorer.score(X_new, ids_new)
        >>> print(results)
           id  sets
        0  101     2
        1  102     1
        """
        # Convert X to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Convert ids to array
        ids = np.array(ids)
        
        # Validate input
        if len(X) != len(ids):
            raise ValueError(
                f"X and ids must have same length. Got X: {len(X)}, ids: {len(ids)}"
            )
        
        # Route to appropriate scoring method
        if self.method in ['intersect', 'venn']:
            return self._score_intersect_venn(X, ids)
        elif self.method == 'ensemble':
            return self._score_ensemble(X, ids)
        else:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                "Expected 'intersect', 'venn', or 'ensemble'."
            )
    
    def _apply_logistic_regression(
        self,
        X: pd.DataFrame,
        equation_dict: dict
    ) -> np.ndarray:
        """
        Apply logistic regression equation to features.
        
        Formula: 1 / (1 + exp(-(constant + sum(feature_i * coef_i))))
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with column names matching equation dict keys.
        equation_dict : dict
            Dictionary mapping feature names to coefficients,
            with 'constant' key for intercept.
        
        Returns
        -------
        np.ndarray
            Predicted probabilities for positive class.
        
        Raises
        ------
        ValueError
            If required features are missing from X.
        """
        # Extract constant (intercept)
        constant = equation_dict.get('constant', 0.0)
        
        # Calculate linear combination
        linear_combination = np.full(len(X), constant)
        
        # Add contribution from each feature
        for feature_name, coef in equation_dict.items():
            if feature_name == 'constant':
                continue
            
            # Check if feature exists in X
            if feature_name not in X.columns:
                raise ValueError(
                    f"Required feature '{feature_name}' not found in input data. "
                    f"Available features: {list(X.columns)}"
                )
            
            # Add feature contribution
            linear_combination += X[feature_name].values * coef
        
        # Apply logistic function
        probabilities = 1.0 / (1.0 + np.exp(-linear_combination))
        
        return probabilities
    
    def _score_intersect_venn(self, X: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
        """
        Score dataset using intersect or venn method.
        
        Steps:
        1. Apply each constituent model equation
        2. Rank predictions and identify top d deciles for each model
        3. Compile IDs from top d deciles across all models
        4. Count occurrences of each ID
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to score.
        ids : np.ndarray
            ID values for each sample.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['id', 'sets'].
        """
        # Track IDs and their occurrence counts
        id_counts = {}
        
        # Process each constituent model
        for model_equation in self.models:
            # Apply equation to get predicted probabilities
            probabilities = self._apply_logistic_regression(X, model_equation)
            
            # Rank predictions (descending order)
            # Create DataFrame for sorting
            score_df = pd.DataFrame({
                'id': ids,
                'score': probabilities
            })
            
            # Sort by score descending
            score_df = score_df.sort_values('score', ascending=False).reset_index(drop=True)
            
            # Calculate decile boundaries
            n_samples = len(score_df)
            samples_per_decile = n_samples / 10.0
            
            # Identify top d deciles
            top_d_cutoff = int(np.ceil(self.d * samples_per_decile))
            
            # Extract IDs from top d deciles
            top_d_ids = score_df.iloc[:top_d_cutoff]['id'].values
            
            # Count occurrences
            for id_value in top_d_ids:
                id_counts[id_value] = id_counts.get(id_value, 0) + 1
        
        # Create result DataFrame
        if not id_counts:
            return pd.DataFrame(columns=['id', 'sets'])
        
        result_df = pd.DataFrame([
            {'id': id_value, 'sets': count}
            for id_value, count in id_counts.items()
        ])
        
        # Sort by sets descending, then by id for consistency
        result_df = result_df.sort_values(
            ['sets', 'id'],
            ascending=[False, True]
        ).reset_index(drop=True)
        
        return result_df
    
    def _score_ensemble(self, X: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
        """
        Score dataset using ensemble method.
        
        Steps:
        1. Apply each constituent model equation to generate probabilities
        2. Apply meta-model equation to constituent probabilities
        3. Return final predicted probabilities
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to score.
        ids : np.ndarray
            ID values for each sample.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['id', 'predicted'].
        """
        # Generate probabilities from each constituent model
        constituent_probabilities = []
        
        for model_equation in self.models:
            # Apply equation to get predicted probabilities
            probabilities = self._apply_logistic_regression(X, model_equation)
            constituent_probabilities.append(probabilities)
        
        # Stack into matrix: rows are samples, columns are model probabilities
        X_meta = np.column_stack(constituent_probabilities)
        
        # Create DataFrame with proper column names for meta-model
        meta_df = pd.DataFrame(
            X_meta,
            columns=[f"model_{i}_prob" for i in range(len(self.models))]
        )
        
        # Apply meta-model equation
        final_probabilities = self._apply_logistic_regression(
            meta_df,
            self.meta_model
        )
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'id': ids,
            'predicted': final_probabilities
        })
        
        return result_df
