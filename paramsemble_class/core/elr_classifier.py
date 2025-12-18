"""Main ELRClassifier for ensemble logistic regression."""
import logging
import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List
from sklearn.base import BaseEstimator, ClassifierMixin

from paramsemble_class.core.feature_sampler import FeatureSampler
from paramsemble_class.core.baseline_model import BaselineModel
from paramsemble_class.core.constituent_model import ConstituentModel
from paramsemble_class.ensemble.intersect import IntersectMethod
from paramsemble_class.ensemble.venn import VennMethod
from paramsemble_class.ensemble.ensemble import EnsembleMethod
from paramsemble_class.utils.validation import ParameterValidator
from paramsemble_class.utils.model_io import ModelIO


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ELRClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble Logistic Regression Classifier.
    
    ELR trains multiple logistic regression models on diverse feature subsets,
    establishes baseline performance using Random Forest, and provides three
    distinct ensemble strategies (intersect, venn, ensemble) for model selection
    and combination.
    
    The classifier follows scikit-learn conventions with fit/predict methods
    while extending functionality to support model persistence via JSON export,
    enabling SQL-based scoring in production environments.
    
    Parameters
    ----------
    m : int, default=100
        Number of feature combinations to generate.
    f : int, default=5
        Number of features per combination.
    sample : str, default="unique"
        Sampling method: "unique" (no replacement) or "replace" (with replacement).
    d : int, default=2
        Number of top deciles to consider (1-10).
    method : str, default="intersect"
        Ensemble method: "intersect", "venn", or "ensemble".
    spread : int, default=10
        Number of top models to select.
    solver : str, default="auto"
        Logistic regression solver or "auto" for automatic selection.
    id_column : str, default="id"
        Name of ID column in datasets.
    elr2json : str, optional
        Path to export all model metrics (JSON format).
    modeljson : str, optional
        Path to export selected model equations (JSON format).
    random_state : int, optional
        Random seed for reproducibility.
    
    Attributes
    ----------
    baseline_model_ : BaselineModel
        Trained baseline Random Forest model.
    constituent_models_ : List[ConstituentModel]
        List of trained constituent logistic regression models.
    constituent_results_ : List[Dict]
        Metrics for all constituent models.
    baseline_results_ : Dict
        Metrics for baseline model.
    selected_indices_ : List[int]
        Indices of selected constituent models.
    result_df_ : pd.DataFrame
        Final predictions/results based on ensemble method.
    meta_equation_ : Dict, optional
        Meta-model equation (only for ensemble method).
    feature_names_ : List[str]
        Names of features used in training.
    n_features_ : int
        Number of features in training data.
    
    Examples
    --------
    >>> from elr import ELRClassifier
    >>> import numpy as np
    >>> 
    >>> # Generate sample data
    >>> X_train = np.random.randn(100, 10)
    >>> y_train = np.random.randint(0, 2, 100)
    >>> X_test = np.random.randn(50, 10)
    >>> y_test = np.random.randint(0, 2, 50)
    >>> ids_test = np.arange(50)
    >>> 
    >>> # Train ELR classifier
    >>> clf = ELRClassifier(m=20, f=3, method='intersect', spread=5)
    >>> clf.fit(X_train, y_train, X_test, y_test, ids_test)
    >>> 
    >>> # Get predictions
    >>> predictions = clf.predict(X_test, ids_test)
    """
    
    def __init__(
        self,
        m: int = 100,
        f: int = 5,
        sample: str = "unique",
        d: int = 2,
        method: str = "intersect",
        spread: int = 10,
        solver: str = "auto",
        id_column: str = "id",
        elr2json: Optional[str] = None,
        modeljson: Optional[str] = None,
        random_state: Optional[int] = None
    ):
        """Initialize ELR Classifier."""
        self.m = m
        self.f = f
        self.sample = sample
        self.d = d
        self.method = method
        self.spread = spread
        self.solver = solver
        self.id_column = id_column
        self.elr2json = elr2json
        self.modeljson = modeljson
        self.random_state = random_state
        
        # Attributes set during fit
        self.baseline_model_ = None
        self.constituent_models_ = None
        self.constituent_results_ = None
        self.baseline_results_ = None
        self.selected_indices_ = None
        self.result_df_ = None
        self.meta_equation_ = None
        self.feature_names_ = None
        self.n_features_ = None
    
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        ids_test: Union[np.ndarray, pd.Series, List]
    ):
        """
        Train ELR classifier.
        
        This method orchestrates the entire training workflow:
        1. Validates inputs
        2. Generates feature combinations
        3. Trains baseline Random Forest
        4. Trains m logistic regression models
        5. Applies ensemble method
        6. Exports JSON if specified
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training target labels.
        X_test : array-like of shape (n_samples, n_features)
            Test feature matrix.
        y_test : array-like of shape (n_samples,)
            Test target labels.
        ids_test : array-like of shape (n_samples,)
            ID values for test samples.
        
        Returns
        -------
        self : ELRClassifier
            Fitted classifier.
        
        Raises
        ------
        ValueError
            If parameters or data are invalid.
        """
        logger.info("Starting ELR classifier training...")
        
        # Step 1: Validate parameters
        logger.info("Validating parameters...")
        params = {
            'm': self.m,
            'f': self.f,
            'sample': self.sample,
            'd': self.d,
            'method': self.method,
            'spread': self.spread,
            'solver': self.solver
        }
        ParameterValidator.validate_parameters(params)
        
        # Step 2: Prepare data
        logger.info("Preparing data...")
        X_train_array, y_train_array, feature_names = self._prepare_data(X_train, y_train)
        X_test_array, y_test_array, _ = self._prepare_data(X_test, y_test, feature_names)
        ids_test_array = np.array(ids_test)
        
        # Validate data
        ParameterValidator.validate_data(X_train_array, y_train_array)
        ParameterValidator.validate_data(X_test_array, y_test_array)
        
        # Store feature information
        self.feature_names_ = feature_names
        self.n_features_ = X_train_array.shape[1]
        
        # Validate f against number of features
        ParameterValidator.validate_f_against_features(self.f, self.n_features_)
        
        # Step 3: Generate feature combinations
        logger.info(f"Generating {self.m} feature combinations with f={self.f}, sample={self.sample}...")
        sampler = FeatureSampler(
            n_features=self.n_features_,
            f=self.f,
            m=self.m,
            sample=self.sample,
            random_state=self.random_state
        )
        feature_combinations = sampler.generate_combinations()
        actual_m = len(feature_combinations)
        
        if actual_m < self.m:
            logger.warning(
                f"Generated {actual_m} combinations (less than requested {self.m}) "
                f"due to maximum combinations limit."
            )
        else:
            logger.info(f"Generated {actual_m} feature combinations.")
        
        # Step 4: Train baseline model
        logger.info("Training baseline Random Forest model...")
        self.baseline_model_ = BaselineModel(random_state=self.random_state)
        self.baseline_model_.fit(X_train_array, y_train_array)
        self.baseline_results_ = self.baseline_model_.evaluate(
            X_test_array, y_test_array, ids_test_array, self.d
        )
        logger.info(
            f"Baseline model - PLR: {self.baseline_results_['plr']:.3f}, "
            f"FNR: {self.baseline_results_['fnr']:.3f}, "
            f"DRP: {self.baseline_results_['drp']:.3f}"
        )
        
        # Step 5: Train constituent models
        logger.info(f"Training {actual_m} constituent logistic regression models...")
        self.constituent_models_ = []
        self.constituent_results_ = []
        
        successful_models = 0
        failed_models = 0
        
        for i, feature_indices in enumerate(feature_combinations):
            try:
                # Create and train constituent model
                model = ConstituentModel(
                    feature_indices=feature_indices,
                    solver=self.solver,
                    random_state=self.random_state
                )
                model.fit(X_train_array, y_train_array)
                
                # Evaluate model
                metrics = model.evaluate(X_test_array, y_test_array, ids_test_array, self.d)
                
                # Get equation dictionary
                equation_dict = model.get_equation_dict(self.feature_names_)
                
                # Store model and results
                self.constituent_models_.append(model)
                self.constituent_results_.append({
                    'plr': metrics['plr'],
                    'fnr': metrics['fnr'],
                    'drp': metrics['drp'],
                    'drs': metrics['drs'],
                    'dps': metrics['dps'],
                    'equation_dict': equation_dict,
                    'feature_indices': feature_indices
                })
                
                successful_models += 1
                
            except Exception as e:
                logger.warning(f"Failed to train model {i+1}/{actual_m}: {str(e)}")
                failed_models += 1
                continue
        
        logger.info(
            f"Successfully trained {successful_models} models "
            f"({failed_models} failed)."
        )
        
        if successful_models == 0:
            raise ValueError(
                "All constituent models failed to train. "
                "Please check your data and parameters."
            )
        
        # Step 6: Apply ensemble method
        logger.info(f"Applying {self.method} ensemble method with spread={self.spread}...")
        self._apply_ensemble_method(X_test_array, y_test_array, ids_test_array)
        
        # Step 7: Export JSON if specified
        if self.elr2json:
            logger.info(f"Exporting all model metrics to {self.elr2json}...")
            ModelIO.export_all_models(self.constituent_results_, self.elr2json)
        
        if self.modeljson:
            logger.info(f"Exporting selected model equations to {self.modeljson}...")
            self._export_selected_models()
        
        logger.info("ELR classifier training complete!")
        
        return self
    
    def _prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None
    ):
        """
        Prepare data for training/testing.
        
        Converts DataFrames to arrays and extracts feature names.
        
        Parameters
        ----------
        X : array-like or DataFrame
            Feature matrix.
        y : array-like or Series
            Target labels.
        feature_names : List[str], optional
            Expected feature names (for validation).
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, List[str]]
            (X_array, y_array, feature_names)
        """
        # Convert X to array
        if isinstance(X, pd.DataFrame):
            # Extract numeric columns only
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            X_array = X[numeric_cols].values
            extracted_feature_names = numeric_cols
        else:
            X_array = np.array(X)
            if feature_names is None:
                extracted_feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
            else:
                extracted_feature_names = feature_names
        
        # Convert y to array
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = np.array(y)
        
        # Validate feature names match if provided
        if feature_names is not None:
            if extracted_feature_names != feature_names:
                raise ValueError(
                    f"Feature names mismatch. Expected {feature_names}, "
                    f"got {extracted_feature_names}"
                )
        
        return X_array, y_array, extracted_feature_names
    
    def _apply_ensemble_method(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ids_test: np.ndarray
    ):
        """
        Apply the selected ensemble method.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test feature matrix.
        y_test : np.ndarray
            Test target labels.
        ids_test : np.ndarray
            Test IDs.
        """
        if self.method == "intersect":
            self.result_df_ = IntersectMethod.select_and_combine(
                self.constituent_results_,
                self.baseline_results_,
                self.spread
            )
            # Store selected indices for export
            ranked_indices = IntersectMethod._rank_models(
                self.constituent_results_,
                self.baseline_results_
            )
            self.selected_indices_ = ranked_indices[:min(self.spread, len(ranked_indices))]
            
        elif self.method == "venn":
            self.result_df_ = VennMethod.select_and_combine(
                self.constituent_results_,
                self.baseline_results_,
                self.spread
            )
            # For venn, we need to track which models were undiscarded
            # This is done internally by the method, so we'll extract it
            ranked_indices = VennMethod._rank_models(
                self.constituent_results_,
                self.baseline_results_
            )
            initial_selection = ranked_indices[:min(2 * self.spread, len(ranked_indices))]
            
            # Determine undiscarded models
            baseline_dps = self.baseline_results_['dps']
            incremental_id_set = set(baseline_dps)
            undiscarded = []
            
            for idx in initial_selection:
                model_dps = self.constituent_results_[idx]['dps']
                unique_ids = model_dps - incremental_id_set
                if unique_ids:
                    undiscarded.append(idx)
                    incremental_id_set.update(unique_ids)
            
            self.selected_indices_ = undiscarded
            
        elif self.method == "ensemble":
            self.result_df_, self.meta_equation_, self.selected_indices_ = EnsembleMethod.select_and_combine(
                self.constituent_models_,
                X_test,
                y_test,
                ids_test,
                self.constituent_results_,
                self.baseline_results_,
                self.spread,
                self.random_state
            )
        
        logger.info(f"Selected {len(self.selected_indices_)} models for {self.method} method.")
    
    def _export_selected_models(self):
        """Export selected model equations to JSON."""
        # Get equation dictionaries for selected models
        selected_equations = [
            self.constituent_results_[idx]['equation_dict']
            for idx in self.selected_indices_
        ]
        
        # Export based on method
        ModelIO.export_selected_models(
            method=self.method,
            d=self.d,
            selected_models=selected_equations,
            meta_model=self.meta_equation_ if self.method == "ensemble" else None,
            filepath=self.modeljson
        )
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        ids: Optional[Union[np.ndarray, pd.Series, List]] = None
    ) -> pd.DataFrame:
        """
        Generate predictions using trained ensemble.
        
        The output format depends on the ensemble method:
        - intersect/venn: DataFrame with 'id' and 'sets' columns
        - ensemble: DataFrame with 'id' and 'predicted' columns
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        ids : array-like of shape (n_samples,), optional
            ID values for samples. If not provided, uses indices.
        
        Returns
        -------
        pd.DataFrame
            Predictions in method-appropriate format.
        
        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if self.constituent_models_ is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )
        
        # Prepare data
        X_array, _, _ = self._prepare_data(X, np.zeros(len(X)), self.feature_names_)
        
        # Use indices if IDs not provided
        if ids is None:
            ids = np.arange(len(X))
        else:
            ids = np.array(ids)
        
        # Generate predictions based on method
        if self.method in ["intersect", "venn"]:
            # For intersect/venn, we need to score with each selected model
            # and apply the same logic as in training
            id_counts = {}
            
            for idx in self.selected_indices_:
                model = self.constituent_models_[idx]
                # Get probability scores
                y_score = model.predict_proba(X_array)[:, 1]
                
                # Sort and get top d deciles
                n_samples = len(y_score)
                decile_size = n_samples // 10
                sorted_indices = np.argsort(y_score)[::-1]
                top_d_indices = sorted_indices[:decile_size * self.d]
                
                # Get IDs in top d deciles
                top_d_ids = ids[top_d_indices]
                
                # Count occurrences
                for id_value in top_d_ids:
                    id_counts[id_value] = id_counts.get(id_value, 0) + 1
            
            # Create DataFrame
            if not id_counts:
                return pd.DataFrame(columns=['id', 'sets'])
            
            result_df = pd.DataFrame([
                {'id': id_value, 'sets': count}
                for id_value, count in id_counts.items()
            ])
            result_df = result_df.sort_values(['sets', 'id'], ascending=[False, True]).reset_index(drop=True)
            
            return result_df
            
        else:  # ensemble method
            # Generate predictions from each selected constituent model
            meta_features = []
            
            for idx in self.selected_indices_:
                model = self.constituent_models_[idx]
                proba = model.predict_proba(X_array)[:, 1]
                meta_features.append(proba)
            
            # Stack into matrix
            X_meta = np.column_stack(meta_features)
            
            # Apply meta-model equation manually
            # meta_equation has keys like "model_0_prob", "model_1_prob", ..., "constant"
            prediction = np.full(len(X), self.meta_equation_['constant'])
            
            for i in range(len(self.selected_indices_)):
                coef = self.meta_equation_[f'model_{i}_prob']
                prediction += coef * X_meta[:, i]
            
            # Apply logistic function
            final_proba = 1 / (1 + np.exp(-prediction))
            
            # Create DataFrame
            result_df = pd.DataFrame({
                'id': ids,
                'predicted': final_proba
            })
            
            return result_df
    
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Generate probability predictions.
        
        For ensemble method, returns predicted probabilities.
        For intersect/venn methods, this is not applicable and raises an error.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        
        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Predicted probabilities for each class.
        
        Raises
        ------
        ValueError
            If model has not been fitted or method is not ensemble.
        """
        if self.constituent_models_ is None:
            raise ValueError(
                "Model must be fitted before prediction. Call fit() first."
            )
        
        if self.method != "ensemble":
            raise ValueError(
                f"predict_proba is only available for ensemble method, "
                f"current method is '{self.method}'."
            )
        
        # Prepare data
        X_array, _, _ = self._prepare_data(X, np.zeros(len(X)), self.feature_names_)
        
        # Generate predictions from each selected constituent model
        meta_features = []
        
        for idx in self.selected_indices_:
            model = self.constituent_models_[idx]
            proba = model.predict_proba(X_array)[:, 1]
            meta_features.append(proba)
        
        # Stack into matrix
        X_meta = np.column_stack(meta_features)
        
        # Apply meta-model equation
        prediction = np.full(len(X), self.meta_equation_['constant'])
        
        for i in range(len(self.selected_indices_)):
            coef = self.meta_equation_[f'model_{i}_prob']
            prediction += coef * X_meta[:, i]
        
        # Apply logistic function for positive class probability
        proba_positive = 1 / (1 + np.exp(-prediction))
        proba_negative = 1 - proba_positive
        
        # Return as (n_samples, 2) array
        return np.column_stack([proba_negative, proba_positive])
