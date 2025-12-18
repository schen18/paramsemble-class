# Design Document

## Overview

The ELR (Ensemble Logistic Regression) package is a scikit-learn compatible Python library that implements an advanced ensemble classification approach based on combinatorial feature selection. The system trains multiple logistic regression models on diverse feature subsets, establishes baseline performance using Random Forest, and provides three distinct ensemble strategies (intersect, venn, ensemble) for model selection and combination.

The architecture follows scikit-learn conventions with fit/predict methods while extending functionality to support model persistence via JSON export, enabling SQL-based scoring in production environments. The package emphasizes interpretability through coefficient extraction and provides specialized metrics (Positive Likelihood Ratio, False Negative Rate, Decile Ranked Performance) tailored for imbalanced classification tasks.

## Architecture

The ELR package follows a modular architecture with clear separation of concerns:

```
elr/
├── core/
│   ├── __init__.py
│   ├── elr_classifier.py      # Main ELRClassifier class
│   └── feature_sampler.py     # Feature combination generation
├── metrics/
│   ├── __init__.py
│   └── performance.py         # PLR, FNR, DRP calculations
├── ensemble/
│   ├── __init__.py
│   ├── intersect.py           # Intersect method implementation
│   ├── venn.py                # Venn method implementation
│   └── ensemble.py            # Ensemble method implementation
├── scoring/
│   ├── __init__.py
│   └── scorer.py              # Model scoring from JSON
├── sql/
│   ├── __init__.py
│   └── generator.py           # SQL generation from JSON
├── utils/
│   ├── __init__.py
│   ├── validation.py          # Parameter validation
│   └── model_io.py            # JSON export/import
└── __init__.py
```

### Key Design Principles

1. **Scikit-learn Compatibility**: Implement BaseEstimator and ClassifierMixin for seamless integration
2. **Separation of Concerns**: Isolate feature sampling, metric calculation, ensemble methods, and scoring
3. **Extensibility**: Allow easy addition of new ensemble methods or metrics
4. **Reproducibility**: Support random state parameters for deterministic results
5. **Performance**: Leverage numpy/pandas vectorization and parallel processing where applicable

## Components and Interfaces

### ELRClassifier

The main class that orchestrates the entire training and prediction workflow.

```python
class ELRClassifier(BaseEstimator, ClassifierMixin):
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
        """
        Initialize ELR Classifier.
        
        Parameters:
        - m: Number of feature combinations to generate
        - f: Number of features per combination
        - sample: "unique" or "replace" for feature sampling
        - d: Number of top deciles to consider
        - method: "intersect", "venn", or "ensemble"
        - spread: Number of top models to select
        - solver: Logistic regression solver or "auto"
        - id_column: Name of ID column in datasets
        - elr2json: Path to export all model metrics
        - modeljson: Path to export selected model equations
        - random_state: Random seed for reproducibility
        """
        
    def fit(self, X_train, y_train, X_test, y_test):
        """
        Train ELR classifier.
        
        Steps:
        1. Validate inputs
        2. Generate feature combinations
        3. Train baseline Random Forest
        4. Train m logistic regression models
        5. Apply ensemble method
        6. Export JSON if specified
        
        Returns: self
        """
        
    def predict(self, X):
        """
        Generate predictions using trained ensemble.
        
        Returns: Predictions based on method type
        """
        
    def predict_proba(self, X):
        """
        Generate probability predictions.
        
        Returns: Probability estimates
        """
```

### FeatureSampler

Generates diverse feature combinations for model training.

```python
class FeatureSampler:
    def __init__(self, n_features: int, f: int, m: int, sample: str, random_state: Optional[int] = None):
        """
        Initialize feature sampler.
        
        Parameters:
        - n_features: Total number of features available
        - f: Features per combination
        - m: Number of combinations to generate
        - sample: "unique" or "replace"
        - random_state: Random seed
        """
        
    def generate_combinations(self) -> List[List[int]]:
        """
        Generate m feature combinations.
        
        Returns: List of feature index lists
        """
        
    def _calculate_max_combinations(self) -> int:
        """
        Calculate maximum possible combinations.
        
        For unique: C(n_features, f)
        For replace: n_features^f
        
        Returns: Maximum combinations
        """
```

### PerformanceMetrics

Calculates specialized classification metrics.

```python
class PerformanceMetrics:
    @staticmethod
    def positive_likelihood_ratio(y_true, y_pred) -> float:
        """
        Calculate PLR = TPR / FPR.
        
        Returns: Positive Likelihood Ratio
        """
        
    @staticmethod
    def false_negative_rate(y_true, y_pred) -> float:
        """
        Calculate FNR = FN / (FN + TP).
        
        Returns: False Negative Rate
        """
        
    @staticmethod
    def decile_ranked_performance(y_true, y_score, d: int) -> float:
        """
        Calculate DRP.
        
        Steps:
        1. Sort predictions descending
        2. Divide into 10 deciles
        3. Calculate TPR in top d deciles
        4. Calculate TPR in full dataset
        5. Return ratio
        
        Returns: Decile Ranked Performance
        """
        
    @staticmethod
    def extract_decile_ranked_set(ids, y_score, d: int) -> Set:
        """
        Extract IDs from top d deciles.
        
        Returns: Set of IDs
        """
        
    @staticmethod
    def extract_decile_positive_set(ids, y_true, y_score, d: int) -> Set:
        """
        Extract true positive IDs from top d deciles.
        
        Returns: Set of true positive IDs
        """
```

### BaselineModel

Trains and evaluates Random Forest baseline.

```python
class BaselineModel:
    def __init__(self, random_state: Optional[int] = None):
        """Initialize baseline Random Forest model."""
        
    def fit(self, X_train, y_train):
        """Train Random Forest on all features."""
        
    def evaluate(self, X_test, y_test, ids, d: int) -> Dict:
        """
        Evaluate baseline model.
        
        Returns: Dict with PLR, FNR, DRP, DRS, DPS
        """
```

### ConstituentModel

Represents a single logistic regression trained on a feature subset.

```python
class ConstituentModel:
    def __init__(self, feature_indices: List[int], solver: str, random_state: Optional[int] = None):
        """
        Initialize constituent logistic regression model.
        
        Parameters:
        - feature_indices: Indices of features to use
        - solver: Scikit-learn solver name
        - random_state: Random seed
        """
        
    def fit(self, X_train, y_train):
        """Train logistic regression on feature subset."""
        
    def evaluate(self, X_test, y_test, ids, d: int) -> Dict:
        """
        Evaluate model and extract metrics.
        
        Returns: Dict with PLR, FNR, DRP, DRS, DPS, equation_dict
        """
        
    def get_equation_dict(self, feature_names: List[str]) -> Dict:
        """
        Extract coefficients as dictionary.
        
        Returns: {"feature1": coef1, ..., "constant": intercept}
        """
```

### EnsembleMethods

Implements the three ensemble strategies.

```python
class IntersectMethod:
    def select_and_combine(
        self,
        constituent_results: List[Dict],
        baseline_results: Dict,
        spread: int
    ) -> pd.DataFrame:
        """
        Implement intersect method.
        
        Steps:
        1. Rank models by PLR, FNR, DRP
        2. Select top n models (spread parameter)
        3. Compile DRS from selected models
        4. Count occurrences per ID
        
        Returns: DataFrame with columns [id, sets]
        """

class VennMethod:
    def select_and_combine(
        self,
        constituent_results: List[Dict],
        baseline_results: Dict,
        spread: int
    ) -> pd.DataFrame:
        """
        Implement venn method.
        
        Steps:
        1. Rank models by PLR, FNR, DRP
        2. Initially select top 2×n models
        3. Filter models with unique DPS vs baseline
        4. Compile DRS from undiscarded models
        5. Count occurrences per ID
        
        Returns: DataFrame with columns [id, sets]
        """

class EnsembleMethod:
    def select_and_combine(
        self,
        constituent_models: List[ConstituentModel],
        X_test,
        y_test,
        baseline_results: Dict,
        spread: int,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Implement ensemble method.
        
        Steps:
        1. Rank models by PLR, FNR, DRP
        2. Select top n models
        3. Generate predicted probabilities from each
        4. Train meta-model with lbfgs solver
        5. Generate final predictions
        
        Returns: (DataFrame with [id, predicted], meta_model_equation)
        """
```

### ModelScorer

Scores new data using saved model JSON.

```python
class ModelScorer:
    def __init__(self, modeljson_path: str):
        """Load model configuration from JSON."""
        
    def score(self, X, ids) -> pd.DataFrame:
        """
        Score dataset using loaded models.
        
        For intersect/venn:
        - Apply each equation dictionary
        - Rank and filter to top d deciles
        - Compile results
        
        For ensemble:
        - Apply constituent equations
        - Apply meta-model equation
        - Return predictions
        
        Returns: Predictions in method-appropriate format
        """
```

### Validation

Validates all input parameters and data.

```python
class ParameterValidator:
    @staticmethod
    def validate_parameters(params: Dict) -> None:
        """
        Validate all ELR parameters.
        
        Raises: ValueError with descriptive message
        """
        
    @staticmethod
    def validate_data(X, y, id_column: str) -> None:
        """
        Validate training/test data.
        
        Checks:
        - No missing values in features
        - ID column exists
        - Shapes match
        
        Raises: ValueError with descriptive message
        """
```

### ModelIO

Handles JSON export and import.

```python
class ModelIO:
    @staticmethod
    def export_all_models(models_data: List[Dict], filepath: str) -> None:
        """
        Export all model metrics to JSON (elr2json).
        
        Format: List of dicts with PLR, FNR, DRP, equation_dict
        """
        
    @staticmethod
    def export_selected_models(
        method: str,
        d: int,
        selected_models: List[Dict],
        meta_model: Optional[Dict],
        filepath: str
    ) -> None:
        """
        Export selected models to JSON (modeljson).
        
        Format: {
            "method": str,
            "d": int,
            "models": [equation_dicts],
            "meta_model": equation_dict (for ensemble only)
        }
        """
        
    @staticmethod
    def load_model(filepath: str) -> Dict:
        """Load model configuration from JSON."""
```

### SQLGenerator

Converts model JSON to executable SQL queries.

```python
class SQLGenerator:
    def __init__(self, modeljson_path: str):
        """
        Initialize SQL generator with model JSON.
        
        Parameters:
        - modeljson_path: Path to model JSON file
        """
        
    def generate_sql(self, table_name: str, id_column: str = "id") -> str:
        """
        Generate SQL query for model scoring.
        
        Parameters:
        - table_name: Name of database table to score
        - id_column: Name of ID column in table
        
        Returns: Complete SQL query string
        """
        
    def _generate_logistic_regression_sql(
        self,
        equation_dict: Dict,
        feature_columns: List[str],
        cte_name: str
    ) -> str:
        """
        Generate SQL for a single logistic regression equation.
        
        Formula: 1 / (1 + EXP(-(constant + feature1*coef1 + feature2*coef2 + ...)))
        
        Returns: CTE SQL string
        """
        
    def _generate_intersect_sql(
        self,
        table_name: str,
        id_column: str,
        constituent_ctes: List[str],
        d: int
    ) -> str:
        """
        Generate SQL for intersect method.
        
        Steps:
        1. Create CTEs for each constituent model
        2. Rank predictions and identify top d deciles for each
        3. Aggregate IDs across models
        4. Count occurrences per ID
        
        Returns: Complete SQL query
        """
        
    def _generate_venn_sql(
        self,
        table_name: str,
        id_column: str,
        constituent_ctes: List[str],
        d: int
    ) -> str:
        """
        Generate SQL for venn method.
        
        Steps:
        1. Create CTEs for each constituent model
        2. Rank predictions and identify top d deciles for each
        3. Aggregate IDs across models
        4. Count occurrences per ID
        
        Returns: Complete SQL query
        """
        
    def _generate_ensemble_sql(
        self,
        table_name: str,
        id_column: str,
        constituent_ctes: List[str],
        meta_model_equation: Dict
    ) -> str:
        """
        Generate SQL for ensemble method.
        
        Steps:
        1. Create CTEs for each constituent model
        2. Join all constituent predictions
        3. Apply meta-model equation to constituent probabilities
        4. Return final probability scores
        
        Returns: Complete SQL query
        """
```

## Data Models

### Model Metrics Dictionary

```python
{
    "plr": float,              # Positive Likelihood Ratio
    "fnr": float,              # False Negative Rate
    "drp": float,              # Decile Ranked Performance
    "drs": Set[Any],           # Decile Ranked Set (IDs)
    "dps": Set[Any],           # Decile Positive Set (IDs)
    "equation_dict": Dict,     # Feature coefficients
    "feature_indices": List[int]  # Features used
}
```

### Equation Dictionary

```python
{
    "feature_name_1": float,   # Coefficient for feature 1
    "feature_name_2": float,   # Coefficient for feature 2
    ...
    "constant": float          # Intercept term
}
```

### Model JSON Structure

```python
{
    "method": str,             # "intersect", "venn", or "ensemble"
    "d": int,                  # Decile parameter
    "models": [                # List of constituent models
        {
            "feature_name_1": float,
            "feature_name_2": float,
            "constant": float
        },
        ...
    ],
    "meta_model": {            # Only for ensemble method
        "model_0_prob": float,
        "model_1_prob": float,
        "constant": float
    }
}
```

## Cor
rectness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Feature combination generation produces correct count

*For any* valid m, f, and feature list, generating feature combinations should produce exactly m featuresets, each containing exactly f features.

**Validates: Requirements 1.1**

### Property 2: Unique sampling prevents duplicates within featuresets

*For any* featureset generated with sample="unique", no feature should appear more than once within that featureset.

**Validates: Requirements 1.2**

### Property 3: Maximum combinations override

*For any* m value that exceeds the maximum possible combinations for given f and sample method, the system should automatically cap m at the calculated maximum.

**Validates: Requirements 1.4**

### Property 4: Maximum combinations calculation correctness

*For any* number of features n, f value, and sample method, the calculated maximum should equal C(n,f) for "unique" sampling or n^f for "replace" sampling.

**Validates: Requirements 1.5**

### Property 5: PLR calculation formula

*For any* confusion matrix with predictions and true labels, the calculated Positive Likelihood Ratio should equal (True Positive Rate) / (False Positive Rate).

**Validates: Requirements 2.2**

### Property 6: FNR calculation formula

*For any* confusion matrix with predictions and true labels, the calculated False Negative Rate should equal FN / (FN + TP).

**Validates: Requirements 2.3**

### Property 7: DRP calculation formula

*For any* predictions, true labels, and d value, the calculated Decile Ranked Performance should equal (TPR in top d deciles) / (TPR in entire test set).

**Validates: Requirements 2.4, 2.5**

### Property 8: DRS extraction correctness

*For any* predictions, IDs, and d value, the Decile Ranked Set should contain exactly the IDs corresponding to records in the top d deciles when sorted by prediction score descending.

**Validates: Requirements 2.6**

### Property 9: DPS extraction correctness

*For any* predictions, true labels, IDs, and d value, the Decile Positive Set should contain only IDs that are both in the top d deciles and are true positives.

**Validates: Requirements 2.7**

### Property 10: Constituent model count matches featureset count

*For any* m featuresets generated, training should produce exactly m logistic regression models.

**Validates: Requirements 3.1**

### Property 11: Specified solver is used consistently

*For any* valid solver name specified (not "auto"), all trained logistic regression models should use that exact solver.

**Validates: Requirements 3.3**

### Property 12: Constituent model metrics completeness

*For any* trained constituent model, evaluation should produce all required metrics: PLR, FNR, DRP, DRS, and DPS.

**Validates: Requirements 3.4**

### Property 13: Equation dictionary structure

*For any* trained constituent model with feature subset, the equation dictionary should contain exactly one key per feature in the subset plus a "constant" key for the intercept.

**Validates: Requirements 3.5**

### Property 14: JSON export round-trip consistency

*For any* set of trained models, exporting to JSON and reading back should preserve all PLR, FNR, DRP values and equation dictionaries.

**Validates: Requirements 4.2**

### Property 15: Intersect method ranking consistency

*For any* set of constituent models with method="intersect", models should be ranked such that higher PLR, lower FNR, and higher DRP result in better rankings.

**Validates: Requirements 5.1**

### Property 16: Intersect method selection count

*For any* spread value n and set of models with method="intersect", exactly n models should be selected (or fewer if insufficient models outperform baseline).

**Validates: Requirements 5.2**

### Property 17: Intersect method ID deduplication

*For any* selected models with method="intersect", the output dataframe should contain no duplicate IDs.

**Validates: Requirements 5.3**

### Property 18: Intersect method sets count accuracy

*For any* selected models with method="intersect", the sets count for each ID should equal the number of selected model DRS that contain that ID.

**Validates: Requirements 5.4**

### Property 19: Intersect modeljson round-trip

*For any* intersect method execution with modeljson specified, saving and loading the JSON should preserve method name, d parameter, and all equation dictionaries.

**Validates: Requirements 5.5**

### Property 20: Venn method initial selection count

*For any* spread value n with method="venn", initially 2×n models should be selected before filtering.

**Validates: Requirements 6.2**

### Property 21: Venn method unique ID identification

*For any* constituent model DPS and baseline DPS with method="venn", unique IDs should be exactly those in the model DPS but not in baseline DPS or incremental set.

**Validates: Requirements 6.3**

### Property 22: Venn method model discarding

*For any* model with method="venn", if it has no unique IDs compared to baseline and incremental set, it should be discarded from final selection.

**Validates: Requirements 6.4**

### Property 23: Ensemble method probability generation

*For any* selected models with method="ensemble", each model should generate predicted probabilities for all test samples, with values in range [0, 1].

**Validates: Requirements 7.3**

### Property 24: Ensemble method output structure

*For any* ensemble execution, the output dataframe should contain all test set IDs and predicted probabilities in range [0, 1].

**Validates: Requirements 7.5**

### Property 25: Ensemble modeljson completeness

*For any* ensemble method execution with modeljson specified, the saved JSON should include method name, all constituent model equation dictionaries, and the meta-model equation dictionary.

**Validates: Requirements 7.6**

### Property 26: Scoring decile restriction for intersect/venn

*For any* scoring dataset with method="intersect" or "venn", output should only include IDs from the top d deciles of each constituent model's predictions.

**Validates: Requirements 8.2**

### Property 27: Ensemble scoring consistency

*For any* test dataset used in training with method="ensemble", scoring that same dataset should produce predictions matching the original fit predictions.

**Validates: Requirements 8.3**

### Property 28: Scoring output format consistency

*For any* method type, the scoring output structure (column names and types) should match the training output structure for that method.

**Validates: Requirements 8.4**

### Property 29: Invalid parameter rejection

*For any* invalid parameter value (m<1, f>n_features, invalid sample/method/d values, missing ID column, or missing values in data), the system should raise a descriptive ValueError.

**Validates: Requirements 10.1**

### Property 30: SQL generation produces valid syntax

*For any* valid model JSON file, the generated SQL should be syntactically valid and executable against a database table with matching feature columns.

**Validates: Requirements 11.1, 11.8**

### Property 31: SQL logistic regression formula correctness

*For any* equation dictionary, the generated SQL should implement the logistic regression formula: 1 / (1 + EXP(-(constant + sum of feature*coefficient products))).

**Validates: Requirements 11.5**

### Property 32: SQL intersect/venn decile logic

*For any* intersect or venn method JSON, the generated SQL should include logic to rank predictions, identify top d deciles for each constituent model, and aggregate results across models.

**Validates: Requirements 11.2, 11.3, 11.6**

### Property 33: SQL ensemble meta-model application

*For any* ensemble method JSON, the generated SQL should apply the meta-model equation to constituent model probabilities to produce final predictions.

**Validates: Requirements 11.4, 11.7**

## Error Handling

### Parameter Validation Errors

The system validates all parameters before execution and raises `ValueError` with descriptive messages:

- **m parameter**: Must be >= 1
- **f parameter**: Must be >= 1 and <= number of features
- **sample parameter**: Must be "unique" or "replace"
- **d parameter**: Must be between 1 and 10 inclusive
- **method parameter**: Must be "intersect", "venn", or "ensemble"
- **spread parameter**: Must be >= 1
- **solver parameter**: Must be "auto" or a valid scikit-learn solver name
- **id_column parameter**: Must exist in both train and test datasets

### Data Validation Errors

- **Missing values**: Raise `ValueError` listing columns with missing values
- **Shape mismatches**: Raise `ValueError` if X and y shapes don't align
- **ID column missing**: Raise `ValueError` if specified ID column not found
- **Empty datasets**: Raise `ValueError` if train or test data is empty

### Runtime Errors

- **Insufficient models**: If fewer than `spread` models outperform baseline, use all available models and log warning
- **Singular matrix**: If logistic regression fails due to multicollinearity, catch exception and skip that featureset with warning
- **File I/O errors**: Wrap JSON export/import in try-except and raise `IOError` with descriptive message
- **Division by zero**: Handle FPR=0 case in PLR calculation by returning infinity or large constant

### Logging

Implement logging at INFO level for:
- Number of featuresets generated
- Baseline model performance
- Number of constituent models trained
- Number of models selected by ensemble method
- File export operations

Implement logging at WARNING level for:
- m parameter override due to max combinations
- Models skipped due to training failures
- Fewer models selected than spread parameter

## Testing Strategy

### Unit Testing

The ELR package will use pytest for unit testing with the following test modules:

**test_feature_sampler.py**
- Test unique sampling produces no duplicates within featuresets
- Test replace sampling allows duplicates
- Test m override when exceeding max combinations
- Test max combinations calculation for both sampling methods
- Test random state reproducibility

**test_metrics.py**
- Test PLR calculation with known confusion matrices
- Test FNR calculation with known confusion matrices
- Test DRP calculation with known predictions and labels
- Test DRS extraction with known scores
- Test DPS extraction with known scores and labels
- Test edge cases (all positive, all negative, perfect predictions)

**test_validation.py**
- Test parameter validation for all invalid cases
- Test data validation for missing values
- Test ID column validation
- Test shape mismatch detection

**test_model_io.py**
- Test JSON export and import for all model types
- Test file path handling
- Test malformed JSON handling

**test_baseline_model.py**
- Test Random Forest training
- Test baseline metric calculation
- Test integration with metrics module

**test_constituent_model.py**
- Test logistic regression training on feature subsets
- Test equation dictionary extraction
- Test solver selection (auto and specified)

**test_ensemble_methods.py**
- Test intersect method selection and combination
- Test venn method filtering and combination
- Test ensemble method meta-model training
- Test ranking logic for all methods

**test_scorer.py**
- Test scoring with intersect method JSON
- Test scoring with venn method JSON
- Test scoring with ensemble method JSON
- Test output format consistency

**test_sql_generator.py**
- Test SQL generation for intersect method
- Test SQL generation for venn method
- Test SQL generation for ensemble method
- Test logistic regression formula in SQL
- Test SQL syntax validity
- Test decile ranking logic in SQL
- Test meta-model application in SQL

**test_elr_classifier.py**
- Test end-to-end fit and predict workflow
- Test sklearn compatibility (BaseEstimator, ClassifierMixin)
- Test parameter passing through components
- Test integration of all modules

### Property-Based Testing

The ELR package will use Hypothesis for property-based testing. Each property-based test will run a minimum of 100 iterations.

**test_properties.py**

Property tests will be implemented for each correctness property defined above. Each test will:
1. Use Hypothesis strategies to generate random valid inputs
2. Execute the relevant system component
3. Assert the property holds

Example strategies:
- `st.integers(min_value=1, max_value=100)` for m, f, spread parameters
- `st.integers(min_value=1, max_value=10)` for d parameter
- `st.sampled_from(["unique", "replace"])` for sample parameter
- `st.sampled_from(["intersect", "venn", "ensemble"])` for method parameter
- Custom strategies for generating valid dataframes with features and labels

Each property-based test will be tagged with a comment referencing the design document:
```python
# Feature: elr-package, Property 1: Feature combination generation produces correct count
@given(m=st.integers(min_value=1, max_value=50),
       f=st.integers(min_value=1, max_value=10),
       n_features=st.integers(min_value=10, max_value=20))
def test_feature_combination_count(m, f, n_features):
    assume(f <= n_features)
    sampler = FeatureSampler(n_features, f, m, "unique")
    combinations = sampler.generate_combinations()
    assert len(combinations) == min(m, comb(n_features, f))
    assert all(len(combo) == f for combo in combinations)
```

### Integration Testing

**test_integration.py**
- Test complete workflow from fit to predict for all three methods
- Test with real-world-like datasets (imbalanced classes, various sizes)
- Test JSON export and scoring workflow end-to-end
- Test sklearn compatibility (cross_val_score, GridSearchCV)

### Performance Testing

**test_performance.py**
- Benchmark training time with various m and f values
- Benchmark scoring time with various dataset sizes
- Verify memory usage stays reasonable for large m values
- Test parallel processing if implemented

### Test Data

Create synthetic datasets with known properties:
- Balanced binary classification (50/50 split)
- Imbalanced binary classification (90/10 split)
- Linearly separable data
- Non-linearly separable data
- High-dimensional data (many features)
- Small sample size data

### Continuous Integration

- Run all tests on Python 3.8, 3.9, 3.10, 3.11
- Run tests on Windows, Linux, macOS
- Measure code coverage (target: >90%)
- Run linting (flake8, black, mypy)
- Generate test reports

## Implementation Notes

### Solver Selection for "auto"

When solver="auto", use the following heuristic:
- If n_samples < 1000 and n_features < 20: use "lbfgs"
- If n_samples >= 1000 and n_features < 100: use "saga"
- If n_features >= 100: use "saga" with L1 penalty
- If multiclass (not applicable here): use "lbfgs"

### Performance Optimizations

1. **Parallel Training**: Use joblib to train constituent models in parallel
2. **Vectorized Operations**: Use numpy/pandas vectorization for metric calculations
3. **Lazy Evaluation**: Only calculate metrics for models that might be selected
4. **Caching**: Cache baseline results to avoid recalculation

### Scikit-learn Compatibility

Implement the following methods for full sklearn compatibility:
- `fit(X, y)`: Modified to accept test data as well
- `predict(X)`: Return predictions based on method
- `predict_proba(X)`: Return probability estimates
- `score(X, y)`: Return accuracy score
- `get_params()`: Return all parameters
- `set_params(**params)`: Set parameters
- `_more_tags()`: Return estimator tags

### Package Structure for PyPI

```
elr/
├── elr/
│   ├── __init__.py
│   ├── core/
│   ├── metrics/
│   ├── ensemble/
│   ├── scoring/
│   └── utils/
├── tests/
│   ├── test_*.py
├── docs/
│   ├── index.md
│   ├── quickstart.md
│   ├── api.md
│   └── examples/
├── examples/
│   ├── basic_usage.py
│   ├── intersect_method.py
│   ├── venn_method.py
│   └── ensemble_method.py
├── setup.py
├── pyproject.toml
├── README.md
├── LICENSE
├── MANIFEST.in
└── requirements.txt
```

### Dependencies

**Required:**
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0 (for combinatorial calculations)

**Development:**
- pytest >= 7.0.0
- hypothesis >= 6.0.0
- black >= 22.0.0
- flake8 >= 4.0.0
- mypy >= 0.950

**Documentation:**
- sphinx >= 4.0.0
- sphinx-rtd-theme >= 1.0.0

### Version Strategy

Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible API changes
- MINOR: Add functionality (backwards compatible)
- PATCH: Bug fixes (backwards compatible)

Initial release: 0.1.0 (beta)
First stable release: 1.0.0

## Future Enhancements

1. **Additional Ensemble Methods**: Support for stacking, bagging, boosting
2. **Feature Importance**: Calculate and export feature importance scores
3. **Multiclass Support**: Extend to multiclass classification problems
4. **Custom Metrics**: Allow users to define custom performance metrics
5. **Visualization**: Add plotting functions for model comparison and performance
6. **Incremental Learning**: Support for partial_fit for large datasets
7. **GPU Acceleration**: Use cuML for GPU-accelerated training
8. **AutoML Integration**: Integration with hyperparameter tuning libraries
9. **Model Explainability**: SHAP values and other interpretability tools
10. **Streaming Predictions**: Support for real-time scoring APIs
