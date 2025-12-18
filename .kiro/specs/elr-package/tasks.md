# Implementation Plan

- [x] 1. Set up project structure and package configuration





  - Create directory structure for elr package with core, metrics, ensemble, scoring, and utils modules
  - Create setup.py and pyproject.toml with package metadata for PyPI distribution
  - Configure dependencies: scikit-learn, pandas, numpy, scipy
  - Create __init__.py files for all modules with proper imports
  - Set up development dependencies: pytest, hypothesis, black, flake8, mypy
  - Create README.md with basic package description and installation instructions
  - _Requirements: 9.4, 9.5, 9.6_

- [x] 2. Implement parameter validation module





  - Create utils/validation.py with ParameterValidator class
  - Implement validate_parameters method to check m, f, sample, d, method, spread, solver, id_column
  - Implement validate_data method to check for missing values, shape mismatches, ID column existence
  - Raise descriptive ValueError messages for all validation failures
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_

- [x] 2.1 Write property test for parameter validation


  - **Property 29: Invalid parameter rejection**
  - **Validates: Requirements 10.1**

- [x] 3. Implement feature sampling module





  - Create core/feature_sampler.py with FeatureSampler class
  - Implement _calculate_max_combinations method using scipy.special.comb for unique and power for replace
  - Implement generate_combinations method to create m featuresets with f features each
  - Handle m override when exceeding maximum combinations
  - Support both "unique" and "replace" sampling methods
  - Support random_state for reproducibility
  - _Requirements: 1.1, 1.2, 1.4, 1.5_

- [x] 3.1 Write property test for feature combination count


  - **Property 1: Feature combination generation produces correct count**
  - **Validates: Requirements 1.1**

- [x] 3.2 Write property test for unique sampling

  - **Property 2: Unique sampling prevents duplicates within featuresets**
  - **Validates: Requirements 1.2**

- [x] 3.3 Write property test for maximum combinations override


  - **Property 3: Maximum combinations override**
  - **Validates: Requirements 1.4**

- [x] 3.4 Write property test for max combinations calculation


  - **Property 4: Maximum combinations calculation correctness**
  - **Validates: Requirements 1.5**

- [x] 4. Implement performance metrics module





  - Create metrics/performance.py with PerformanceMetrics class
  - Implement positive_likelihood_ratio method calculating TPR/FPR
  - Implement false_negative_rate method calculating FN/(FN+TP)
  - Implement decile_ranked_performance method with sorting, deciling, and TPR ratio calculation
  - Implement extract_decile_ranked_set method to get IDs from top d deciles
  - Implement extract_decile_positive_set method to get true positive IDs from top d deciles
  - Handle edge cases like FPR=0 in PLR calculation
  - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

- [x] 4.1 Write property test for PLR calculation


  - **Property 5: PLR calculation formula**
  - **Validates: Requirements 2.2**

- [x] 4.2 Write property test for FNR calculation


  - **Property 6: FNR calculation formula**
  - **Validates: Requirements 2.3**

- [x] 4.3 Write property test for DRP calculation


  - **Property 7: DRP calculation formula**
  - **Validates: Requirements 2.4, 2.5**

- [x] 4.4 Write property test for DRS extraction


  - **Property 8: DRS extraction correctness**
  - **Validates: Requirements 2.6**

- [x] 4.5 Write property test for DPS extraction


  - **Property 9: DPS extraction correctness**
  - **Validates: Requirements 2.7**

- [x] 5. Implement baseline model module





  - Create core/baseline_model.py with BaselineModel class
  - Implement fit method using sklearn RandomForestClassifier with all features
  - Implement evaluate method to calculate PLR, FNR, DRP, DRS, DPS on test data
  - Store baseline metrics for comparison
  - Support random_state parameter
  - _Requirements: 2.1, 2.8_

- [x] 6. Implement constituent model module





  - Create core/constituent_model.py with ConstituentModel class
  - Implement fit method using sklearn LogisticRegression on feature subset
  - Implement solver selection logic for "auto" parameter based on dataset characteristics
  - Implement evaluate method to calculate PLR, FNR, DRP, DRS, DPS on test data
  - Implement get_equation_dict method to extract coefficients and intercept
  - Handle training failures gracefully with try-except
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 6.1 Write property test for constituent model count


  - **Property 10: Constituent model count matches featureset count**
  - **Validates: Requirements 3.1**

- [x] 6.2 Write property test for solver consistency

  - **Property 11: Specified solver is used consistently**
  - **Validates: Requirements 3.3**

- [x] 6.3 Write property test for metrics completeness

  - **Property 12: Constituent model metrics completeness**
  - **Validates: Requirements 3.4**

- [x] 6.4 Write property test for equation dictionary structure

  - **Property 13: Equation dictionary structure**
  - **Validates: Requirements 3.5**

- [x] 7. Implement model I/O module





  - Create utils/model_io.py with ModelIO class
  - Implement export_all_models method to save all constituent model metrics to JSON (elr2json)
  - Implement export_selected_models method to save method, d, and selected model equations to JSON (modeljson)
  - Implement load_model method to read model configuration from JSON
  - Handle file I/O errors with descriptive messages
  - Ensure JSON structure supports SQL equation reconstruction
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 7.1 Write property test for JSON round-trip


  - **Property 14: JSON export round-trip consistency**
  - **Validates: Requirements 4.2**

- [x] 8. Implement intersect ensemble method





  - Create ensemble/intersect.py with IntersectMethod class
  - Implement model ranking by PLR (higher better), FNR (lower better), DRP (higher better)
  - Implement select_and_combine method to select top n models based on spread parameter
  - Filter models that outperform baseline
  - Compile DRS from selected models and deduplicate IDs
  - Count occurrences of each ID across selected model DRS
  - Return DataFrame with columns [id, sets]
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 8.1 Write property test for intersect ranking


  - **Property 15: Intersect method ranking consistency**
  - **Validates: Requirements 5.1**

- [x] 8.2 Write property test for intersect selection count


  - **Property 16: Intersect method selection count**
  - **Validates: Requirements 5.2**

- [x] 8.3 Write property test for intersect deduplication


  - **Property 17: Intersect method ID deduplication**
  - **Validates: Requirements 5.3**

- [x] 8.4 Write property test for intersect sets count


  - **Property 18: Intersect method sets count accuracy**
  - **Validates: Requirements 5.4**

- [x] 8.5 Write property test for intersect modeljson round-trip


  - **Property 19: Intersect modeljson round-trip**
  - **Validates: Requirements 5.5**

- [x] 9. Implement venn ensemble method





  - Create ensemble/venn.py with VennMethod class
  - Implement model ranking by PLR, FNR, DRP
  - Implement select_and_combine method to initially select top 2×n models
  - Compare each model's DPS against baseline DPS to identify unique IDs
  - Maintain incremental ID set and discard models with no unique IDs
  - Compile DRS from undiscarded models and deduplicate IDs
  - Count occurrences of each ID across undiscarded model DRS
  - Return DataFrame with columns [id, sets]
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 9.1 Write property test for venn initial selection count


  - **Property 20: Venn method initial selection count**
  - **Validates: Requirements 6.2**

- [x] 9.2 Write property test for venn unique ID identification

  - **Property 21: Venn method unique ID identification**
  - **Validates: Requirements 6.3**

- [x] 9.3 Write property test for venn model discarding

  - **Property 22: Venn method model discarding**
  - **Validates: Requirements 6.4**

- [x] 10. Implement ensemble ensemble method





  - Create ensemble/ensemble.py with EnsembleMethod class
  - Implement model ranking by PLR, FNR, DRP
  - Implement select_and_combine method to select top n models
  - Score test set with each selected model to generate predicted probabilities
  - Train meta-model LogisticRegression with lbfgs solver using constituent predictions as features
  - Generate final predictions from meta-model
  - Return DataFrame with columns [id, predicted] and meta-model equation dictionary
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 10.1 Write property test for ensemble probability generation


  - **Property 23: Ensemble method probability generation**
  - **Validates: Requirements 7.3**

- [x] 10.2 Write property test for ensemble output structure


  - **Property 24: Ensemble method output structure**
  - **Validates: Requirements 7.5**

- [x] 10.3 Write property test for ensemble modeljson completeness


  - **Property 25: Ensemble modeljson completeness**
  - **Validates: Requirements 7.6**

- [x] 11. Implement model scorer module





  - Create scoring/scorer.py with ModelScorer class
  - Implement __init__ to load model configuration from modeljson file
  - Implement score method for intersect/venn methods: apply each equation, rank, filter to top d deciles
  - Implement score method for ensemble method: apply constituent equations, then meta-model equation
  - Ensure output format matches training method output structure
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 11.1 Write property test for scoring decile restriction


  - **Property 26: Scoring decile restriction for intersect/venn**
  - **Validates: Requirements 8.2**

- [x] 11.2 Write property test for ensemble scoring consistency


  - **Property 27: Ensemble scoring consistency**
  - **Validates: Requirements 8.3**

- [x] 11.3 Write property test for scoring output format


  - **Property 28: Scoring output format consistency**
  - **Validates: Requirements 8.4**

- [x] 12. Implement main ELRClassifier class





  - Create core/elr_classifier.py with ELRClassifier class
  - Inherit from sklearn BaseEstimator and ClassifierMixin
  - Implement __init__ with all parameters: m, f, sample, d, method, spread, solver, id_column, elr2json, modeljson, random_state
  - Implement fit method orchestrating: validation, feature sampling, baseline training, constituent training, ensemble method application, JSON export
  - Implement predict method to generate predictions based on method type
  - Implement predict_proba method for probability estimates
  - Implement get_params and set_params for sklearn compatibility
  - Add logging for key operations
  - _Requirements: 9.1, 9.2, 9.3_

- [x] 12.1 Write unit tests for ELRClassifier API


  - Test fit method accepts correct parameters
  - Test predict method returns predictions
  - Test sklearn compatibility (BaseEstimator, ClassifierMixin)
  - Test end-to-end workflow for all three methods
  - _Requirements: 9.1, 9.2, 9.3_
- [x] 13. Implement SQL generator module

  - Create sql/generator.py with SQLGenerator class
  - Implement __init__ to load model JSON file
  - Implement generate_sql method to create complete SQL query based on method type
  - Implement _generate_logistic_regression_sql to create CTE for single logistic regression equation
  - Implement _generate_intersect_sql to create SQL with decile ranking and ID aggregation logic
  - Implement _generate_venn_sql to create SQL with decile ranking and ID aggregation logic
  - Implement _generate_ensemble_sql to create SQL with constituent CTEs and meta-model application
  - Ensure generated SQL uses proper logistic regression formula: 1 / (1 + EXP(-(constant + sum)))
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8_

- [x] 13.1 Write property test for SQL syntax validity


  - **Property 30: SQL generation produces valid syntax**
  - **Validates: Requirements 11.1, 11.8**

- [x] 13.2 Write property test for SQL logistic regression formula


  - **Property 31: SQL logistic regression formula correctness**
  - **Validates: Requirements 11.5**

- [x] 13.3 Write property test for SQL intersect/venn decile logic


  - **Property 32: SQL intersect/venn decile logic**
  - **Validates: Requirements 11.2, 11.3, 11.6**

- [x] 13.4 Write property test for SQL ensemble meta-model


  - **Property 33: SQL ensemble meta-model application**
  - **Validates: Requirements 11.4, 11.7**

- [x] 13.5 Write unit tests for SQL generator


  - Test SQL generation for intersect method with sample model JSON
  - Test SQL generation for venn method with sample model JSON
  - Test SQL generation for ensemble method with sample model JSON
  - Test that generated SQL can be parsed (syntax check)
  - Test that feature names are properly escaped in SQL
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [x] 14. Create package exports and documentation





  - Update elr/__init__.py to export ELRClassifier, ModelScorer, and SQLGenerator
  - Create docstrings for all public classes and methods following NumPy style
  - Create examples/ directory with basic_usage.py, intersect_method.py, venn_method.py, ensemble_method.py, sql_generation.py
  - Update README.md with installation instructions, quick start guide, SQL generation examples
  - Create LICENSE file (MIT or Apache 2.0)
  - Create MANIFEST.in for package data
  - _Requirements: 9.6, 11.1_

- [x] 15. Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Build and test PyPI package





  - Test package installation with pip install -e .
  - Verify package is importable: from elr import ELRClassifier, SQLGenerator
  - Run all unit tests and property tests
  - Test package building with python -m build
  - Test package installation from wheel
  - Verify all dependencies are correctly specified
  - _Requirements: 9.6_

- [x] 16.1 Write integration tests


  - Test complete workflow from fit to predict for all methods
  - Test with synthetic datasets (balanced, imbalanced, high-dimensional)
  - Test JSON export and scoring workflow end-to-end
  - Test SQL generation and execution against SQLite database
  - Test sklearn compatibility with cross_val_score
  - _Requirements: 9.1, 9.2, 9.3, 11.8_

- [x] 17. Final checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.
