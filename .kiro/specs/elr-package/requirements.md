# Requirements Document

## Introduction

The ELR (Ensemble Logistic Regression) package is a Python library designed for advanced classification tasks using ensemble methods based on combinatorial feature selection and logistic regression. The package integrates with scikit-learn and provides three distinct ensemble strategies (intersect, venn, and ensemble) to improve model performance through feature diversity. ELR establishes baseline performance using Random Forest, trains multiple logistic regression models on different feature combinations, and selects top-performing models based on specialized metrics including Positive Likelihood Ratio, False Negative Rate, and Decile Ranked Performance. The package supports both model training and scoring workflows, with exportable model configurations in JSON format for deployment.

## Glossary

- **ELR System**: The Ensemble Logistic Regression package that trains and scores classification models
- **Featureset**: A subset of features selected from the complete feature list for training a single model
- **Positive Likelihood Ratio (PLR)**: The ratio of True Positive Rate to False Positive Rate
- **False Negative Rate (FNR)**: The proportion of actual positives incorrectly classified as negatives
- **Decile Ranked Performance (DRP)**: The ratio of True Positive Rate in top d deciles to True Positive Rate in the entire test set
- **Decile Ranked Set (DRS)**: The set of IDs from the top d deciles of predicted scores
- **Decile Positive Set (DPS)**: The set of IDs from the top d deciles that are true positives
- **Baseline Model**: A Random Forest classifier trained on all features to establish performance benchmarks
- **Constituent Model**: An individual logistic regression model trained on a specific featureset
- **Equation Dictionary**: A dictionary mapping feature names to coefficients plus intercept under "constant" key
- **Model JSON**: A JSON file containing method type, parameters, and equation dictionaries for scoring

## Requirements

### Requirement 1

**User Story:** As a data scientist, I want to generate diverse feature combinations for model training, so that I can explore different feature interactions and improve model robustness.

#### Acceptance Criteria

1. WHEN the ELR System receives m parameter, f parameter, and feature list, THE ELR System SHALL generate m featuresets each containing f features
2. WHERE sample parameter equals "unique", THE ELR System SHALL ensure no feature appears more than once within a single featureset
3. WHERE sample parameter equals "replace", THE ELR System SHALL allow features to appear multiple times within a single featureset
4. WHEN m parameter exceeds the maximum possible combinations for the given f and sample method, THE ELR System SHALL automatically override m with the calculated maximum
5. WHEN the ELR System calculates maximum combinations, THE ELR System SHALL use combinatorial mathematics based on the number of features, f parameter, and sample method

### Requirement 2

**User Story:** As a data scientist, I want to establish baseline model performance using Random Forest, so that I can compare ensemble logistic regression results against a standard benchmark.

#### Acceptance Criteria

1. WHEN the ELR System receives training data and test data, THE ELR System SHALL train a Random Forest classifier using all features on the training data
2. WHEN the Baseline Model generates predictions on test data, THE ELR System SHALL calculate Positive Likelihood Ratio as True Positive Rate divided by False Positive Rate
3. WHEN the Baseline Model generates predictions on test data, THE ELR System SHALL calculate False Negative Rate
4. WHEN the Baseline Model generates predictions on test data, THE ELR System SHALL calculate Decile Ranked Performance using the d parameter
5. WHEN calculating Decile Ranked Performance, THE ELR System SHALL sort predicted scores from highest to lowest, divide into deciles, and compute the ratio of True Positive Rate in top d deciles to True Positive Rate in entire test set
6. WHEN the Baseline Model generates predictions on test data, THE ELR System SHALL extract Decile Ranked Set from the id column for records in top d deciles
7. WHEN the Baseline Model generates predictions on test data, THE ELR System SHALL extract Decile Positive Set from the id column for true positive records in top d deciles
8. WHEN baseline metrics are calculated, THE ELR System SHALL store Positive Likelihood Ratio, False Negative Rate, Decile Ranked Performance, and Decile Ranked Set for comparison

### Requirement 3

**User Story:** As a data scientist, I want to train multiple logistic regression models on different feature combinations, so that I can identify which feature subsets produce the best classification performance.

#### Acceptance Criteria

1. WHEN the ELR System has generated m featuresets, THE ELR System SHALL train m logistic regression models using scikit-learn, one for each featureset
2. WHERE solver parameter equals "auto", THE ELR System SHALL automatically select the optimal scikit-learn solver based on dataset size and variable characteristics
3. WHERE solver parameter specifies a scikit-learn solver name, THE ELR System SHALL use the specified solver for all logistic regression models
4. WHEN each Constituent Model is trained, THE ELR System SHALL evaluate the model on test data and calculate Positive Likelihood Ratio, False Negative Rate, Decile Ranked Performance, Decile Ranked Set, and Decile Positive Set
5. WHEN each Constituent Model is trained, THE ELR System SHALL extract an Equation Dictionary containing feature names as keys, coefficient values as values, and intercept value under the "constant" key
6. WHEN each Constituent Model is evaluated, THE ELR System SHALL store all metrics and the Equation Dictionary for subsequent selection and export

### Requirement 4

**User Story:** As a data scientist, I want to export all trained model equations and metrics to JSON format, so that I can review model performance and use equations for SQL-based scoring.

#### Acceptance Criteria

1. WHERE elr2json parameter is not null, THE ELR System SHALL consolidate Positive Likelihood Ratio, False Negative Rate, Decile Ranked Performance, and Equation Dictionary for all Constituent Models
2. WHERE elr2json parameter is not null, THE ELR System SHALL export the consolidated data to a JSON file at the file path specified in elr2json parameter
3. WHEN exporting to JSON, THE ELR System SHALL structure the data to enable reconstruction of logistic regression equations in SQL format

### Requirement 5

**User Story:** As a data scientist, I want to use the intersect method to identify IDs that appear in multiple top-performing models, so that I can find high-confidence predictions supported by diverse feature combinations.

#### Acceptance Criteria

1. WHERE method parameter equals "intersect", THE ELR System SHALL rank all Constituent Models by Positive Likelihood Ratio, False Negative Rate, and Decile Ranked Performance
2. WHERE method parameter equals "intersect", THE ELR System SHALL select the top n Constituent Models as defined by the spread parameter, prioritizing models that outperform the Baseline Model
3. WHERE method parameter equals "intersect", THE ELR System SHALL compile Decile Ranked Sets from all selected Constituent Models and deduplicate IDs
4. WHERE method parameter equals "intersect", THE ELR System SHALL output a dataframe with deduplicated IDs in one column and a sets column indicating the count of Decile Ranked Sets containing each ID
5. WHERE method parameter equals "intersect" and modeljson parameter is not null, THE ELR System SHALL save the method name, d parameter value, and Equation Dictionaries of selected models to the file path specified in modeljson parameter

### Requirement 6

**User Story:** As a data scientist, I want to use the venn method to discover unique predictions not captured by the baseline model, so that I can identify cases where alternative feature combinations provide novel insights.

#### Acceptance Criteria

1. WHERE method parameter equals "venn", THE ELR System SHALL rank all Constituent Models by Positive Likelihood Ratio, False Negative Rate, and Decile Ranked Performance
2. WHERE method parameter equals "venn", THE ELR System SHALL initially select the top 2×n Constituent Models where n is defined by the spread parameter, prioritizing models that outperform the Baseline Model
3. WHERE method parameter equals "venn", THE ELR System SHALL compare Decile Positive Set of each selected Constituent Model against the Decile Positive Set of the Baseline Model and identify unique IDs
4. WHERE method parameter equals "venn", THE ELR System SHALL discard any Constituent Model that has no unique IDs compared to the Baseline Model Decile Positive Set and the incremental ID set
5. WHERE method parameter equals "venn", THE ELR System SHALL compile Decile Ranked Sets from all undiscarded Constituent Models and deduplicate IDs
6. WHERE method parameter equals "venn", THE ELR System SHALL output a dataframe with deduplicated IDs in one column and a sets column indicating the count of Decile Ranked Sets containing each ID
7. WHERE method parameter equals "venn" and modeljson parameter is not null, THE ELR System SHALL save the method name, d parameter value, and Equation Dictionaries of undiscarded models to the file path specified in modeljson parameter

### Requirement 7

**User Story:** As a data scientist, I want to use the ensemble method to create a meta-model from top-performing logistic regressions, so that I can combine their predictions into a single optimized classifier.

#### Acceptance Criteria

1. WHERE method parameter equals "ensemble", THE ELR System SHALL rank all Constituent Models by Positive Likelihood Ratio, False Negative Rate, and Decile Ranked Performance
2. WHERE method parameter equals "ensemble", THE ELR System SHALL select the top n Constituent Models as defined by the spread parameter, prioritizing models that outperform the Baseline Model
3. WHERE method parameter equals "ensemble", THE ELR System SHALL score the entire test set with each selected Constituent Model to generate predicted probabilities
4. WHERE method parameter equals "ensemble", THE ELR System SHALL train a meta-model logistic regression using lbfgs solver with predicted probabilities from selected Constituent Models as feature inputs
5. WHERE method parameter equals "ensemble", THE ELR System SHALL output a dataframe with test set IDs in one column and predicted probabilities from the meta-model in a predicted column
6. WHERE method parameter equals "ensemble" and modeljson parameter is not null, THE ELR System SHALL save the method name, Equation Dictionaries of selected Constituent Models, and the meta-model Equation Dictionary to the file path specified in modeljson parameter

### Requirement 8

**User Story:** As a data scientist, I want to score new datasets using previously trained models, so that I can apply the ensemble approach to production data without retraining.

#### Acceptance Criteria

1. WHEN the ELR System receives a scoring dataset and a Model JSON file, THE ELR System SHALL load the method type and Equation Dictionaries from the Model JSON file
2. WHERE the Model JSON method equals "intersect" or "venn", THE ELR System SHALL score the dataset using each Constituent Model Equation Dictionary, rank predictions, and restrict output to top d deciles for each model
3. WHERE the Model JSON method equals "ensemble", THE ELR System SHALL score the dataset using each Constituent Model Equation Dictionary to generate predicted probabilities, then apply the meta-model Equation Dictionary to produce final predictions
4. WHEN scoring is complete, THE ELR System SHALL output predictions in a format consistent with the training method output structure

### Requirement 9

**User Story:** As a Python developer, I want the ELR package to follow scikit-learn conventions and be installable via PyPI, so that I can easily integrate it into my machine learning workflows.

#### Acceptance Criteria

1. THE ELR System SHALL implement a fit method that accepts training data, test data, and configuration parameters
2. THE ELR System SHALL implement a predict method that accepts new data and returns predictions
3. THE ELR System SHALL provide a scikit-learn compatible API with fit and predict methods
4. THE ELR System SHALL include proper package metadata for PyPI distribution including package name, version, author, description, and dependencies
5. THE ELR System SHALL specify scikit-learn, pandas, and numpy as required dependencies in package configuration
6. WHEN the ELR System is installed via pip, THE ELR System SHALL be importable as a Python module

### Requirement 10

**User Story:** As a data scientist, I want clear parameter validation and error messages, so that I can quickly identify and fix configuration issues.

#### Acceptance Criteria

1. WHEN the ELR System receives invalid parameter values, THE ELR System SHALL raise descriptive exceptions indicating the parameter name and valid value range
2. WHEN m parameter is less than 1, THE ELR System SHALL raise a validation error
3. WHEN f parameter exceeds the number of available features, THE ELR System SHALL raise a validation error
4. WHEN sample parameter is not "unique" or "replace", THE ELR System SHALL raise a validation error
5. WHEN method parameter is not "intersect", "venn", or "ensemble", THE ELR System SHALL raise a validation error
6. WHEN d parameter is not between 1 and 10 inclusive, THE ELR System SHALL raise a validation error
7. WHEN the id column specified in id parameter does not exist in the dataset, THE ELR System SHALL raise a validation error
8. WHEN training data or test data contains missing values in feature columns, THE ELR System SHALL raise a validation error with information about affected columns

### Requirement 11

**User Story:** As a data engineer, I want to convert trained model equations to SQL format, so that I can deploy models directly in database environments for efficient scoring at scale.

#### Acceptance Criteria

1. WHEN the ELR System receives a Model JSON file, THE ELR System SHALL provide a function to convert Equation Dictionaries to SQL format
2. WHERE the Model JSON method equals "intersect", THE ELR System SHALL generate SQL with Common Table Expressions (CTEs) for each Constituent Model equation and additional SQL to identify IDs appearing in top d deciles across multiple models
3. WHERE the Model JSON method equals "venn", THE ELR System SHALL generate SQL with CTEs for each Constituent Model equation and additional SQL to identify IDs appearing in top d deciles across multiple models
4. WHERE the Model JSON method equals "ensemble", THE ELR System SHALL generate SQL with CTEs for each Constituent Model equation and a final SELECT statement applying the meta-model equation to output predicted probabilities
5. WHEN generating SQL for a Constituent Model, THE ELR System SHALL create a logistic regression equation using the formula: 1 / (1 + EXP(-(constant + feature1*coef1 + feature2*coef2 + ...)))
6. WHEN generating SQL for intersect or venn methods, THE ELR System SHALL include logic to rank predictions, identify top d deciles for each model, and aggregate results across models
7. WHEN generating SQL for ensemble method, THE ELR System SHALL apply the meta-model equation to the predicted probabilities from constituent CTEs to produce final probability scores
8. WHEN SQL generation is complete, THE ELR System SHALL return a valid SQL query string that can be executed against a database table with the same feature columns
