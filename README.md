# Paramsemble-Class - Parametric Ensemble for Classification

Paramsemble-Class (arametric Ensemble for Classification) is a Python library for advanced classification tasks using ensemble methods based on combinatorial feature selection and parametric logistic regression.

This apprach can outperform XGBoost for datasets that are highly heterogenous, or have high variability of noise (e.g. changing customer behaviors over time), or contain rare events. This approach has been validated during COVID-19 where models based on other ensemble methods like random forests, XGBoost etc. collapsed, but models using this approach maintained their performance despite some declines.


## Features

- **Multiple Ensemble Strategies**: Choose from three distinct ensemble methods:
  - **Intersect**: Identify high-confidence predictions supported by diverse feature combinations
  - **Venn**: Discover unique predictions not captured by baseline models
  - **Ensemble**: Create meta-models from top-performing logistic regressions

- **Scikit-learn Compatible**: Seamless integration with scikit-learn workflows using familiar `fit` and `predict` methods

- **Specialized Metrics**: Tailored for imbalanced classification tasks:
  - Positive Likelihood Ratio (PLR)
  - False Negative Rate (FNR)
  - Decile Ranked Performance (DRP)

- **SQL Export**: Convert trained models to SQL queries for efficient database-level scoring

- **Model Persistence**: Export and import models via JSON for production deployment

## Installation

### From PyPI (when published)

```bash
pip install paramsemble-class
```

### From Source

```bash
git clone https://github.com/schen18/paramsemble-class.git
cd paramsemble-class
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from paramsemble_class import ELRClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize ELR classifier
clf = ELRClassifier(
    m=100,              # Number of feature combinations
    f=5,                # Features per combination
    method="intersect", # Ensemble method
    spread=10,          # Number of top models to select
    d=2,                # Top deciles to consider
    random_state=42
)

# Train the model
clf.fit(X_train, y_train, X_test, y_test)

# Make predictions
predictions = clf.predict(X_test)
```

## Ensemble Methods

### Intersect Method

Identifies IDs that appear in multiple top-performing models, providing high-confidence predictions:

```python
clf = ELRClassifier(method="intersect", spread=10, d=2)
clf.fit(X_train, y_train, X_test, y_test)
results = clf.predict(X_test)  # Returns DataFrame with [id, sets]
```

### Venn Method

Discovers unique predictions not captured by the baseline model:

```python
clf = ELRClassifier(method="venn", spread=10, d=2)
clf.fit(X_train, y_train, X_test, y_test)
results = clf.predict(X_test)  # Returns DataFrame with [id, sets]
```

### Ensemble Method

Creates a meta-model combining predictions from top-performing models:

```python
clf = ELRClassifier(method="ensemble", spread=10)
clf.fit(X_train, y_train, X_test, y_test)
predictions = clf.predict(X_test)  # Returns DataFrame with [id, predicted]
```

## SQL Generation

Export trained models to SQL for database-level scoring:

```python
from paramsemble_class import SQLGenerator

# After training and exporting model JSON
generator = SQLGenerator("model.json")
sql_query = generator.generate_sql("my_table", id_column="customer_id")

# Execute the SQL query in your database
```

## Model Persistence

Export models for production use:

```python
# Export all model metrics
clf = ELRClassifier(elr2json="all_models.json")
clf.fit(X_train, y_train, X_test, y_test)

# Export selected models for scoring
clf = ELRClassifier(modeljson="selected_models.json")
clf.fit(X_train, y_train, X_test, y_test)

# Score new data using saved models
from paramsemble_class import ModelScorer
scorer = ModelScorer("selected_models.json")
predictions = scorer.score(X_new, ids_new)
```

## Parameters

- `m` (int): Number of feature combinations to generate (default: 100)
- `f` (int): Number of features per combination (default: 5)
- `sample` (str): Feature sampling method - "unique" or "replace" (default: "unique")
- `d` (int): Number of top deciles to consider (1-10) (default: 2)
- `method` (str): Ensemble method - "intersect", "venn", or "ensemble" (default: "intersect")
- `spread` (int): Number of top models to select (default: 10)
- `solver` (str): Logistic regression solver or "auto" (default: "auto")
- `id_column` (str): Name of ID column in datasets (default: "id")
- `elr2json` (str): Path to export all model metrics (optional)
- `modeljson` (str): Path to export selected model equations (optional)
- `random_state` (int): Random seed for reproducibility (optional)

## Requirements

- Python >= 3.8
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black paramsemble_class/ tests/

# Lint code
flake8 paramsemble_class/ tests/

# Type checking
mypy paramsemble_class/
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use Paramsemble-Class in your research, please cite:

```
@software{paramsemble_class,
  title = {Paramsemble-Class: Ensemble Logistic Regression},
  author = {Stephen Chen},
  year = {2024},
  url = {https://github.com/schen18/paramsemble-class}
}
```
