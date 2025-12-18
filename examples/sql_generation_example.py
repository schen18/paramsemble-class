"""Example demonstrating SQL generation from ELR models."""
import json
import tempfile
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from paramsemble_class.sql.generator import SQLGenerator


def example_intersect_sql():
    """Generate SQL for intersect method."""
    print("=" * 80)
    print("INTERSECT METHOD SQL GENERATION")
    print("=" * 80)
    
    # Create sample model configuration
    config = {
        'method': 'intersect',
        'd': 2,
        'models': [
            {'age': 0.05, 'income': 0.0003, 'credit_score': 0.01, 'constant': -2.5},
            {'age': 0.03, 'debt_ratio': -1.2, 'employment_years': 0.08, 'constant': -1.8},
            {'income': 0.0002, 'credit_score': 0.012, 'debt_ratio': -0.9, 'constant': -2.1}
        ]
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        json_path = f.name
    
    # Generate SQL
    generator = SQLGenerator(json_path)
    sql = generator.generate_sql('loan_applications', 'application_id')
    
    print("\nGenerated SQL:")
    print(sql)
    print("\n")


def example_ensemble_sql():
    """Generate SQL for ensemble method."""
    print("=" * 80)
    print("ENSEMBLE METHOD SQL GENERATION")
    print("=" * 80)
    
    # Create sample model configuration
    config = {
        'method': 'ensemble',
        'd': 2,
        'models': [
            {'age': 0.05, 'income': 0.0003, 'constant': -2.5},
            {'credit_score': 0.01, 'debt_ratio': -1.2, 'constant': -1.8}
        ],
        'meta_model': {
            'model_0_prob': 0.6,
            'model_1_prob': 0.4,
            'constant': -0.1
        }
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        json_path = f.name
    
    # Generate SQL
    generator = SQLGenerator(json_path)
    sql = generator.generate_sql('loan_applications', 'application_id')
    
    print("\nGenerated SQL:")
    print(sql)
    print("\n")


if __name__ == '__main__':
    example_intersect_sql()
    example_ensemble_sql()
