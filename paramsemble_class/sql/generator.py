"""SQL generation from model JSON for database scoring."""
import json
from typing import Dict, List, Any
from ..utils.model_io import ModelIO


class SQLGenerator:
    """
    Generates SQL queries from model JSON files for database scoring.
    
    This class converts trained ELR model configurations (stored in JSON format)
    into executable SQL queries that can be run directly against database tables.
    It supports all three ensemble methods: intersect, venn, and ensemble.
    
    The generated SQL uses Common Table Expressions (CTEs) to organize the logic
    and implements the logistic regression formula for probability calculation.
    
    Attributes
    ----------
    model_config : Dict[str, Any]
        Loaded model configuration containing method, d parameter, and equations.
    method : str
        Ensemble method: "intersect", "venn", or "ensemble".
    d : int
        Number of top deciles to consider (1-10).
    models : List[Dict[str, float]]
        List of constituent model equation dictionaries.
    meta_model : Dict[str, float], optional
        Meta-model equation dictionary (only for ensemble method).
    
    Examples
    --------
    >>> generator = SQLGenerator('model_config.json')
    >>> sql = generator.generate_sql('customer_data', 'customer_id')
    >>> print(sql)
    """
    
    def __init__(self, modeljson_path: str):
        """
        Initialize SQL generator with model JSON file.
        
        Parameters
        ----------
        modeljson_path : str
            Path to the model JSON file (modeljson format).
        
        Raises
        ------
        IOError
            If file cannot be read.
        ValueError
            If JSON structure is invalid.
        """
        self.model_config = ModelIO.load_model(modeljson_path)
        self.method = self.model_config['method']
        self.d = self.model_config['d']
        self.models = self.model_config['models']
        self.meta_model = self.model_config.get('meta_model', None)
    
    def generate_sql(self, table_name: str, id_column: str = "id") -> str:
        """
        Generate SQL query for model scoring.
        
        Creates a complete SQL query that can be executed against a database
        table to score records using the loaded model configuration.
        
        Parameters
        ----------
        table_name : str
            Name of the database table to score.
        id_column : str, default="id"
            Name of the ID column in the table.
        
        Returns
        -------
        str
            Complete SQL query string ready for execution.
        
        Examples
        --------
        >>> generator = SQLGenerator('model.json')
        >>> sql = generator.generate_sql('customers', 'customer_id')
        """
        if self.method == "intersect":
            return self._generate_intersect_sql(table_name, id_column)
        elif self.method == "venn":
            return self._generate_venn_sql(table_name, id_column)
        elif self.method == "ensemble":
            return self._generate_ensemble_sql(table_name, id_column)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _generate_logistic_regression_sql(
        self,
        equation_dict: Dict[str, float],
        table_name: str,
        cte_name: str
    ) -> str:
        """
        Generate SQL CTE for a single logistic regression equation.
        
        Creates a Common Table Expression that applies the logistic regression
        formula: 1 / (1 + EXP(-(constant + feature1*coef1 + feature2*coef2 + ...)))
        
        Parameters
        ----------
        equation_dict : Dict[str, float]
            Dictionary mapping feature names to coefficients, with 'constant' key.
        table_name : str
            Name of the source table.
        cte_name : str
            Name for the CTE.
        
        Returns
        -------
        str
            SQL CTE string.
        """
        # Extract constant (intercept)
        constant = equation_dict.get('constant', 0.0)
        
        # Build feature terms
        feature_terms = []
        for feature_name, coefficient in equation_dict.items():
            if feature_name != 'constant':
                # Escape feature names with double quotes for SQL
                escaped_feature = f'"{feature_name}"' if not feature_name.startswith('"') else feature_name
                feature_terms.append(f"({escaped_feature} * {coefficient})")
        
        # Build the linear combination
        if feature_terms:
            linear_combination = f"{constant} + " + " + ".join(feature_terms)
        else:
            linear_combination = str(constant)
        
        # Build the logistic regression formula
        probability_formula = f"1.0 / (1.0 + EXP(-({linear_combination})))"
        
        # Create CTE
        cte_sql = f"""{cte_name} AS (
    SELECT 
        *,
        {probability_formula} AS predicted_probability
    FROM {table_name}
)"""
        
        return cte_sql
    
    def _generate_intersect_sql(
        self,
        table_name: str,
        id_column: str
    ) -> str:
        """
        Generate SQL for intersect method.
        
        Creates SQL that:
        1. Generates CTEs for each constituent model
        2. Ranks predictions and identifies top d deciles for each model
        3. Aggregates IDs across models
        4. Counts occurrences per ID
        
        Parameters
        ----------
        table_name : str
            Name of the database table to score.
        id_column : str
            Name of the ID column.
        
        Returns
        -------
        str
            Complete SQL query.
        """
        # Generate CTEs for each constituent model
        ctes = []
        for i, model_equation in enumerate(self.models):
            cte_name = f"model_{i}"
            cte_sql = self._generate_logistic_regression_sql(
                model_equation,
                table_name,
                cte_name
            )
            ctes.append(cte_sql)
        
        # Generate decile ranking CTEs for each model
        decile_ctes = []
        for i in range(len(self.models)):
            decile_cte_name = f"model_{i}_deciles"
            decile_cte = f"""{decile_cte_name} AS (
    SELECT 
        "{id_column}",
        predicted_probability,
        NTILE(10) OVER (ORDER BY predicted_probability DESC) AS decile_rank
    FROM model_{i}
)"""
            decile_ctes.append(decile_cte)
        
        # Generate top d deciles CTEs for each model
        top_decile_ctes = []
        for i in range(len(self.models)):
            top_decile_cte_name = f"model_{i}_top_deciles"
            top_decile_cte = f"""{top_decile_cte_name} AS (
    SELECT "{id_column}"
    FROM model_{i}_deciles
    WHERE decile_rank <= {self.d}
)"""
            top_decile_ctes.append(top_decile_cte)
        
        # Union all top decile IDs
        union_parts = []
        for i in range(len(self.models)):
            union_parts.append(f'    SELECT "{id_column}" FROM model_{i}_top_deciles')
        
        union_cte = f"""all_top_decile_ids AS (
{chr(10).join(f"{part}" if i == 0 else f"    UNION ALL{chr(10)}{part}" for i, part in enumerate(union_parts))}
)"""
        
        # Final aggregation
        final_query = f"""SELECT 
    "{id_column}" AS id,
    COUNT(*) AS sets
FROM all_top_decile_ids
GROUP BY "{id_column}"
ORDER BY sets DESC, "{id_column}" ASC"""
        
        # Combine all CTEs and final query
        all_ctes = ctes + decile_ctes + top_decile_ctes + [union_cte]
        full_query = f"WITH {',\n'.join(all_ctes)}\n{final_query};"
        
        return full_query
    
    def _generate_venn_sql(
        self,
        table_name: str,
        id_column: str
    ) -> str:
        """
        Generate SQL for venn method.
        
        Creates SQL that:
        1. Generates CTEs for each constituent model
        2. Ranks predictions and identifies top d deciles for each model
        3. Aggregates IDs across models
        4. Counts occurrences per ID
        
        Note: The venn method uses the same SQL structure as intersect for scoring.
        The difference in model selection happens during training, not scoring.
        
        Parameters
        ----------
        table_name : str
            Name of the database table to score.
        id_column : str
            Name of the ID column.
        
        Returns
        -------
        str
            Complete SQL query.
        """
        # Venn method uses the same SQL structure as intersect for scoring
        # The difference is in which models were selected during training
        return self._generate_intersect_sql(table_name, id_column)
    
    def _generate_ensemble_sql(
        self,
        table_name: str,
        id_column: str
    ) -> str:
        """
        Generate SQL for ensemble method.
        
        Creates SQL that:
        1. Generates CTEs for each constituent model
        2. Joins all constituent predictions
        3. Applies meta-model equation to constituent probabilities
        4. Returns final probability scores
        
        Parameters
        ----------
        table_name : str
            Name of the database table to score.
        id_column : str
            Name of the ID column.
        
        Returns
        -------
        str
            Complete SQL query.
        """
        if self.meta_model is None:
            raise ValueError("Meta-model is required for ensemble method but was not found in configuration.")
        
        # Generate CTEs for each constituent model
        ctes = []
        for i, model_equation in enumerate(self.models):
            cte_name = f"model_{i}"
            cte_sql = self._generate_logistic_regression_sql(
                model_equation,
                table_name,
                cte_name
            )
            ctes.append(cte_sql)
        
        # Create a CTE that joins all constituent model predictions
        join_parts = [f'model_0."{id_column}"']
        select_parts = [f'model_0."{id_column}"']
        
        for i in range(len(self.models)):
            select_parts.append(f'model_{i}.predicted_probability AS model_{i}_prob')
        
        # Build JOIN clause
        join_clause = "model_0"
        for i in range(1, len(self.models)):
            join_clause += f'\n    INNER JOIN model_{i} ON model_0."{id_column}" = model_{i}."{id_column}"'
        
        constituent_predictions_cte = f"""constituent_predictions AS (
    SELECT 
        {',\n        '.join(select_parts)}
    FROM {join_clause}
)"""
        
        # Apply meta-model equation
        meta_constant = self.meta_model.get('constant', 0.0)
        meta_terms = []
        
        for feature_name, coefficient in self.meta_model.items():
            if feature_name != 'constant':
                # Feature names in meta-model should be like "model_0_prob"
                meta_terms.append(f"({feature_name} * {coefficient})")
        
        # Build meta-model linear combination
        if meta_terms:
            meta_linear_combination = f"{meta_constant} + " + " + ".join(meta_terms)
        else:
            meta_linear_combination = str(meta_constant)
        
        # Build meta-model probability formula
        meta_probability_formula = f"1.0 / (1.0 + EXP(-({meta_linear_combination})))"
        
        # Final query
        final_query = f"""SELECT 
    "{id_column}" AS id,
    {meta_probability_formula} AS predicted
FROM constituent_predictions
ORDER BY predicted DESC, "{id_column}" ASC"""
        
        # Combine all CTEs and final query
        all_ctes = ctes + [constituent_predictions_cte]
        full_query = f"WITH {',\n'.join(all_ctes)}\n{final_query};"
        
        return full_query
