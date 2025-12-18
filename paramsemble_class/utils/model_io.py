"""Model I/O utilities for JSON export and import."""
import json
import os
from typing import Dict, List, Optional, Any


class ModelIO:
    """
    Handles JSON export and import for ELR models.
    
    Provides methods to:
    - Export all constituent model metrics to JSON (elr2json)
    - Export selected model equations to JSON (modeljson)
    - Load model configurations from JSON
    
    The JSON structure is designed to support SQL equation reconstruction
    and model scoring in production environments.
    """
    
    @staticmethod
    def export_all_models(models_data: List[Dict[str, Any]], filepath: str) -> None:
        """
        Export all constituent model metrics to JSON (elr2json).
        
        This method saves comprehensive metrics for all trained constituent models,
        including PLR, FNR, DRP, and equation dictionaries. This is useful for
        model analysis and comparison.
        
        Parameters
        ----------
        models_data : List[Dict[str, Any]]
            List of dictionaries, each containing:
            - 'plr': Positive Likelihood Ratio (float)
            - 'fnr': False Negative Rate (float)
            - 'drp': Decile Ranked Performance (float)
            - 'equation_dict': Dictionary mapping feature names to coefficients
            - 'feature_indices': List of feature indices used (optional)
        filepath : str
            Path where the JSON file will be saved.
        
        Raises
        ------
        IOError
            If file cannot be written with descriptive message.
        ValueError
            If models_data is empty or invalid.
        
        Examples
        --------
        >>> models_data = [
        ...     {
        ...         'plr': 2.5,
        ...         'fnr': 0.15,
        ...         'drp': 1.8,
        ...         'equation_dict': {'feature1': 0.5, 'feature2': -0.3, 'constant': 0.1}
        ...     }
        ... ]
        >>> ModelIO.export_all_models(models_data, 'models.json')
        """
        if not models_data:
            raise ValueError("models_data cannot be empty. Provide at least one model to export.")
        
        try:
            # Convert sets to lists for JSON serialization
            serializable_data = []
            for model in models_data:
                model_copy = {}
                
                for key, value in model.items():
                    # Convert sets to lists
                    if isinstance(value, set):
                        model_copy[key] = [int(x) if hasattr(x, 'item') else x for x in value]
                    # Convert numpy types to Python types
                    elif hasattr(value, 'item'):  # numpy scalar
                        model_copy[key] = value.item()
                    # Convert lists with numpy types
                    elif isinstance(value, list):
                        model_copy[key] = [x.item() if hasattr(x, 'item') else x for x in value]
                    # Convert dicts (like equation_dict) with numpy types
                    elif isinstance(value, dict):
                        model_copy[key] = {k: v.item() if hasattr(v, 'item') else v for k, v in value.items()}
                    else:
                        model_copy[key] = value
                
                serializable_data.append(model_copy)
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Write to JSON file
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
        except (IOError, OSError) as e:
            raise IOError(
                f"Failed to write model data to '{filepath}'. "
                f"Error: {str(e)}"
            )
        except Exception as e:
            raise IOError(
                f"Unexpected error while exporting models to '{filepath}'. "
                f"Error: {str(e)}"
            )
    
    @staticmethod
    def export_selected_models(
        method: str,
        d: int,
        selected_models: List[Dict[str, float]],
        meta_model: Optional[Dict[str, float]],
        filepath: str
    ) -> None:
        """
        Export selected model equations to JSON (modeljson).
        
        This method saves the configuration needed for model scoring,
        including the ensemble method, decile parameter, and equation
        dictionaries for selected models. For ensemble method, it also
        includes the meta-model equation.
        
        Parameters
        ----------
        method : str
            Ensemble method: "intersect", "venn", or "ensemble".
        d : int
            Number of top deciles to consider (1-10).
        selected_models : List[Dict[str, float]]
            List of equation dictionaries for selected constituent models.
            Each dictionary maps feature names to coefficients with
            'constant' key for intercept.
        meta_model : Dict[str, float], optional
            Meta-model equation dictionary (only for ensemble method).
            Maps constituent model probability names to coefficients.
        filepath : str
            Path where the JSON file will be saved.
        
        Raises
        ------
        IOError
            If file cannot be written with descriptive message.
        ValueError
            If method is invalid or selected_models is empty.
        
        Examples
        --------
        >>> selected_models = [
        ...     {'feature1': 0.5, 'feature2': -0.3, 'constant': 0.1},
        ...     {'feature1': 0.7, 'feature3': 0.2, 'constant': -0.2}
        ... ]
        >>> ModelIO.export_selected_models(
        ...     method='intersect',
        ...     d=2,
        ...     selected_models=selected_models,
        ...     meta_model=None,
        ...     filepath='model_config.json'
        ... )
        """
        # Validate method
        valid_methods = ["intersect", "venn", "ensemble"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method '{method}'. Must be one of {valid_methods}."
            )
        
        # Validate selected_models
        if not selected_models:
            raise ValueError(
                "selected_models cannot be empty. Provide at least one model to export."
            )
        
        # Validate d parameter
        if not (1 <= d <= 10):
            raise ValueError(
                f"Parameter 'd' must be between 1 and 10 inclusive, got {d}."
            )
        
        # Validate meta_model for ensemble method before trying to write
        if method == "ensemble":
            if meta_model is None:
                raise ValueError(
                    "meta_model is required for ensemble method but was not provided."
                )
        
        try:
            # Build JSON structure
            model_config = {
                "method": method,
                "d": d,
                "models": selected_models
            }
            
            # Add meta-model for ensemble method
            if method == "ensemble":
                model_config["meta_model"] = meta_model
            
            # Create directory if it doesn't exist
            directory = os.path.dirname(filepath)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Write to JSON file
            with open(filepath, 'w') as f:
                json.dump(model_config, f, indent=2)
                
        except (IOError, OSError) as e:
            raise IOError(
                f"Failed to write model configuration to '{filepath}'. "
                f"Error: {str(e)}"
            )
        except Exception as e:
            raise IOError(
                f"Unexpected error while exporting model configuration to '{filepath}'. "
                f"Error: {str(e)}"
            )
    
    @staticmethod
    def load_model(filepath: str) -> Dict[str, Any]:
        """
        Load model configuration from JSON file.
        
        Reads a model configuration JSON file (modeljson format) and
        returns the parsed dictionary containing method, d parameter,
        constituent model equations, and meta-model equation (if applicable).
        
        Parameters
        ----------
        filepath : str
            Path to the JSON file to load.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'method': Ensemble method string
            - 'd': Decile parameter (int)
            - 'models': List of equation dictionaries
            - 'meta_model': Meta-model equation dict (only for ensemble method)
        
        Raises
        ------
        IOError
            If file cannot be read with descriptive message.
        ValueError
            If JSON structure is invalid or missing required fields.
        
        Examples
        --------
        >>> config = ModelIO.load_model('model_config.json')
        >>> print(config['method'])
        'intersect'
        >>> print(len(config['models']))
        10
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise IOError(
                f"Model configuration file not found: '{filepath}'. "
                "Please provide a valid file path."
            )
        
        try:
            # Read JSON file
            with open(filepath, 'r') as f:
                model_config = json.load(f)
            
            # Validate required fields
            required_fields = ['method', 'd', 'models']
            missing_fields = [field for field in required_fields if field not in model_config]
            
            if missing_fields:
                raise ValueError(
                    f"Invalid model configuration: missing required fields {missing_fields}. "
                    f"Expected fields: {required_fields}"
                )
            
            # Validate method
            valid_methods = ["intersect", "venn", "ensemble"]
            if model_config['method'] not in valid_methods:
                raise ValueError(
                    f"Invalid method '{model_config['method']}' in configuration. "
                    f"Must be one of {valid_methods}."
                )
            
            # Validate d parameter
            if not isinstance(model_config['d'], int) or not (1 <= model_config['d'] <= 10):
                raise ValueError(
                    f"Invalid 'd' parameter in configuration: {model_config['d']}. "
                    "Must be an integer between 1 and 10 inclusive."
                )
            
            # Validate models list
            if not isinstance(model_config['models'], list) or not model_config['models']:
                raise ValueError(
                    "Invalid 'models' field in configuration. "
                    "Must be a non-empty list of equation dictionaries."
                )
            
            # Validate meta_model for ensemble method
            if model_config['method'] == 'ensemble':
                if 'meta_model' not in model_config:
                    raise ValueError(
                        "Invalid configuration for ensemble method: "
                        "'meta_model' field is required but missing."
                    )
                if not isinstance(model_config['meta_model'], dict):
                    raise ValueError(
                        "Invalid 'meta_model' field in configuration. "
                        "Must be a dictionary mapping features to coefficients."
                    )
            
            return model_config
            
        except json.JSONDecodeError as e:
            raise IOError(
                f"Failed to parse JSON from '{filepath}'. "
                f"The file may be corrupted or contain invalid JSON. "
                f"Error: {str(e)}"
            )
        except (IOError, OSError) as e:
            raise IOError(
                f"Failed to read model configuration from '{filepath}'. "
                f"Error: {str(e)}"
            )
