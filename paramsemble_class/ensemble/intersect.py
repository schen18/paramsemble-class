"""Intersect ensemble method for ELR."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any


class IntersectMethod:
    """
    Intersect ensemble method for identifying high-confidence predictions.
    
    The intersect method selects top-performing constituent models and
    identifies IDs that appear in multiple models' Decile Ranked Sets (DRS).
    This approach finds predictions supported by diverse feature combinations.
    
    The method:
    1. Ranks models by PLR (higher better), FNR (lower better), DRP (higher better)
    2. Selects top n models based on spread parameter
    3. Filters models that outperform baseline
    4. Compiles DRS from selected models
    5. Counts occurrences of each ID across model DRS
    
    Returns a DataFrame with deduplicated IDs and their occurrence counts.
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
        # Normalize each metric and combine
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
        constituent_results: List[Dict[str, Any]],
        baseline_results: Dict[str, Any],
        spread: int
    ) -> pd.DataFrame:
        """
        Select top models and combine their Decile Ranked Sets.
        
        Parameters
        ----------
        constituent_results : List[Dict[str, Any]]
            List of dictionaries containing metrics for each constituent model.
            Each dict must have keys: 'plr', 'fnr', 'drp', 'drs', 'dps'
        baseline_results : Dict[str, Any]
            Dictionary containing baseline model metrics.
            Must have keys: 'plr', 'fnr', 'drp', 'drs', 'dps'
        spread : int
            Number of top models to select.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'id': Deduplicated IDs from selected model DRS
            - 'sets': Count of how many selected model DRS contain each ID
        """
        # Rank models
        ranked_indices = IntersectMethod._rank_models(
            constituent_results,
            baseline_results
        )
        
        # Select top n models (or all if fewer than n outperform baseline)
        n_to_select = min(spread, len(ranked_indices))
        selected_indices = ranked_indices[:n_to_select]
        
        # If no models selected, return empty DataFrame
        if not selected_indices:
            return pd.DataFrame(columns=['id', 'sets'])
        
        # Compile DRS from selected models
        id_counts = {}
        for idx in selected_indices:
            drs = constituent_results[idx]['drs']
            for id_value in drs:
                id_counts[id_value] = id_counts.get(id_value, 0) + 1
        
        # Create DataFrame
        if not id_counts:
            return pd.DataFrame(columns=['id', 'sets'])
        
        df = pd.DataFrame([
            {'id': id_value, 'sets': count}
            for id_value, count in id_counts.items()
        ])
        
        # Sort by sets descending, then by id for consistency
        df = df.sort_values(['sets', 'id'], ascending=[False, True]).reset_index(drop=True)
        
        return df
