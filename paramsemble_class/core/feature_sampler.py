"""Feature sampling module for generating diverse feature combinations."""

from typing import List, Optional
import numpy as np
from scipy.special import comb


class FeatureSampler:
    """
    Generates diverse feature combinations for ensemble model training.
    
    Supports two sampling methods:
    - "unique": Features cannot repeat within a featureset (combinations)
    - "replace": Features can repeat within a featureset (combinations with replacement)
    
    Parameters
    ----------
    n_features : int
        Total number of features available
    f : int
        Number of features per combination
    m : int
        Number of combinations to generate
    sample : str
        Sampling method: "unique" or "replace"
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_features: int,
        f: int,
        m: int,
        sample: str,
        random_state: Optional[int] = None
    ):
        self.n_features = n_features
        self.f = f
        self.m = m
        self.sample = sample
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
    def generate_combinations(self) -> List[List[int]]:
        """
        Generate m feature combinations.
        
        Returns
        -------
        List[List[int]]
            List of feature index lists, where each inner list contains f feature indices
        """
        max_combinations = self._calculate_max_combinations()
        
        # Override m if it exceeds maximum possible combinations
        actual_m = min(self.m, max_combinations)
        
        if self.sample == "unique":
            return self._generate_unique_combinations(actual_m)
        else:  # sample == "replace"
            return self._generate_replace_combinations(actual_m)
    
    def _calculate_max_combinations(self) -> int:
        """
        Calculate maximum possible combinations.
        
        For unique sampling: C(n_features, f) = n! / (f! * (n-f)!)
        For replace sampling: C(n_features + f - 1, f) = multicombinations
        
        Note: For "replace" sampling, order doesn't matter in feature selection,
        so we use the multicombination formula (combinations with replacement where
        order is irrelevant), not n^f.
        
        Returns
        -------
        int
            Maximum number of possible combinations
        """
        if self.sample == "unique":
            # Use scipy.special.comb for combinations
            return int(comb(self.n_features, self.f, exact=True))
        else:  # sample == "replace"
            # Use multicombination formula: C(n + f - 1, f)
            # This counts combinations with replacement where order doesn't matter
            return int(comb(self.n_features + self.f - 1, self.f, exact=True))
    
    def _generate_unique_combinations(self, actual_m: int) -> List[List[int]]:
        """
        Generate combinations without replacement (no duplicates within featureset).
        
        Parameters
        ----------
        actual_m : int
            Number of combinations to generate (already capped at maximum)
            
        Returns
        -------
        List[List[int]]
            List of unique feature combinations
        """
        combinations = []
        seen = set()
        
        # If we need all possible combinations, generate them systematically
        max_combinations = self._calculate_max_combinations()
        if actual_m == max_combinations and max_combinations <= 10000:
            # Generate all combinations systematically for small sets
            from itertools import combinations as iter_combinations
            all_combos = list(iter_combinations(range(self.n_features), self.f))
            # Shuffle to maintain randomness
            self.rng.shuffle(all_combos)
            return [list(combo) for combo in all_combos[:actual_m]]
        
        # Otherwise, generate randomly until we have enough unique combinations
        max_attempts = actual_m * 100  # Prevent infinite loops
        attempts = 0
        
        while len(combinations) < actual_m and attempts < max_attempts:
            # Generate a random combination
            combo = sorted(self.rng.choice(self.n_features, size=self.f, replace=False))
            combo_tuple = tuple(combo)
            
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                combinations.append(list(combo))
            
            attempts += 1
        
        return combinations
    
    def _generate_replace_combinations(self, actual_m: int) -> List[List[int]]:
        """
        Generate combinations with replacement (duplicates allowed within featureset).
        
        Parameters
        ----------
        actual_m : int
            Number of combinations to generate (already capped at maximum)
            
        Returns
        -------
        List[List[int]]
            List of feature combinations (may contain duplicate features)
        """
        max_combinations = self._calculate_max_combinations()
        
        # If we need all possible combinations, generate them systematically
        if actual_m == max_combinations and max_combinations <= 10000:
            # Generate all multicombinations systematically
            from itertools import combinations_with_replacement
            all_combos = list(combinations_with_replacement(range(self.n_features), self.f))
            # Shuffle to maintain randomness
            self.rng.shuffle(all_combos)
            return [list(combo) for combo in all_combos[:actual_m]]
        
        # Otherwise, generate randomly
        combinations = []
        seen = set()
        
        max_attempts = actual_m * 1000  # Increase attempts for better coverage
        attempts = 0
        
        while len(combinations) < actual_m and attempts < max_attempts:
            # Generate a random combination with replacement
            combo = sorted(self.rng.choice(self.n_features, size=self.f, replace=True))
            combo_tuple = tuple(combo)
            
            if combo_tuple not in seen:
                seen.add(combo_tuple)
                combinations.append(list(combo))
            
            attempts += 1
        
        return combinations
