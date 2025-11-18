#!/usr/bin/env python3
"""
Configuration management for API keys and settings
"""

import os
import json
import random
import itertools
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

def get_sambanova_api_keys():
    """Get SambaNova API keys from environment variables"""
    keys_str = os.getenv('SAMBANOVA_API_KEYS', '')
    if not keys_str:
        raise ValueError("SAMBANOVA_API_KEYS not found in environment variables")
    
    # Split by comma and strip whitespace
    keys = [key.strip() for key in keys_str.split(',') if key.strip()]
    
    if not keys:
        raise ValueError("No valid SambaNova API keys found in SAMBANOVA_API_KEYS")
    
    return keys


class RotatingAPIKey(str):
    """
    A string subclass that auto-rotates through multiple API keys on each access.
    Inherits from str to pass isinstance(obj, str) checks.
    """
    def __new__(cls, api_keys):
        # Create a string instance with the first API key
        if not api_keys:
            raise ValueError("api_keys list cannot be empty")
        
        # Create the string instance with the first key (for isinstance checks)
        instance = str.__new__(cls, api_keys[0])
        return instance
    
    def __init__(self, api_keys):
        # Don't call super().__init__() as str is immutable
        self.api_keys = api_keys.copy()  # Make a copy to avoid modifying original
        random.shuffle(self.api_keys)
        self.cycle = itertools.cycle(self.api_keys)
        # Initialize with the first key
        self._current_key = next(self.cycle)
    
    def _rotate_and_get(self):
        """Get current key and rotate to next for the next access"""
        current = self._current_key
        self._current_key = next(self.cycle)
        return current
    
    def __str__(self):
        """Auto-rotate on string conversion - this is likely when the key is actually used"""
        return self._rotate_and_get()
    
    def __repr__(self):
        """For debugging - doesn't rotate"""
        return f"RotatingAPIKey(current='{self._current_key}', total={len(self.api_keys)} keys)"
    
    # Override methods that might be used to access the key value
    def __format__(self, format_spec):
        """Auto-rotate when formatted"""
        return format(self._rotate_and_get(), format_spec)
    
    def encode(self, encoding='utf-8', errors='strict'):
        """Auto-rotate when encoded"""
        return self._rotate_and_get().encode(encoding, errors)
    
    # For methods that check properties without using the value, use current key without rotating
    def startswith(self, prefix, start=None, end=None):
        """Check startswith using current key without rotating"""
        if start is not None and end is not None:
            return self._current_key.startswith(prefix, start, end)
        elif start is not None:
            return self._current_key.startswith(prefix, start)
        else:
            return self._current_key.startswith(prefix)
    
    def __hash__(self):
        """Use current key for hashing without rotating"""
        return hash(self._current_key)
    
    def __len__(self):
        """Return length without rotating"""
        return len(self._current_key)
    
    def __getitem__(self, key):
        """Rotate when indexed"""
        return self._rotate_and_get()[key]
    
    def __contains__(self, item):
        """Check containment without rotating"""
        return item in self._current_key


def process_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load and process data from a JSONL file.
    
    Args:
        data_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    print(f"Loaded {len(data)} samples from {data_path}")
    return data


class DataProcessor:
    """
    Processor for handling data evaluation and accuracy computation.
    """
    
    def __init__(self, eval_metric: str = "exact_match"):
        """
        Initialize the data processor.
        
        Args:
            eval_metric: The evaluation metric to use ('exact_match', 'contains', etc.)
        """
        self.eval_metric = eval_metric
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if a predicted answer matches the ground truth.
        
        Args:
            predicted: The predicted answer
            ground_truth: The ground truth answer
            
        Returns:
            Boolean indicating if the answer is correct
        """
        if predicted is None or ground_truth is None:
            return False
        
        # Normalize strings
        pred_normalized = str(predicted).strip().lower()
        gt_normalized = str(ground_truth).strip().lower()
        
        if self.eval_metric == "exact_match":
            return pred_normalized == gt_normalized
        elif self.eval_metric == "contains":
            return gt_normalized in pred_normalized
        else:
            # Default to exact match
            return pred_normalized == gt_normalized
    
    def evaluate_accuracy(self, predicted: List[str], ground_truth: List[str]) -> tuple:
        """
        Evaluate accuracy across a list of predictions.
        
        Args:
            predicted: List of predicted answers
            ground_truth: List of ground truth answers
            
        Returns:
            Tuple of (accuracy, num_correct)
        """
        if len(predicted) != len(ground_truth):
            raise ValueError(f"Length mismatch: predicted has {len(predicted)} items, ground_truth has {len(ground_truth)} items")
        
        if len(predicted) == 0:
            return 0.0, 0
        
        correct = sum(
            self.answer_is_correct(pred, gt) 
            for pred, gt in zip(predicted, ground_truth)
        )
        
        accuracy = correct / len(predicted)
        return accuracy, correct