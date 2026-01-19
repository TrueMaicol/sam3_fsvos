"""
RandomStateManager - Handles saving and restoring random state across SLURM job boundaries
for reproducible evaluation following the official MiniVSPW benchmark protocol.
"""

import pickle
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


class RandomStateManager:
    """
    Manages random state for reproducible multi-run evaluation.
    
    Usage:
        manager = RandomStateManager(save_dir='/path/to/states')
        
        # Run 0: Initialize from seed
        manager.initialize_seed(seed=2020)
        
        # Runs 1-4: Load state from previous run
        manager.load_state(fold=0, run_number=2)  # loads state from run 1
        
        # After evaluation: Save state
        manager.save_state(fold=0, run_number=2)
    """
    
    def __init__(self, save_dir: str):
        """
        Args:
            save_dir: Directory where random state files will be saved/loaded
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def initialize_seed(self, seed: int):
        """
        Initialize all random number generators from a seed.
        Call this for run 0 only.
        """
        cudnn.benchmark = False
        cudnn.deterministic = True
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        print(f"Initialized random state with seed {seed}")
    
    def save_state(self, fold: int, run_number: int):
        """
        Save complete random state after finishing a run.
        """
        state = {
            'numpy': np.random.get_state(),
            'python': random.getstate(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        
        filepath = os.path.join(self.save_dir, f'state_fold{fold}_run{run_number}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Saved random state to {filepath}")
    
    def load_state(self, fold: int, run_number: int):
        """
        Load random state from a previous run.
        Loads state from run_number - 1.
        """
        prev_run = run_number
        filepath = os.path.join(self.save_dir, f'state_fold{fold}_run{prev_run}.pkl')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Random state file not found: {filepath}\n"
                f"Make sure run {prev_run} completed successfully."
            )
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        np.random.set_state(state['numpy'])
        random.setstate(state['python'])
        torch.set_rng_state(state['torch'])
        
        if state['torch_cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['torch_cuda'])
        
        # Set cudnn settings for consistency
        cudnn.benchmark = False
        cudnn.deterministic = True
        
        print(f"Loaded random state from {filepath}")
    
    def get_state_filepath(self, fold: int, run_number: int) -> str:
        """Return the filepath for a given fold and run."""
        return os.path.join(self.save_dir, f'state_fold{fold}_run{run_number}.pkl')
    
    def state_exists(self, fold: int, run_number: int) -> bool:
        """Check if a state file exists for a given fold and run."""
        return os.path.exists(self.get_state_filepath(fold, run_number))