import numpy as np

def normalize(X):
    """Normalisation Z-score (standardization)"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Éviter division par 0
    X_norm = (X - mean) / std
    return X_norm, {'mean': mean, 'std': std}

def apply_normalization(X, params):
    """Applique la même normalisation"""
    return (X - params['mean']) / params['std']
