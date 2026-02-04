import numpy as np

def sigmoid(z):
    """
    Sigmoid pour la sortie (classification binaire)
    Transforme n'importe quel nombre en probabilité entre 0 et 1
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip pour éviter overflow

def sigmoid_derivative(a):
    """Dérivée de sigmoid (pour backprop)"""
    return a * (1 - a)

def relu(z):
    """
    ReLU pour les couches cachées
    Simple : si négatif → 0, sinon → garde la valeur
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """Dérivée de ReLU (pour backprop)"""
    return (z > 0).astype(float)
