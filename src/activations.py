import numpy as np

def sigmoid(z):
    """
    Sigmoid pour la sortie (classification binaire)
    Transforme n'importe quel nombre en probabilité entre 0 et 1
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip pour éviter overflow

def softmax(z):
    """
    Softmax pour la sortie (classification multi-classes)
    Transforme un vecteur de nombres en probabilités qui somment à 1
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # soustraction pour stabilité numérique
    # print(f"exp_z shape: {exp_z.shape}, exp_z sample: {exp_z[0]}")
    
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z):
    """
    ReLU pour les couches cachées
    Simple : si négatif → 0, sinon → garde la valeur
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """Dérivée de ReLU (pour backprop)"""
    return (z > 0).astype(float)
