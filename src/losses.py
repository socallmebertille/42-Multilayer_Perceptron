import numpy as np

def binary_crossentropy(y_true, y_pred):
    """
    Mesure l'erreur entre prédiction et vérité
    y_true: [0, 1, 1, 0, ...] (vrais labels)
    y_pred: [0.2, 0.9, 0.8, 0.1, ...] (prédictions)
    """
    epsilon = 1e-15  # Pour éviter log(0) qui fait -inf
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_derivative(y_true, y_pred):
    """Dérivée de BCE (pour backprop)"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred))

def categorical_crossentropy(y_true, y_pred):
    """
    Categorical Crossentropy pour multi-classes
    y_true: one-hot encoded vectors
    y_pred: predicted probabilities for each class
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def categorical_crossentropy_derivative(y_true, y_pred):
    """Dérivée de CCE (pour backprop)"""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true / y_pred) / y_true.shape[0]