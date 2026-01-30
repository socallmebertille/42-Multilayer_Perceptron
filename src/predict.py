import numpy as np
from src.utils import binary_class

def prediction_phase(test_set, model_file):
    """
    Loads the trained model and makes predictions on the test dataset.

    Prediction phase :
            python mlp.py --dataset <dataset_test_file> --predict saved_model.npy
        return :
            > loading model 'saved_model.npy' from disk...
                # Somme de proba_class_0 + proba_class_1 = 1 avec fonction d'activation softmax mais pas
                # necessairement avec sigmoid
            -> (y_hat, y_true) - raw[ proba_class_0 proba_class_1 ]
                ...
            -> (y_hat, y_true) - raw[ proba_class_0 proba_class_1 ]
            > correctly predicted : (x/nb_samples)
            > loss (mean squarred error) : x
            > loss (binary crossentropy) : x
    """
    
    x_test = test_set[:, 2:]
    y_test = binary_class(test_set[:, 1:2], 'B', 'M')
    nb_samples = x_test.shape[0]

    y_hat = np.zeros(nb_samples)  # Dummy predictions

    for i in range(nb_samples):
        # Prediction logic here
        print(f"-> ({int(y_test[i][0])}, {int(y_hat[i])}) - raw[ {0} {0} ]")

    correct_predictions = sum(1 for i in range(nb_samples) if y_hat[i] == y_test[i])
    print(f"> correctly predicted : ({correct_predictions}/{nb_samples})")
    print(f"> loss (mean squarred error) : {0}")
    print(f"> loss (binary crossentropy) : {0}")