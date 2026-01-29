import numpy as np
from src.utils import lire_csv, ecrire_csv

def training_phase(train_set, config):
    """
    Trains the MLP model on the training dataset and validates it on the validation set.

    Training phase : 
            python mlp.py --dataset <dataset_train_file>
        return :
            x_train shape : (m, n)
            x_valid shape : (m_val, n)
                # Training history
            epoch 0/nb_epochs - loss: x - val_loss: x
            ...
            epoch nb_epochs/nb_epochs - loss: x - val_loss: x
                # File wich contains the trained model parameters with all weights
            > saving model './saved_model.npy' to disk...
        display :
            a graph of 2 curves (training loss and validation loss) with "epochs" in x-axis and "loss" in y-axis
    """

    valid_set = np.array(lire_csv("datasets/valid_set.csv"))
    if train_set is None or valid_set is None:
        print("Error: train_set or valid_set is None.")
        return

    x_train = train_set[:, 2:]
    y_train = train_set[:, 1:2]
    x_valid = valid_set[:, 2:]
    y_valid = valid_set[:, 1:2]
    print(f"x_train shape : {x_train.shape}")
    print(f"x_valid shape : {x_valid.shape}")

    epochs = 10

    for epoch in range(epochs):
        # Training logic here
        print(f"epoch {epoch}/{epochs} - loss: {0} - val_loss: {0}")

    return