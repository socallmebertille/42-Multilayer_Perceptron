import sys
import numpy as np
from pathlib import Path
from utils import lire_csv, ecrire_csv

def splitting_phase(dataset_file, train_ratio):
    """
    Splits the dataset into training, validation, and test sets based on the given train ratio.
    """
    if not isinstance(dataset_file, np.ndarray) or not isinstance(train_ratio, tuple):
        print("Error: dataset_file should be a numpy array.")
        return None, None, None
    
    np.random.shuffle(dataset_file) # Shuffle the dataset before splitting

    total_samples = dataset_file.shape[0]
    train_size = int(total_samples * train_ratio[0])
    valid_size = int(total_samples * train_ratio[1])

    train_set = dataset_file[:train_size]
    valid_set = dataset_file[train_size:train_size + valid_size]
    test_set = dataset_file[train_size + valid_size:]

    dossier = Path("datasets")
    dossier.mkdir(exist_ok=True) # Create directory if it doesn't exist

    ecrire_csv(dossier / "train_set.csv", train_set.tolist())
    ecrire_csv(dossier / "valid_set.csv", valid_set.tolist())
    ecrire_csv(dossier / "test_set.csv", test_set.tolist())

    return train_set, valid_set, test_set

def training_phase(dataset_file):
    """
    Trains the MLP model on the training dataset and validates it on the validation set.
    """
    pass

def prediction_phase(dataset_file, model_file):
    """
    Loads the trained model and makes predictions on the test dataset.
    """
    pass

def main():
    """
    Multi-Layer Perceptron (MLP) for binary classification.

        Splitting phase :
            python mlp.py --dataset <dataset_file> --split <train_ratio>
        return :
            > splitting dataset '<dataset_file>' into train, validation & test sets with train ratio = <train_ratio>
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

    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print("Usage: python mlp.py --dataset <dataset_file> --split <train_ratio>")
        print("                     --dataset <dataset_train_file>")
        print("                     --dataset <dataset_test_file> --predict saved_model.npy")
        return 1

    for arg in sys.argv:

        if arg == "--dataset" and len(sys.argv) > sys.argv.index(arg) + 1:
            dataset_file = sys.argv[sys.argv.index(arg) + 1]

            if len(sys.argv) == 3 and not dataset_file == "datasets/train_set.csv":
                if not dataset_file.startswith("datasets/"):
                    print("Error: you have to split the dataset before training.")
                else:
                    print("Error: you have to provide the train_set.csv from 'datasets/' folder.")
                return 1
            if len(sys.argv) > 3 and sys.argv[3] == "--predict" and not dataset_file == "datasets/test_set.csv":
                if not dataset_file.startswith("datasets/"):
                    print("Error: you have to split the dataset before prediction.")
                else:
                    print("Error: you have to provide the test_set.csv from 'datasets/' folder.")
                return 1
            
            load_data = np.array(lire_csv(dataset_file))
            if not load_data.size:
                return 1
            print(f"> dataset '{dataset_file}' of lenght {load_data.shape[0]} loaded from disk...")
            if len(sys.argv) == 3:
                print(load_data[0:2])  # Display first 2 rows as a sample
                training_phase(load_data)

        if arg == "--split" and len(sys.argv) > sys.argv.index(arg) + 1 and load_data is not None:
            train_ratio = tuple(map(float, sys.argv[sys.argv.index(arg) + 1].split(',')))
            if train_ratio[0] + train_ratio[1] >= 1.0:
                print("Error: train_ratio must be in the form x,y where x + y < 1.0")
                return 1
            print(f"> splitting dataset '{dataset_file}' into train, validation & test sets with train ratio = {train_ratio}")
            train_set, valid_set, test_set = splitting_phase(load_data, train_ratio)
            if train_set is None or valid_set is None or test_set is None:
                return 1
            print(f"  > train_set shape : {train_set.shape}")
            print(f"  > valid_set shape : {valid_set.shape}")
            print(f"  > test_set shape  : {test_set.shape}")
            print(f"  > total_set shape : ({train_set.shape[0] + valid_set.shape[0] + test_set.shape[0]}, {train_set.shape[1]})")
            return 1

        if arg == "--predict":
            if not len(sys.argv) > sys.argv.index(arg) + 1:
                print("Error: you have to provide the saved_model.npy file for prediction.")
                return 1
            model_file = sys.argv[sys.argv.index(arg) + 1]
            if model_file is None or not model_file == "saved_model.npy":
                print("Error: you have to provide the saved_model.npy file for prediction.")
                return 1
            print(f"> loading model '{model_file}' from disk...")
            load_model = np.array(lire_csv(model_file))
            if not load_model.size:
                print("Error: you have to train the model before prediction.")
                return 1
            prediction_phase(load_data, load_model)
            return 1

    return 1

if __name__ == "__main__":
    main()
