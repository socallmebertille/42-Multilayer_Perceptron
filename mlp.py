import sys
from utils import lire_csv, ecrire_csv

def splitting_phase(dataset_file, train_ratio):
    """
    Splits the dataset into training, validation, and test sets based on the given train ratio.
    """
    pass

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
            load_data = lire_csv(dataset_file)
            if not load_data:
                return 1
        if arg == "--split" and len(sys.argv) > sys.argv.index(arg) + 1:
            train_ratio = float(sys.argv[sys.argv.index(arg) + 1])
            print(f"> splitting dataset '{dataset_file}' into train, validation & test sets with train ratio = {train_ratio}")
            splitting_phase(load_data, train_ratio)
            return 1
        if arg == "--predict" and len(sys.argv) > sys.argv.index(arg) + 1:
            model_file = sys.argv[sys.argv.index(arg) + 1]
            print(f"> loading model '{model_file}' from disk...")
            load_model = lire_csv(model_file)
            if not load_model:
                return 1
            prediction_phase(load_data, load_model)
            return 1
        if arg == "--dataset" and len(sys.argv) > sys.argv.index(arg) + 1:
            training_phase(load_data)
    return 1

if __name__ == "__main__":
    main()
