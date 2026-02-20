import sys
from pathlib import Path
import numpy as np

from src.parsing import parse_arguments
from src.config import merge_config, validate_config
from src.utils import lire_csv
from src.split_data import splitting_phase
from src.preprocessing import normalize, apply_normalization
from src.my_mlp import MyMLP


def load_dataset(path):
    data = np.array(lire_csv(path))
    if data is None or data.size == 0:
        print(f"Error: the dataset '{path}' is empty or could not be loaded.")
        sys.exit(1)
    print(f"{data}") # Affiche les 5 premières lignes du dataset pour vérifier le chargement
    X = data[:, 2:].astype(np.float64)
    Y = np.where(data[:, 1:2] == 'B', 0.0, 1.0).astype(np.float64)
    return X, Y, data

def main():
    """
    Multi-Layer Perceptron (MLP) for binary classification.
    """
    np.random.seed(42) # Pour rendre le comportement aléatoire reproductible
    
    args = parse_arguments()
    data_file = args.dataset
    if not Path(data_file).exists():
        print(f"Error: the file {data_file} does not exist.")
        return 1
    
    if data_file != "datasets/test_set.csv" and args.predict:
        print("Please, split your dataset with the correct flags before predict the set if you have not.")
        print("Or, give the correct set of test.")
        return 1
    if data_file != "datasets/train_set.csv" and not args.split and not args.predict:
        print("Please, split your dataset with the correct flags before training the set if you have not.")
        print("Or, give the correct set of training.")
        return 1

    X, Y, args.dataset = load_dataset(args.dataset)

    config = merge_config(args, X.shape)
    config = validate_config(config)
    
    # Détection du mode
    if args.split:
        # PHASE DE SPLITTING
        args.split = tuple(map(float, args.split.split(',')))
        if sum(args.split) >= 1.0:
            print("Error: the sum of split ratios must be less than 1.0.")
            return 1
        print(f"> splitting dataset '{data_file}' into train, validation & test sets with train ratio = {args.split}")
        splitting_phase(args.dataset, args.split)
        
    elif args.predict:
        # PHASE DE PRÉDICTION
        mlp = MyMLP(config, None)
        mlp.load(args.predict)

        print(f"> loss model : {mlp.config['training']['loss']}")
        
        if mlp.config['training']['loss'] == 'categoricalCrossentropy' and Y.shape[1] == 1: # one-hot Y
            Y = np.hstack((1 - Y, Y))
        mlp.predict(X, Y)
        
    else:
        # PHASE DE TRAINING (par défaut)
        X, norm_params = normalize(X)
        mlp = MyMLP(config, norm_params)
        x_valid, y_valid, valid_set = load_dataset("datasets/valid_set.csv")
        x_valid = apply_normalization(x_valid, norm_params)

        if args.loss == 'categoricalCrossentropy' and Y.shape[1] == 1: # one-hot Y
            Y = np.hstack((1 - Y, Y))  # Convertir en one-hot pour 2 classes
            y_valid = np.hstack((1 - y_valid, y_valid))  # Convertir en one-hot pour 2 classes
        
        print(f"x_train shape : {X.shape}")
        print(f"x_valid shape : {x_valid.shape}")

        mlp.train(X, Y, x_valid, y_valid)
        mlp.save("saved_model.npy")

    return 0

if __name__ == "__main__":
    sys.exit(main())