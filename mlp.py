import argparse
from pathlib import Path
import numpy as np

from src.utils import lire_csv, binary_class
from src.split_data import splitting_phase
from src.my_mlp import MyMLP

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Multilayer Perceptron for binary classification'
    )
    
    # Argument obligatoire
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset CSV file')
    
    # Mode selection
    parser.add_argument('--split', type=str,
                       help='Split ratio (format: train,valid). Ex: 0.7,0.15')
    parser.add_argument('--predict', type=str,
                       help='Path to saved model for prediction')
    
    # Training parameters
    parser.add_argument('--config', type=str,
                       help='Path to config file (.txt)')
    parser.add_argument('--layers', nargs='+', type=int,
                       help='Hidden layer sizes. Ex: --layers 24 24 24')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--loss', type=str, default='binary_crossentropy',
                       choices=['binary_crossentropy', 'categorical_crossentropy'],
                       help='Loss function')
    
    return parser.parse_args()


def parse_config_file(path):
    config = {'network': {}, 'training': {}}
    section = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('['):
                section = line[1:-1]
                continue

            key, value = map(str.strip, line.split('=', 1))

            if section == 'network':
                if key == 'layers':
                    value = list(map(int, value.split(',')))
                config['network'][key] = value

            elif section == 'training':
                if key in ('epochs', 'batch_size'):
                    value = int(value)
                elif key == 'learning_rate':
                    value = float(value)
                config['training'][key] = value

    print(f"> configuration loaded from '{path}': \n{config}")
    return config


def main():
    """
    Multi-Layer Perceptron (MLP) for binary classification.
    """

    args = parse_arguments()
    if Path(args.dataset).is_file() is True:
        data_file = args.dataset
        args.dataset = np.array(lire_csv(args.dataset))
        X = args.dataset[:, 2:]
        Y = binary_class(args.dataset[:, 1:2], 'B', 'M')
    else:
        print(f"Error: the file {args.dataset} does not exist.")
        return
    
    if args.config:
        config = parse_config_file(args.config)
    else:
        config = {
            'network': {
                'input_size': X.shape[1],
                'layers': args.layers or [24, 24],
                'output_size': 1,
                'activation_hidden': 'relu',
                'activation_output': 'sigmoid'
            },
            'training': {
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'loss': args.loss
            }
        }

    mlp = MyMLP(config)

    # Détection du mode
    if args.split:
        # PHASE DE SPLITTING
        args.split = tuple(map(float, args.split.split(',')))
        print(f"> splitting dataset '{data_file}' into train, validation & test sets with train ratio = {args.split}")
        splitting_phase(args.dataset, args.split)
        
    elif args.predict:
        # PHASE DE PRÉDICTION
        print(f"> loading model '{args.predict}' from disk...")
        mlp.load(args.predict)
        mlp.predict(X, Y)
        
    else:
        # PHASE DE TRAINING (par défaut)
        valid_set = np.array(lire_csv("datasets/valid_set.csv"))
        if valid_set is None:
            print("Error: valid_set is None.")
            return

        x_valid = valid_set[:, 2:]
        y_valid = binary_class(valid_set[:, 1:2], 'B', 'M')
        print(f"x_train shape : {X.shape}")
        print(f"x_valid shape : {x_valid.shape}")

        mlp.train(X, Y, x_valid, y_valid)
        mlp.save("saved_model.npy")

    return 0

if __name__ == "__main__":
    main()