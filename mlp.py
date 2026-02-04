import argparse
from pathlib import Path
import numpy as np

from src.utils import lire_csv
from src.split_data import splitting_phase
from src.preprocessing import normalize, apply_normalization
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
    parser.add_argument('--layer', nargs='+', type=int,
                       help='Hidden layer sizes ∈ ℕ*. Ex: --layer 24 24 24')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs ∈ ℕ*')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate ∈ [0, 1]')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size ∈ ℕ*')
    parser.add_argument('--loss', type=str, default='binaryCrossentropy',
                       choices=['binaryCrossentropy', 'categoricalCrossentropy'],
                       help='Loss function')
    parser.add_argument('--input_size', type=int,
                       help='Input size ∈ [1, +inf] (number of features)')
    parser.add_argument('--output_size', type=int,
                       help='Output size ∈ [1, +inf] (number of classes)')
    parser.add_argument('--activation_hidden', type=str, default='relu',
                        choices=['sigmoid', 'relu', 'tanh'],
                       help='Activation function for hidden layers')
    parser.add_argument('--activation_output', type=str, default='sigmoid',
                        choices=['sigmoid', 'softmax', 'linear'],
                       help='Activation function for output layer')
    parser.add_argument('--weights_init', type=str, default='heUniform',
                       choices=['heUniform', 'heNormal', 'xavierUniform', 'random'],
                       help='Weights initialization method')
    
    return parser.parse_args()


def parse_config_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    # Valeurs par défaut
    default_config = {
        'network': {
            'input_size': None,  # Sera inféré du dataset
            'layer': [24, 24],
            'output_size': 1,
            'activation_hidden': 'relu',
            'activation_output': 'sigmoid',
            'weights_init': 'heUniform'
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.01,
            'batch_size': 32,
            'loss': 'binaryCrossentropy'
        }
    }
    
    config = {'network': {}, 'training': {}}
    section = None

    try:
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if line.startswith('['):
                    section = line[1:-1]
                    if section not in ['network', 'training']:
                        raise ValueError(f"Unknown section [{section}]")
                    continue

                if '=' not in line:
                    raise ValueError(f"Invalid syntax (expected 'key = value')")

                key, value = map(str.strip, line.split('=', 1))

                if section == 'network':
                    if key == 'layer':
                        layer = [int(x.strip()) for x in value.split(',')]
                        if not layer or any(l <= 0 for l in layer):
                            raise ValueError(f"layers must be positive integers, got {value}")
                        config['network'][key] = layer
                        
                    elif key == 'input_size':
                        size = int(value)
                        if size <= 0:
                            raise ValueError(f"input_size must be > 0, got {size}")
                        config['network'][key] = size
                        
                    elif key == 'output_size':
                        size = int(value)
                        if size <= 0:
                            raise ValueError(f"output_size must be > 0, got {size}")
                        config['network'][key] = size
                        
                    elif key == 'activation_hidden':
                        valid = ['sigmoid', 'relu', 'tanh']
                        if value not in valid:
                            raise ValueError(f"activation_hidden must be one of {valid}, got '{value}'")
                        config['network'][key] = value
                        
                    elif key == 'activation_output':
                        valid = ['sigmoid', 'softmax', 'linear']
                        if value not in valid:
                            raise ValueError(f"activation_output must be one of {valid}, got '{value}'")
                        config['network'][key] = value
                        
                    elif key == 'weights_init':
                        valid = ['heUniform', 'heNormal', 'xavierUniform', 'random']
                        if value not in valid:
                            raise ValueError(f"weights_init must be one of {valid}, got '{value}'")
                        config['network'][key] = value
                    else:
                        print(f"Warning: Unknown network parameter '{key}' ignored")

                elif section == 'training':
                    if key == 'epochs':
                        epochs = int(value)
                        if epochs <= 0:
                            raise ValueError(f"epochs must be > 0, got {epochs}")
                        config['training'][key] = epochs
                        
                    elif key == 'learning_rate':
                        lr = float(value)
                        if lr <= 0 or lr > 1:
                            raise ValueError(f"learning_rate must be in (0, 1], got {lr}")
                        config['training'][key] = lr
                        
                    elif key == 'batch_size':
                        batch = int(value)
                        if batch <= 0:
                            raise ValueError(f"batch_size must be > 0, got {batch}")
                        config['training'][key] = batch
                        
                    elif key == 'loss':
                        valid = ['binaryCrossentropy', 'categoricalCrossentropy']
                        if value not in valid:
                            raise ValueError(f"loss must be one of {valid}, got '{value}'")
                        config['training'][key] = value
                    else:
                        print(f"Warning: Unknown training parameter '{key}' ignored")

    except ValueError as e:
        raise ValueError(f"Error parsing config file at line {line_num}: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error at line {line_num}: {e}")
    
    # Fusionner avec les valeurs par défaut (pour les clés manquantes)
    for section in ['network', 'training']:
        for key, default_value in default_config[section].items():
            if key not in config[section]:
                config[section][key] = default_value
                print(f"Info: Using default {section}.{key} = {default_value}")
    
    print(f"> configuration loaded from '{path}'")
    return config

def validate_config(config, X_shape):
    """Valide la cohérence globale de la config avec les données"""
    errors = []
    
    # Validation réseau
    if config['network']['input_size'] is None:
        config['network']['input_size'] = X_shape[1]
    elif config['network']['input_size'] != X_shape[1]:
        errors.append(
            f"input_size ({config['network']['input_size']}) "
            f"!= dataset features ({X_shape[1]})"
        )
    
    # Validation cohérence loss / activation output
    if config['training']['loss'] == 'categoricalCrossentropy':
        if config['network']['activation_output'] != 'softmax':
            errors.append(
                "categoricalCrossentropy requires softmax output activation, "
                f"got '{config['network']['activation_output']}'"
            )
        if config['network']['output_size'] < 2:
            errors.append(
                f"categoricalCrossentropy requires output_size >= 2, "
                f"got {config['network']['output_size']}. "
                "Try --output_size 2 for binary classification with softmax."
            )
    
    elif config['training']['loss'] == 'categoricalCrossentropy':
        if config['network']['activation_output'] != 'softmax':
            errors.append(
                "categoricalCrossentropy requires softmax output activation, "
                f"got '{config['network']['activation_output']}'"
            )
    
    if errors:
        print("\n❌ Configuration validation errors:")
        for err in errors:
            print(f"  - {err}")
        raise ValueError("Invalid configuration")
    
    return config


def main():
    """
    Multi-Layer Perceptron (MLP) for binary classification.
    """

    args = parse_arguments()
    if Path(args.dataset).is_file() is True:
        data_file = args.dataset
        args.dataset = np.array(lire_csv(args.dataset))
        X = args.dataset[:, 2:].astype(np.float64)
        Y = np.where(args.dataset[:, 1:2] == 'B', 0.0, 1.0).astype(np.float64)
        X, norm_params = normalize(X)
        print(f"✓ X normalized: mean={X.mean():.4f}, std={X.std():.4f}")
    else:
        print(f"Error: the file {args.dataset} does not exist.")
        return
    
    if args.config:
        config = parse_config_file(args.config)
        config = validate_config(config, X.shape)
    else:
        config = {
            'network': {
                'input_size': args.input_size or X.shape[1],
                'layer': args.layer or [24, 24],
                'output_size': args.output_size or 1,
                'activation_hidden': args.activation_hidden or 'relu',
                'activation_output': args.activation_output or 'sigmoid',
                'weights_init': args.weights_init or 'heUniform'
            },
            'training': {
                'epochs': args.epochs or 50,
                'learning_rate': args.learning_rate or 0.01,
                'batch_size': args.batch_size or 32,
                'loss': args.loss or 'binaryCrossentropy'
            }
        }
        config = validate_config(config, X.shape)

    if args.loss == 'categoricalCrossentropy' or config['training']['loss'] == 'categoricalCrossentropy':
        Y = np.hstack((1 - Y, Y))  # Convertir en one-hot pour 2 classes
    
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

        x_valid = valid_set[:, 2:].astype(np.float64)
        x_valid = apply_normalization(x_valid, norm_params)
        y_valid = np.where(valid_set[:, 1:2] == 'B', 0.0, 1.0).astype(np.float64)
        if args.loss == 'categoricalCrossentropy' or config['training']['loss'] == 'categoricalCrossentropy':
            y_valid = np.hstack((1 - y_valid, y_valid))  # Convertir en one-hot pour 2 classes
        print(f"x_train shape : {X.shape}")
        print(f"x_valid shape : {x_valid.shape}")

        mlp.train(X, Y, x_valid, y_valid)
        mlp.save("saved_model.npy")

    return 0

if __name__ == "__main__":
    main()