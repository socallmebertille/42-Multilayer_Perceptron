
import argparse

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
    parser.add_argument('--config', type=str, help='Path to config file (.txt)')
    parser.add_argument('--layer', nargs='+', type=int,
                       help='Hidden layer sizes ∈ ℕ*. Ex: --layer 24 24 24')
    parser.add_argument('--epochs', type=int, help='Number of training epochs ∈ ℕ*')
    parser.add_argument('--learning_rate', type=float, help='Learning rate ∈ [0, 1]')
    parser.add_argument('--batch_size', type=int, help='Batch size ∈ ℕ*')
    parser.add_argument('--loss', type=str, help='Loss function',
                       choices=['binaryCrossentropy', 'categoricalCrossentropy'])
    parser.add_argument('--input_size', type=int, help='Input size ∈ [1, +inf] (number of features)')
    parser.add_argument('--output_size', type=int, help='Output size ∈ [1, +inf] (number of classes)')
    parser.add_argument('--activation_hidden', type=str, choices=['sigmoid', 'relu', 'tanh'],
                       help='Activation function for hidden layers')
    parser.add_argument('--activation_output', type=str, choices=['sigmoid', 'softmax', 'linear'],
                       help='Activation function for output layer')
    parser.add_argument('--weights_init', type=str, choices=['heUniform', 'heNormal', 'xavierUniform', 'random'],
                       help='Weights initialization method')
    
    return parser.parse_args()
