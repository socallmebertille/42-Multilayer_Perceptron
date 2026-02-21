from pathlib import Path

DEFAULT_CONFIG = {
    'network': {
        'layer': [24, 24],
        'activation_hidden': 'relu',
        'weights_init': 'heUniform'
    },
    'training': {
        'epochs': 100,
        'learning_rate': 0.01,
        'batch_size': 32,
        'loss': 'binaryCrossentropy'
    }
}

def parse_config_file(path):
    config = {'network': {}, 'training': {}}
    section = None
    
    with open(path) as f:

        for line in f:
            line = line.strip() # strip : supprimer les espaces au début et à la fin de la ligne

            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                section = line[1:-1]
                continue

            key, value = map(str.strip, line.split("="))

            if key == "layer":
                value = [int(x) for x in value.split(",")]
            elif key in ["epochs", "batch_size"]:
                value = int(value)
            elif key == "learning_rate":
                value = float(value)

            config[section][key] = value

    return config


def merge_config(args, X_shape):

    config = {
        'network': DEFAULT_CONFIG['network'].copy(),
        'training': DEFAULT_CONFIG['training'].copy()
    }

    if args.config:
        file_config = parse_config_file(args.config)
        config['network'].update(file_config['network']) # update : ajoute ou remplace seulement les clés présentes dans file_config['network'], mais conserve le reste
        config['training'].update(file_config['training'])

    if args.layer is not None:
        config['network']['layer'] = args.layer
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.loss is not None:
        config['training']['loss'] = args.loss
    if args.activation_hidden is not None:
        config['network']['activation_hidden'] = args.activation_hidden
    if args.weights_init is not None:
        config['network']['weights_init'] = args.weights_init
        
    # Auto-set based on loss and dataset shape
    config['network']['input_size'] = X_shape[1]
    
    if config['training']['loss'] == 'binaryCrossentropy':
        config['network']['output_size'] = 1
        config['network']['activation_output'] = 'sigmoid'
    elif config['training']['loss'] == 'categoricalCrossentropy':
        config['network']['output_size'] = 2
        config['network']['activation_output'] = 'softmax'

    return config


def validate_config(config):

    net = config['network']
    train = config['training']

    if any(l <= 0 for l in net['layer']):
        raise ValueError("Invalid layer size")
    if net['activation_hidden'] not in ['sigmoid', 'relu']:
        raise ValueError("Invalid activation_hidden")
    if net['weights_init'] not in ['heUniform', 'heNormal', 'xavierUniform', 'xavierNormal', 'random']:
        raise ValueError("Invalid weights_init")
    if train['epochs'] <= 0:
        raise ValueError("Invalid epochs")
    if not (0 < train['learning_rate'] <= 1):
        raise ValueError("Invalid learning_rate")
    if train['batch_size'] <= 0:
        raise ValueError("Invalid batch_size")
    if train['loss'] not in ['binaryCrossentropy', 'categoricalCrossentropy']:
        raise ValueError("Invalid loss")

    return config
