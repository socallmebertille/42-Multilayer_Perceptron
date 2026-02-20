from pathlib import Path

DEFAULT_CONFIG = {
    'network': {
        'input_size': None,
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
            elif key in ["input_size", "output_size", "epochs", "batch_size"]:
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

    activation_output_set = False
    output_size_set = False

    if args.config:
        file_config = parse_config_file(args.config)
        config['network'].update(file_config['network']) # update : ajoute ou remplace seulement les clés présentes dans file_config['network'], mais conserve le reste
        config['training'].update(file_config['training'])
        activation_output_set = 'activation_output' in file_config['network']
        output_size_set = 'output_size' in file_config['network']

    if args.layer:
        config['network']['layer'] = args.layer
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.loss:
        config['training']['loss'] = args.loss
    if args.activation_hidden:
        config['network']['activation_hidden'] = args.activation_hidden
    if args.activation_output:
        config['network']['activation_output'] = args.activation_output
        activation_output_set = True
    if args.weights_init:
        config['network']['weights_init'] = args.weights_init
        
    if args.input_size:
        config['network']['input_size'] = args.input_size
    else:
        config['network']['input_size'] = X_shape[1]
    if args.output_size:
        config['network']['output_size'] = args.output_size
        output_size_set = True

    if config['training']['loss'] == 'binaryCrossentropy':
        if not activation_output_set:
            config['network']['activation_output'] = 'sigmoid'
        if not output_size_set:
            config['network']['output_size'] = 1
    elif config['training']['loss'] == 'categoricalCrossentropy':
        if not activation_output_set:
            config['network']['activation_output'] = 'softmax'
        if not output_size_set:
            config['network']['output_size'] = 2

    return config


def validate_config(config):

    net = config['network']
    train = config['training']

    if any(l <= 0 for l in net['layer']):
        raise ValueError("Invalid layer size")
    if net['input_size'] is not None and net['input_size'] <= 0:
        raise ValueError("Invalid input_size")
    if net['output_size'] <= 0:
        raise ValueError("Invalid output_size")
    if train['epochs'] <= 0:
        raise ValueError("Invalid epochs")
    if not (0 < train['learning_rate'] <= 1):
        raise ValueError("Invalid learning_rate")
    if train['batch_size'] <= 0:
        raise ValueError("Invalid batch_size")
    
    if train['loss'] == 'binaryCrossentropy' and net['output_size'] != 1:
        raise ValueError("For binaryCrossentropy loss, output_size must be 1")
    if train['loss'] == 'categoricalCrossentropy' and net['output_size'] != 2:
        raise ValueError("For categoricalCrossentropy loss, output_size must be 2")
    
    if train['loss'] == 'binaryCrossentropy' and net['activation_output'] != 'sigmoid':
        raise ValueError("For binaryCrossentropy loss, activation_output must be sigmoid")
    if train['loss'] == 'categoricalCrossentropy' and net['activation_output'] != 'softmax':
        raise ValueError("For categoricalCrossentropy loss, activation_output must be softmax")

    return config
