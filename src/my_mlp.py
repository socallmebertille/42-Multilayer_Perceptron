import numpy as np
from src.layers import DenseLayer

class MyMLP:
    def __init__(self, config):
        self.config = config
        self.network = None
        self.loss = config['training']['loss']
        self.learning_rate = config['training']['learning_rate']
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']

        self.build_network()

    def build_network(self):
        net_cfg = self.config['network']
        layers = []

        input_size = int(net_cfg['input_size'])
        hidden_layers = net_cfg['layers']
        output_size = int(net_cfg['output_size'])

        prev_size = input_size
        for size in hidden_layers:
            layers.append(DenseLayer(
                prev_size,
                size,
                activation=net_cfg['activation_hidden']
            ))
            prev_size = size

        layers.append(DenseLayer(
            prev_size,
            output_size,
            activation=net_cfg['activation_output']
        ))


    def train(self, X_train, y_train, X_val, y_val):
        for epoch in range(self.epochs):
            y_pred = self.forward(X_train)
            loss = compute_loss(y_train, y_pred, self.loss)
            self.backward(y_train, y_pred, self.learning_rate)

            val_loss = self.network.evaluate(X_val, y_val, self.loss)
            print(f"epoch {epoch}/{self.epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")

    def predict(self, X, Y):
        nb_samples = X.shape[0]
        y_pred = np.zeros(nb_samples)  # Dummy predictions
        for i in range(nb_samples):
            # Prediction logic here
            print(f"-> ({int(Y[i][0])}, {int(y_pred[i])}) - raw[ {0} {0} ]")
        
        correct_predictions = sum(1 for i in range(nb_samples) if y_pred[i] == Y[i])
        print(f"> correctly predicted : ({correct_predictions}/{nb_samples})")
        print(f"> loss (mean squarred error) : {0}")
        print(f"> loss (binary crossentropy) : {0}")

    def save(self, path):
        self.network.save(path)
        print(f"> saving model '{path}' to disk...")

    def load(self, path):
        self.network.load(path)
