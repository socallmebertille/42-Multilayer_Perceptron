import numpy as np
from src.activations import sigmoid, sigmoid_derivative, softmax, softmax_derivative, relu, relu_derivative
from src.losses import binary_crossentropy, binary_crossentropy_derivative, categorical_crossentropy, categorical_crossentropy_derivative

class MyMLP:

    def __init__(self, config):
        self.config = config
        self.weights = []
        self.biases = []

        # Construire l'architecture du réseau
        self._build_network()
    
    def _build_network(self):
        """Construit les couches avec initialisation des poids"""
        # Architecture complète : [input] -> [hidden layers] -> [output]
        layer_sizes = (
            [self.config['network']['input_size']] +      # [30]
            self.config['network']['layer'] +              # [24, 24, 24]
            [self.config['network']['output_size']]        # [1]
        )
        # Résultat : [30, 24, 24, 24, 1]
        
        print(f"Network architecture: {layer_sizes}")
        
        # Initialiser poids et biais pour chaque connexion entre couches
        for i in range(len(layer_sizes) - 1):
            # Créer les poids entre couche i et couche i+1
            W = self._initialize_weights(
                layer_sizes[i],      # input de cette couche
                layer_sizes[i + 1],  # output de cette couche
                self.config['network']['weights_init']
            )
            b = np.zeros(layer_sizes[i + 1])  # Un biais par neurone de sortie
            
            self.weights.append(W)
            self.biases.append(b)
            
            print(f"  Layer {i}: W shape = {W.shape}, b shape = {b.shape}")

    def _initialize_weights(self, n_in, n_out, method):
        """Initialise une matrice de poids"""
        if method == 'heUniform':
            limit = np.sqrt(6 / n_in)
            return np.random.uniform(-limit, limit, (n_in, n_out))
        elif method == 'heNormal':
            return np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
        elif method == 'xavierUniform':
            limit = np.sqrt(6 / (n_in + n_out))
            return np.random.uniform(-limit, limit, (n_in, n_out))
        elif method == 'xavierNormal':
            return np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))
        else:
            raise ValueError(f"Unknown initialization: {method}")

    def forward(self, X):
        """
        Fait passer les données dans le réseau (de gauche à droite)
        
        X: (batch_size, 30) → données d'entrée
        Returns: (batch_size, 1) → prédictions
        """
        # On va stocker toutes les activations pour le backward
        self.activations = [X]  # a^(0) = X
        self.z_values = []      # Pour stocker les z de chaque couche
        
        a = X.astype(np.float64)  # Activation de la couche actuelle (au début = input)
        
        # Pour chaque couche (sauf la dernière)
        for i in range(len(self.weights) - 1):
            # 1. Calcul de z = a @ W + b
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            # 2. Activation ReLU pour couches cachées
            a = relu(z)
            self.activations.append(a)
        
        # Dernière couche (output) avec sigmoid ou softmax
        z = a @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        
        if self.config['training']['loss'] == 'binaryCrossentropy'and self.config['network']['output_size'] == 1:
            a = sigmoid(z)  # Probabilité entre 0 et 1
        else:
            a = softmax(z)  # Probabilités pour multi-classes
        self.activations.append(a)
        
        return a  # Prédiction finale
    
    def backward(self, y_true):
        """
        Calcule comment ajuster les poids (de droite à gauche)
        
        Utilise la chain rule (règle de dérivation en chaîne)
        """
        m = y_true.shape[0]  # Nombre d'exemples dans le batch
        
        # Stockage des gradients
        grads_w = []
        grads_b = []
        
        # 1. Erreur de la couche de sortie
        # Pour sigmoid + binary crossentropy, la formule se simplifie !
        delta = self.activations[-1] - y_true  # ŷ - y (SUPER SIMPLE !)
        
        # 2. Remonter couche par couche (de la fin vers le début)
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient pour les poids : dW = a_precedente^T @ delta
            dW = self.activations[i].T @ delta / m
            
            # Gradient pour les biais : db = moyenne de delta
            db = np.mean(delta, axis=0)
            
            grads_w.insert(0, dW)  # Insérer au début (car on remonte)
            grads_b.insert(0, db)
            
            # Si pas la première couche, propager l'erreur
            if i > 0:
                # Propager l'erreur à la couche précédente
                delta = (delta @ self.weights[i].T) * relu_derivative(self.z_values[i-1])
        
        return grads_w, grads_b
    
    def update_weights(self, grads_w, grads_b):
        """
        Met à jour les poids avec la règle simple :
        nouveau_poids = ancien_poids - learning_rate * gradient
        """
        lr = self.config['training']['learning_rate']
        
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]

    def train(self, X_train, y_train, X_valid, y_valid):
        epochs = self.config['training']['epochs']
        batch_size = self.config['training']['batch_size']

        # Pour l'early stopping
        best_val_loss = float('inf')
        patience = 10  # Nombre d'epochs sans amélioration avant d'arrêter
        patience_counter = 0
        best_weights = None
        best_biases = None
    
        for epoch in range(epochs):

            # 1. Mélanger les données
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # 2. Découper en mini-batches
            num_batches = len(X_train) // batch_size
            
            for i in range(num_batches):
                # Extraire un batch
                start = i * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                # 3. Forward pass
                y_pred = self.forward(X_batch)
                
                # 4. Backward pass
                grads_w, grads_b = self.backward(y_batch)
                
                # 5. Update des poids
                self.update_weights(grads_w, grads_b)
            
            # 6. Calculer les loss pour affichage
            train_pred = self.forward(X_train)          
            valid_pred = self.forward(X_valid)

            # Choisir la bonne loss
            loss_fn = (categorical_crossentropy 
                    if self.config['training']['loss'] == 'categoricalCrossentropy'
                    else binary_crossentropy)
            
            # Dans la boucle
            train_loss = loss_fn(y_train, train_pred)
            valid_loss = loss_fn(y_valid, valid_pred)
            
            # 7. Afficher
            print(f"epoch {epoch+1:02d}/{epochs} - loss: {train_loss:.4f} - val_loss: {valid_loss:.4f}")

            # Early stopping
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                patience_counter = 0
                # Sauvegarder les meilleurs poids
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                print(f"  → New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n✓ Early stopping at epoch {epoch+1}")
                    print(f"  Best validation loss was {best_val_loss:.4f} at epoch {epoch+1-patience}")
                    # Restaurer les meilleurs poids
                    self.weights = best_weights
                    self.biases = best_biases
                    break


    def predict(self, X, Y):
        nb_samples = X.shape[0]
        # y_pred = self.network.forward(X)
        # y_pred = np.argmax(y_pred, axis=1)
        y_pred = 0
        for i in range(nb_samples):
            # Prediction logic here
            print(f"-> ({int(Y[i][0])}, {int(y_pred[i])}) - raw[ {0} {0} ]")
        
        correct_predictions = sum(1 for i in range(nb_samples) if y_pred[i] == Y[i])
        print(f"> correctly predicted : ({correct_predictions}/{nb_samples})")
        print(f"> loss (mean squarred error) : {0}")
        print(f"> loss (binary crossentropy) : {0}")

    def save(self, path):
        try:
            model = {
                'weights': self.weights,
                'biases': self.biases
            }
            np.save(path, model)
            print(f"> saving model '{path}' to disk...")
        except Exception as e:
            print(f"Error while saving model to {path} : {e}")
            return

    def load(self, path):
        model = np.load(path, 'r')
        self.weights = model['weights']
        self.biases = model['biases']
        print(f"> loading model '{path}' from disk...")
        print(f"model : {model}")
