import numpy as np
from Models.NeuralNetUtilities import NeuralNetUtilities


class NeuralNet:

    WEIGHT_MULTIPLIER = 0.01

    # Actions:
    # Attack
    # Cast
    # Move Right
    # Move Left
    # Idle
    # Jump
    # Heal
    # Dash

    def __init__(self, output_size: int = 7):
        self.input_size = 0
        self.hidden_layers = []
        self.output_size = output_size  # attack, cast, move, jump, heal, dash
        self.weights = []
        self.biases = []
        self.reward = 0

    def copy(self):

        new_net = NeuralNet(self.output_size)

        new_net.input_size = self.input_size
        new_net.hidden_layers = self.hidden_layers.copy()

        new_net.weights = [w.copy() for w in self.weights]
        new_net.biases = [b.copy() for b in self.biases]

        return new_net

    def initialize(self, input_size: int, hidden_layers = [128, 64]):
        self.input_size = input_size
        self.hidden_layers = hidden_layers

        # connecting Input layer -> Hidden 1 layer:
        # (input_size: rows, hidden_1_size: columns)
        self.weights.append(np.random.randn(self.input_size, hidden_layers[0]) * NeuralNet.WEIGHT_MULTIPLIER)
        self.biases.append(np.zeros((1, self.hidden_layers[0])))

        # connecting Hidden n layer -> Hidden n + 1 layer
        for i in range(len(hidden_layers) - 1):
            self.weights.append(np.random.randn(hidden_layers[i], hidden_layers[i + 1]) * NeuralNet.WEIGHT_MULTIPLIER)
            self.biases.append(np.zeros((1, hidden_layers[i + 1])))

        # connecting Hidden last layer -> Output layer:
        # (hidden[n - 1]: rows, 6: columns)
        self.weights.append(np.random.randn(self.hidden_layers[-1], self.output_size) * NeuralNet.WEIGHT_MULTIPLIER)
        self.biases.append(np.zeros((1, self.output_size)))

    def forward(self, data):

        current_data = np.array(data)

        if current_data.ndim == 1:
            current_data = current_data.reshape(1, -1)
        elif current_data.ndim > 2:
            current_data = current_data.reshape(current_data.shape[0], -1)

        layer = [current_data]

        for i in range(len(self.weights)):

            z = np.dot(layer[-1], self.weights[i]) + self.biases[i]

            if i < len(self.weights) - 1:
                activation = NeuralNetUtilities.relu(z)
            else:
                activation = NeuralNetUtilities.sigmoid(z)

            layer.append(activation)

        return layer[-1]
