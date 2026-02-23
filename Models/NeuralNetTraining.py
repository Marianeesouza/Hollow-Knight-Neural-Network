import numpy as np

from Models.NeuralNetUtilities import NeuralNetUtilities

class NeuralNetTraining:

    e_start = 1
    e_min = 0.01
    epsilon = 1
    decay_linear = 0.001
    learning_rate = 0.00025

    @classmethod
    def optimize(cls, weights: list, biases: list, state: np.ndarray, target_q_values: np.ndarray):

        if state.ndim == 1:
            state = state.reshape(1, -1)

        if target_q_values.ndim == 1:
            target_q_values = target_q_values.reshape(1, -1)

        activations = [state]
        z_values = []

        for i in range(len(weights)):
            z = np.dot(activations[-1], weights[i]) + biases[i]
            z_values.append(z)

            if i < len(weights) - 1:
                activation = NeuralNetUtilities.relu(z)
            else:
                activation = NeuralNetUtilities.sigmoid(z)

            activations.append(activation)

        current_q_values = activations[-1]

        nabla_w = [np.zeros(w.shape) for w in weights]
        nabla_b = [np.zeros(b.shape) for b in biases]

        error = current_q_values - target_q_values
        delta = error * NeuralNetUtilities.sigmoid_derivative(z_values[-1])

        nabla_w[-1] = np.dot(activations[-2].T, delta)
        nabla_b[-1] = np.sum(delta, axis = 0, keepdims = True)

        for l in range(2, len(weights) + 1):
            z = z_values[-l]
            derivative = NeuralNetUtilities.relu_derivative(z)

            delta = np.dot(delta, weights[-l+1].T) * derivative

            nabla_w[-l] = np.dot(activations[-l-1].T, delta)
            nabla_b[-l] = np.sum(delta, axis = 0, keepdims = True)

        batch_size = state.shape[0]

        new_weights = [w - ((cls.learning_rate / batch_size) * nw) for w, nw in zip(weights, nabla_w)]
        new_biases = [b - ((cls.learning_rate / batch_size) * nb) for b, nb in zip(biases, nabla_b)]

        return new_weights, new_biases

    # Learning equations
    @staticmethod
    def bellman(reward: float, discount_factor: float, next_q_values: list):
        return reward + discount_factor * np.max(next_q_values)

    @classmethod
    def gradient_descent_weight(cls, weights: np.ndarray, inputs: np.ndarray, n: int,
                         target_q_values: np.ndarray, current_q_values: np.ndarray, z_output: np.ndarray) -> np.ndarray:

        error = current_q_values - target_q_values
        activation_derivative = NeuralNetUtilities.sigmoid_derivative(z_output)

        delta = error * activation_derivative

        if inputs.ndim == 1: inputs = inputs.reshape(1, -1)
        if delta.ndim == 1: delta = delta.reshape(1, -1)

        gradient = (2 / n) * np.dot(inputs.T, delta)

        return weights - (cls.learning_rate * gradient)

    @classmethod
    def gradient_descent_bias(cls, biases: np.ndarray, n: int,
                         target_q_values: np.ndarray, current_q_values: np.ndarray, z_output: np.ndarray) -> np.ndarray:

        error = current_q_values - target_q_values
        activation_derivative = NeuralNetUtilities.sigmoid_derivative(z_output)

        delta = error * activation_derivative
        gradient = (2 / n) * np.sum(delta, axis=0)

        return biases - (cls.learning_rate * gradient)

    @classmethod
    def update_epsilon(cls):
        cls.epsilon = max(cls.e_min, (cls.epsilon - cls.decay_linear))
        return cls.epsilon