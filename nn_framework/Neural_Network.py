import numpy as np


class NeuralNetwork:
    def __init__(self, layers, cost_function, softmax=True, adam_optimizer=False):
        self.layers = layers
        self.cost_function = cost_function
        self.softmax = softmax
        self.adam_optimizer = adam_optimizer
        self.costs = []

        if self.adam_optimizer:
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.t = 0

            self.v_dW = [np.zeros_like(layer.W) for layer in self.layers]
            self.v_db = [np.zeros_like(layer.b) for layer in self.layers]
            self.s_dW = [np.zeros_like(layer.W) for layer in self.layers]
            self.s_db = [np.zeros_like(layer.b) for layer in self.layers]

    def softmax_function(self, x):
        exp_scores = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    def forward_propagation(self, X):
        activations = [X]
        z_values = []

        current_activation = X

        for layer in self.layers:
            Z, A = layer.forward(current_activation)
            z_values.append(Z)
            activations.append(A)
            current_activation = A

        if self.softmax:
            activations[-1] = self.softmax_function(activations[-1])

        return activations[-1], (activations, z_values)

    def _backward_propagation(self, cache, Y):
        activations, _ = cache
        num_examples = Y.shape[1]
        num_layers = len(self.layers)

        dW_list = [None] * num_layers
        db_list = [None] * num_layers

        output_activation = activations[-1]
        previous_activation = activations[-2]

        dW_list[-1], db_list[-1] = self.cost_function.cost_function_deriv(
            output_activation,
            Y,
            previous_activation,
        )

        dZ_next = output_activation - Y

        for layer_index in reversed(range(num_layers - 1)):
            current_layer = self.layers[layer_index]
            next_layer = self.layers[layer_index + 1]

            dZ = (
                np.dot(next_layer.W.T, dZ_next)
                * current_layer.activation_function_derivative(activations[layer_index + 1])
            )

            dW_list[layer_index] = (1 / num_examples) * np.dot(dZ, activations[layer_index].T)
            db_list[layer_index] = (1 / num_examples) * np.sum(dZ, axis=1, keepdims=True)

            dZ_next = dZ

        return dW_list, db_list

    def _update_params(self, dW_list, db_list, learning_rate):
        if self.adam_optimizer:
            self.t += 1

        for i, layer in enumerate(self.layers):
            dW = dW_list[i]
            db = db_list[i]

            if self.adam_optimizer:
                self.v_dW[i] = self.beta1 * self.v_dW[i] + (1 - self.beta1) * dW
                self.v_db[i] = self.beta1 * self.v_db[i] + (1 - self.beta1) * db

                self.s_dW[i] = self.beta2 * self.s_dW[i] + (1 - self.beta2) * (dW ** 2)
                self.s_db[i] = self.beta2 * self.s_db[i] + (1 - self.beta2) * (db ** 2)

                v_corrected_dW = self.v_dW[i] / (1 - self.beta1 ** self.t)
                v_corrected_db = self.v_db[i] / (1 - self.beta1 ** self.t)

                s_corrected_dW = self.s_dW[i] / (1 - self.beta2 ** self.t)
                s_corrected_db = self.s_db[i] / (1 - self.beta2 ** self.t)

                layer.W -= learning_rate * v_corrected_dW / (np.sqrt(s_corrected_dW) + self.epsilon)
                layer.b -= learning_rate * v_corrected_db / (np.sqrt(s_corrected_db) + self.epsilon)
            else:
                layer.W -= learning_rate * dW
                layer.b -= learning_rate * db

    def get_predictions(self, AL):
        return np.argmax(AL, axis=0)

    def gradient_descent(self, X, Y, X_val, Y_val, learning_rate, iterations, batch_size=-1):
        self.costs = []

        if batch_size == -1:
            batch_size = X.shape[1]

        num_examples = X.shape[1]

        for epoch in range(iterations):
            permutation = np.random.permutation(num_examples)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]

            for start_idx in range(0, num_examples, batch_size):
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[:, start_idx:end_idx]
                Y_batch = Y_shuffled[:, start_idx:end_idx]

                AL_batch, cache = self.forward_propagation(X_batch)
                dW_list, db_list = self._backward_propagation(cache, Y_batch)
                self._update_params(dW_list, db_list, learning_rate)

            AL_train, _ = self.forward_propagation(X)
            train_loss = self.cost_function.cost_function(AL_train, Y)

            AL_val, _ = self.forward_propagation(X_val)
            val_loss = self.cost_function.cost_function(AL_val, Y_val)

            print(
                f"Epoch {epoch + 1}/{iterations} - "
                f"train_loss: {train_loss:.6f} - "
                f"val_loss: {val_loss:.6f}"
            )

            self.costs.append(val_loss)

        return self.layers