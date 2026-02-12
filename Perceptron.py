import numpy as np
from numpy.typing import NDArray


class BatchPerceptron:
    
    def __init__(self, learning_rate: float, epochs: int):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.error_history = []
        self.weight_history = []

    def _initialize(self, n_features: int):
        # Inicializa pesos en rango pequeño
        self.weights = np.random.uniform(-5, 5, n_features + 1)

    def _prepare_inputs(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        # Agrega columna de bias
        ones = np.ones((X.shape[0], 1))
        return np.concatenate((ones, X), axis=1)

    def _step_function(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Función escalón
        return (z >= 0).astype(int)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]):
        X_bias = self._prepare_inputs(X)
        self._initialize(X.shape[1])

        for _ in range(self.epochs):

            self.weight_history.append(self.weights.copy())

            # Forward
            linear_output = X_bias @ self.weights
            predictions = self._step_function(linear_output)

            # Error vector
            error_vector = predictions - y

            # Batch update
            gradient = X_bias.T @ error_vector
            self.weights -= self.learning_rate * gradient

            # Global error (Norma L2)
            global_error = np.linalg.norm(error_vector)
            self.error_history.append(global_error)

    def get_error_history(self):
        return np.array(self.error_history)

    def get_weight_history(self):
        return np.array(self.weight_history)
