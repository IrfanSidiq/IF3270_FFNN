from typing import List
import numpy as np
import math

from src.tensor import Tensor
from src.layer import Layer
from src.loss_function import LossFunction, MeanSquaredError
from src.optimizer import Optimizer, StochasticGradientDescent


class FFNN:
    optimizer: Optimizer
    loss_function: LossFunction
    layers: List[Layer]
    output: Tensor

    def __init__(self, layers: List[Layer]) -> None:
        if not layers:
            raise ValueError("Layers cannot be empty!")

        if not layers[0].weights:
            raise ValueError(
                "Input size of the first layer must be specified.\n"
                "Example: FFNN(["
                "   Dense(16, activation=\"relu\", kernel_initializer=\"he_normal\", input_size=4),\n"
                "   Dense(32, activation=\"sigmoid\", kernel_initializer=\"he_normal\")\n"
                "])"
            )

        self.layers = layers
        for i in range(1, len(self.layers)):
            self.layers[i].initialize_weights(self.layers[i - 1].get_neuron_size())
        
        self.loss_function = None
        self.output = None

    def get_parameters(self) -> List[Tensor]:
        """
        Returns all parameters in the neural network.
        """
        parameters = []
        for layer in self.layers:
            parameters.extend(layer.get_parameters())

        return parameters

    def compile(self, optimizer: str = "sgd", loss: str = "mean_squared_error") -> None:
        """
        Initializes optimizer and loss function of the network.
        """
        match optimizer:
            case "sgd":
                self.optimizer = StochasticGradientDescent(self.get_parameters())
            case _:
                raise ValueError(
                    f"Optimizer '{optimizer}' is not supported. "
                    "Supported parameters: 'sgd', blablablabla"
                )
        
        match loss:
            case "mean_squared_error":
                self.loss_function = MeanSquaredError
            case _:
                raise ValueError(
                    f"Loss function '{optimizer}' is not supported. "
                    "Supported parameters: 'mean_squared_error', blablablabla"
                )

    def forward(self, input: Tensor) -> Tensor:
        """
        Initiates forwardpropagation through the network.
        """
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        
        self.output = x
        return x
    
    def backward(self, y_true: np.ndarray) -> None:
        """
        Computes gradient through backpropagation.
        """
        if not self.output:
            raise ValueError("Forwardpropagation has not been done yet!")
        elif y_true.shape[0] != self.output.data.shape[0]:
            raise ValueError(f"y_true and y_pred have different dimensions: {y_true.shape[0]} and {self.output.data.shape[0]}")

        loss = self.output.compute_loss(y_true, self.loss_function)
        loss.backward()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10, batch_size: int = 32, validation_data: tuple = ()):
        """
        Trains the model with given training data. Input must be a 2D NumPy array, containing multiple data records for training.
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train and y_train must have the same number of entries!")
        
        batches = []
        for i in range(math.ceil(X_train.shape[0] / batch_size)):
            batch = []
            for j in range(i * batch_size, min(i * batch_size + batch_size, X_train.shape[0])):
                batch.append((X_train[j], y_train[j]))
            
            batches.append(batch)

        for _ in range(epochs):
            for batch in batches:
                self.optimizer.zero_grad()
                for sample in batch:
                    self.forward(Tensor(sample[0])) # X
                    self.backward(Tensor(sample[1])) # y
                
                self.optimizer.step()
        
    def predict(self, X_test: np.ndarray):
        """
        Predicts the class of given test data. Input must be a 2D NumPy array, containing multiple data records to be predicted.
        """
        y_pred = []
        for sample in X_test:
            y_pred.append(self.forward(Tensor(sample)))
        
        return y_pred
    
    