from typing import List
import numpy as np
import math

from src.tensor import Tensor
from src.layer import Layer, Dense
from src.loss_function import LossFunction, MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy
from src.activation_function import Softmax
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
                "Example: FFNN([\n"
                "   Dense(16, activation=\"relu\", kernel_initializer=\"he_normal\", input_size=4),\n"
                "   Dense(32, activation=\"sigmoid\", kernel_initializer=\"he_normal\")\n"
                "])\n"
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
            case opt if isinstance(opt, Optimizer):
                self.optimizer = opt 
            case "sgd":
                self.optimizer = StochasticGradientDescent()
            case _:
                raise ValueError(
                    f"Optimizer '{optimizer}' is not supported. "
                    "Supported parameters: 'sgd', blablablabla"
                )
        
        self.optimizer.set_parameters(self.get_parameters())
        
        match loss:
            case "mean_squared_error":
                self.loss_function = MeanSquaredError
            case "binary_crossentropy":
                self.loss_function = BinaryCrossEntropy
            case "categorical_crossentropy":
                self.loss_function = CategoricalCrossEntropy
            case _:
                raise ValueError(
                    f"Loss function '{optimizer}' is not supported. "
                    "Supported parameters: 'mean_squared_error', 'binary_crossentropy', 'categorical_crossentropy'"
                )

    def forward(self, input: Tensor) -> Tensor:
        """
        Initiates forwardpropagation through the network.
        """
        x = input
        for layer in self.layers:
            x = x.add_x0()
            x = layer.forward(x)
        
        self.output = x
        self.output.tensor_type = "output"
        return x.data
    
    def backward(self, y_true: np.ndarray) -> None:
        """
        Computes gradient through backpropagation.
        """
        if not self.output:
            raise RuntimeError("Forwardpropagation has not been done yet! Initiate forwardpropagation using forward().")
        elif y_true.shape[0] != self.output.data.shape[0]:
            raise ValueError(f"y_true and y_pred have different dimensions: {y_true.shape[0]} and {self.output.data.shape[0]}")

        if self.layers[-1].activation_function is Softmax:            
            def __backward():
                self.output._Tensor__children[0].gradient += self.output.data - y_true

            self.output._Tensor__backward = __backward

        loss = self.output.compute_loss(y_true, self.loss_function)
        loss.backward()
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10, batch_size: int = 32, validation_data: tuple = ()):
        """
        Trains the model with given training data.
        Input must be a 2D NumPy array, containing one or multiple data records for training.
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train and y_train must have the same number of entries!")
        
        if self.optimizer is None:
            raise RuntimeError(f"Model has not been compiled yet! Compile the model using compile().")
        
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
                    self.backward(sample[1]) # y
                
                self.optimizer.step()
            
            if validation_data:
                self.output = Tensor(np.array(self.predict(validation_data[0]))) # X
                loss = self.output.compute_loss(validation_data[1], self.loss_function) # y
                print(f"Epoch {_+1}, Validation Loss: {loss.data[0]}")
        
    def predict(self, X_test: np.ndarray):
        """
        Predicts the class of given test data.
        Input must be a 2D NumPy array, containing multiple data records to be predicted.
        """
        y_pred = []
        for sample in X_test:
            y_pred.append(self.forward(Tensor(sample)))
        
        return y_pred
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluates the model performance on the test set.
        Automatically detects if it's classification or regression based on loss function.
        """
        y_pred = self.predict(X_test)
        loss = self.loss_function.forward(y_test, y_pred)
        
        if self.loss_function is MeanSquaredError: 
            metric = np.mean((y_pred - y_test) ** 2)
            metric_name = "MSE"
        elif self.loss_function is CategoricalCrossEntropy:
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_test_labels = np.argmax(y_test, axis=1)
            metric = np.mean(y_pred_labels == y_test_labels)
            metric_name = "Accuracy"
        elif self.loss_function is BinaryCrossEntropy:
            y_pred_labels = (y_pred > 0.5).astype(int)
            metric = np.mean(y_pred_labels == y_test)
            metric_name = "Accuracy"
        else:
            raise ValueError("Unknown loss function type.")

        print(f"Loss: {loss:.4f}, {metric_name}: {metric:.4f}")
        return loss, metric