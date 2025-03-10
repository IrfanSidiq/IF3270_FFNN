from abc import ABC, abstractmethod
import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def forward(x: np.ndarray):
        pass

    @abstractmethod
    def backward(x: np.ndarray):
        pass


class Linear(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray):
        return x
    
    @staticmethod
    def backward(x: np.ndarray):
        return np.ones_like(x)


class ReLU(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray):
        return np.maximum(0, x)
    
    @staticmethod
    def backward(x: np.ndarray):
        return (x > 0).astype(float)


class Sigmoid(ActivationFunction):
    @staticmethod
    def forward(x: np.ndarray):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def backward(x: np.ndarray):
        sigmoid_x = Sigmoid.forward(x)
        return sigmoid_x * (1 - sigmoid_x)