from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Computes loss value using loss function.
        """
        pass

    @abstractmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray):
        """
        Computes gradient of loss function.
        """
        pass


class MeanSquaredError(LossFunction):
    @staticmethod
    def forward(y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def backward(y_true: np.ndarray, y_pred: np.ndarray):
        n = y_true.shape[0]
        return (2 / n) * (y_pred - y_true)