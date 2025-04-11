from typing import List
import numpy as np

from src.loss_function import LossFunction
from src.tensor import Tensor

class RegularizedLoss:
    loss_function: LossFunction
    parameters: List[Tensor]
    regularization_type: str
    lambda_reg: float

    def __init__(self, loss_function: LossFunction, regularization_type: str = 'l2', lambda_reg: float = 0.001) -> None:
        if regularization_type != "l1" and regularization_type != "l2":
            raise RuntimeError(f"Regularization type '{regularization_type}' is not supported.\nSupported regularization type: 'l1', 'l2'")

        self.loss_function = loss_function
        self.parameters = None
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg

    @property
    def __name__(self) -> None:
        return f"{self.loss_function.__name__} with {self.regularization_type.upper()} Regularization"
    
    def set_parameters(self, parameters: List[Tensor]) -> None:
        """
        Loads model's parameter into this regularizer.
        """
        self.parameters = parameters

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes loss value using loss function and regularization method.
        """
        loss_function_val = self.loss_function.forward(y_true, y_pred)
        
        reg_term = 0
        for param in self.parameters:
            if param.requires_grad:
                if self.regularization_type == 'l1':
                    reg_term += np.sum(np.abs(param.data))
                elif self.regularization_type == 'l2':
                    reg_term += np.sum(np.square(param.data))
        
        regularized_loss = loss_function_val + self.lambda_reg * reg_term
        return regularized_loss

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Computes gradient of loss function, and apply regularization to parameters.
        """
        if self.parameters is None:
            raise RuntimeError(f"Parameters has not been set yet! Set the parameters to be updated using set_parameters().")
        
        base_gradient = self.loss_function.backward(y_true, y_pred)

        for param in self.parameters:
            if param.requires_grad:
                if self.regularization_type == 'l1':
                    param.gradient += self.lambda_reg * np.sign(param.data)
                elif self.regularization_type == 'l2':
                    param.gradient += 2 * self.lambda_reg * param.data

        return base_gradient
