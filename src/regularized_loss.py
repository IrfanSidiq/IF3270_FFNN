from typing import List
import numpy as np

from src.loss_function import LossFunction
from src.tensor import Tensor

class RegularizedLoss:
    def __init__(self, loss_function: LossFunction, parameters: List[Tensor], regularization_type: str ='l2', lambda_reg: float = 0.01) -> None:
        self.loss_function = loss_function
        self.parameters = parameters
        self.regularization_type = regularization_type
        self.lambda_reg = lambda_reg
        
    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        loss_function_val = self.loss_function(y_pred, y_true)
        
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
        base_gradient = self.loss_function.backward(y_true, y_pred)

        for param in self.parameters:
            if param.requires_grad:
                if self.regularization_type == 'l1':
                    param.gradient += self.lambda_reg * np.sign(param.data)
                elif self.regularization_type == 'l2':
                    param.gradient += 2 * self.lambda_reg * param.data

        return base_gradient
