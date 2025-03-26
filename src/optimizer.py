from abc import ABC, abstractmethod
from typing import List
import numpy as np

from src.tensor import Tensor


class Optimizer(ABC):
    parameters: List[Tensor]
    learning_rate: float

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.parameters = None

    def set_parameters(self, parameters: List[Tensor]):
        self.parameters = parameters

    @abstractmethod
    def step(self) -> None:
        """
        Updates parameters based on learning rate.
        """
        pass

    def zero_grad(self) -> None:
        """
        Sets all parameters' gradient to zero.
        """
        if self.parameters is None:
            raise RuntimeError(f"Parameters has not been set yet! Set the parameters to optimize using set_parameters().")

        for param in self.parameters:
            param.gradient.fill(0)


class StochasticGradientDescent(Optimizer):
    def step(self) -> None:
        if self.parameters is None:
            raise RuntimeError(f"Parameters has not been set yet! Set the parameters to optimize using set_parameters().")
        
        for param in self.parameters:
            if param.requires_grad:
                param.data -= self.learning_rate * param.gradient