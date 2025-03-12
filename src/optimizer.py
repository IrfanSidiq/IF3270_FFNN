from abc import ABC, abstractmethod
from typing import List
import numpy as np

from src.tensor import Tensor


class Optimizer(ABC):
    parameters: List[Tensor]
    learning_rate: float

    def __init__(self, parameters: List[Tensor], learning_rate: float = 0.01) -> None:
        self.parameters = parameters
        self.learning_rate = learning_rate

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
        for param in self.parameters:
            param.gradient.fill(0)


class StochasticGradientDescent(Optimizer):
    def step(self) -> None:
        for param in self.parameters:
            if param.requires_grad:
                param.data -= self.learning_rate * param.gradient