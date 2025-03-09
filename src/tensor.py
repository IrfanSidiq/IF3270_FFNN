from typing import Callable
import numpy as np

from src.activation_function import ActivationFunction
from src.loss_function import LossFunction


class Tensor:
    data: np.ndarray
    gradient: np.ndarray
    __children: set
    __op: str
    __backward: Callable[[], None]
    type: str

    def __init__(self, data: np.ndarray, __children: tuple = (), __op: str = "", tensor_type: str = ""):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected np.ndarray, but got {type(data).__name__} instead")
        
        self.data = data
        self.gradient = np.zeros_like(self.data, dtype=float)
        self.__children = set(__children)
        self.__op = __op
        self.__backward = lambda: None
        self.tensor_type = tensor_type

    def __repr__(self):
        return f"Value: {self.data}, Gradient: {self.gradient}"

    def __add__(self, other):
        if not isinstance(other, Tensor):
            if not isinstance(other, np.ndarray):
                raise TypeError(f"Expected Tensor or np.ndarray, but got {type(other).__name__} instead")
                
            other = Tensor(other)

        if self.data.shape != other.data.shape:
            raise ValueError(f"Values of different shapes cannot be operated: {self.data.shape} and {other.data.shape}")

        res = Tensor(self.data + other.data, (self, other), "+")

        def __backward():
            self.gradient += res.gradient
            other.gradient += res.gradient

        res.__backward = __backward
        return res
    
    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            other = Tensor(other)
        elif isinstance(other, (float, int)):
            other = Tensor(np.array(other))
        
        if self.data.shape != other.data.shape:
            raise ValueError(f"Values of different shapes cannot be operated: {self.data.shape} and {other.data.shape}")

        res = Tensor(self.data * other.data, (self, other), "*")

        def __backward():
            self.gradient += other.data * res.gradient
            other.gradient += self.data * res.gradient

        res.__backward = __backward
        return res
    
    def __pow__(self, other):
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError(f"Expected int or float, but got {type(other).__name__} instead")

        res = Tensor(self.data ** other, (self, ), f'**{other}')

        def __backward():
            self.grad += (other * self.data**(other - 1)) * res.gradient
        
        res.__backward = __backward
        return res

    def __neg__(self):
        return Tensor(-self.data, self.__children, self.__op)

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        result = self + other
        self.data = result.data
        return self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
    
    def __isub__(self, other):
        result = self - other
        self.data = result.data
        return self

    def __rmul__(self, other):
        return self * other
    
    def __imul__(self, other):
        result = self * other
        self.data = result.data
        return self

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            other.data = other.data.astype(float)
        
        if isinstance(other, np.ndarray):
            other = other.astype(float)

        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    
    def __array__(self, dtype=None):
        return self.data if dtype is None else self.data.astype(dtype)

    def compute_activation(self, activation_function: ActivationFunction):
        res = activation_function.forward(self.data)
        res = Tensor(res, (self,), type(activation_function).__name__)

        def __backward():
            self.gradient += activation_function.backward(self.data) * res.gradient
        
        res.__backward = __backward
        return res

    def compute_loss(self, y_pred: "Tensor", loss_function: LossFunction):
        res = loss_function.forward(self.data, y_pred.data)
        res = Tensor(np.array(res), (self,), type(loss_function).__name__)

        def __backward():
            self.gradient += loss_function.backward(self.data, y_pred.data) * res.gradient
        
        res.__backward = __backward
        return res
    
    def backward(self):
        topo = []
        visited = set()
        
        def topological_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v.__children:
                    topological_sort(child)
                topo.append(v)

        topological_sort(self)

        self.gradient = np.ones_like(self.data, dtype=float)
        for v in reversed(topo):
            v.__backward()