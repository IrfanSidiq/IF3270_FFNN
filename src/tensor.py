from typing import Callable, List
import numpy as np

from src.activation_function import ActivationFunction
from src.loss_function import LossFunction


class Tensor:
    data: np.ndarray
    gradient: np.ndarray
    __children: List["Tensor"]
    __op: str
    __backward: Callable[[], None]
    requires_grad: bool
    tensor_type: str

    def __init__(self, data: np.ndarray, __children: List["Tensor"] = [], __op: str = "", tensor_type: str = "") -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected np.ndarray, but got {type(data).__name__} instead")
        
        self.data = data.astype(float)
        self.gradient = np.zeros_like(self.data, dtype=float)
        self.__children = __children
        self.__op = __op
        self.__backward = lambda: None
        self.requires_grad = True
        self.tensor_type = tensor_type

    def __repr__(self) -> str:
        return (
            f"Value: {self.data}, Gradient: {self.gradient}, "
            f"Op: {self.__op if self.__op else 'None'}"
            f"{(', Type: ' + self.tensor_type) if self.tensor_type else ''}"
        )

    def add_x0(self) -> "Tensor":
        """
        Adds x0 = 1 to the tensor. Used for preparing output of layer-i as an input to the layer-i+1.
        """
        self.data = np.concatenate([np.array([1]), self.data])
        self.gradient = np.concatenate([np.array([0]), self.gradient])
        return self

    def __add__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        if not isinstance(other, Tensor):
            if not isinstance(other, np.ndarray):
                raise TypeError(f"Expected Tensor or np.ndarray, but got {type(other).__name__} instead")
                
            other = Tensor(other)

        if self.data.shape != other.data.shape:
            raise ValueError(f"Values of different shapes cannot be operated: {self.data.shape} and {other.data.shape}")

        res = Tensor(self.data + other.data, [self, other], "+")

        def __backward():
            self.gradient += res.gradient
            other.gradient += res.gradient

        res.__backward = __backward
        return res
    
    def __mul__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        if isinstance(other, np.ndarray):
            other = Tensor(other)
        elif isinstance(other, (float, int)):
            other = Tensor(np.array(other))
        
        if self.data.shape != other.data.shape:
            raise ValueError(f"Values of different shapes cannot be operated: {self.data.shape} and {other.data.shape}")

        res = Tensor(self.data * other.data, [self, other], "*")

        def __backward():
            self.gradient += other.data * res.gradient
            other.gradient += self.data * res.gradient

        res.__backward = __backward
        return res
    
    def __pow__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        if not isinstance(other, int) and not isinstance(other, float):
            raise TypeError(f"Expected int or float, but got {type(other).__name__} instead")

        res = Tensor(self.data ** other, [self], f'**{other}')

        def __backward():
            self.gradient += (other * self.data**(other - 1)) * res.gradient
        
        res.__backward = __backward
        return res

    def __neg__(self) -> "Tensor":
        return Tensor(-self.data, self.__children, self.__op)

    def __radd__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        return self + other

    def __iadd__(self, other: "Tensor") -> "Tensor":
        result = self + other
        self.data = result.data
        return self

    def __sub__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        return self + (-other)

    def __rsub__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        return other + (-self)
    
    def __isub__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        result = self - other
        self.data = result.data
        return self

    def __rmul__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        return self * other
    
    def __imul__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        result = self * other
        self.data = result.data
        return self

    def __truediv__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        if isinstance(other, Tensor):
            other.data = other.data.astype(float)
        
        if isinstance(other, np.ndarray):
            other = other.astype(float)

        return self * other**-1

    def __rtruediv__(self, other: "Tensor" | np.ndarray) -> "Tensor":
        return other * self**-1
    
    def __array__(self, dtype=None) -> np.ndarray:
        return self.data if dtype is None else self.data.astype(dtype)

    def sum(self) -> "Tensor":
        """
        Computes the sum of 1D tensor elements.
        """
        res = Tensor(np.sum(self.data).reshape(1), [self], "sum")

        def __backward():
            self.gradient += res.gradient[0]
        
        res.__backward = __backward
        return res

    def compute_activation(self, activation_function: ActivationFunction) -> "Tensor":
        """
        Computes the activation output using given activation function.
        """
        res = activation_function.forward(self.data)
        res = Tensor(res, [self], activation_function.__name__)

        def __backward():
            self.gradient += activation_function.backward(self.data) * res.gradient
        
        res.__backward = __backward
        return res

    def compute_loss(self, y_true: np.ndarray, loss_function: LossFunction) -> "Tensor":
        """
        Computes the loss value using given loss function and y_true.
        """
        res = loss_function.forward(y_true, self.data)
        res = Tensor(np.array([res]), [self], loss_function.__name__)

        def __backward():
            self.gradient += loss_function.backward(y_true, self.data) * res.gradient[0]
        
        res.__backward = __backward
        return res

    def concat(self, tensors: List["Tensor"]) -> "Tensor":
        """
        Concats multiple 1D tensors into one long 1D tensor. Used for combining outputs of multiple neurons into one output.
        """
        combined_data = self.data
        for tensor in tensors:
            combined_data = np.append(combined_data, tensor.data)
        
        tensors.insert(0, self)
        res = Tensor(combined_data, tensors, "concat")

        def __backward():
            idx = 0
            for tensor in res.__children:
                tensor.gradient += res.gradient[idx]
                idx = (idx + 1) % len(res.gradient)

        res.__backward = __backward
        return res
    
    def backward(self) -> None:
        """
        Starts automated differentiation, calculating gradients from the root of operation tree all the way to the leaves.
        """
        topo = []
        visited = set()
        
        def topological_sort(v: Tensor):
            if v not in visited:
                visited.add(v)
                for child in v.__children:
                    topological_sort(child)
                topo.append(v)

        topological_sort(self)

        self.gradient = np.ones_like(self.data, dtype=float)
        for v in reversed(topo):
            if v.require_grad:
                v.__backward()