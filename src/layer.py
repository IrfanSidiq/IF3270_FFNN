from typing import List
from abc import ABC, abstractmethod
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.tensor import Tensor
from src.activation_function import ActivationFunction, Linear, ReLU, Sigmoid, HyperbolicTangent, Softmax, GELU
from src.weight_initializer import (
    WeightInitializer, ZeroInitializer, RandomUniformInitializer, 
    RandomNormalInitializer, GlorotUniformInitializer, HeNormalInitializer
)


class Layer(ABC):
    activation_function: ActivationFunction
    weight_initializer: WeightInitializer
    weights: List[Tensor]
    gradients: List[Tensor]
    output: Tensor

    @abstractmethod
    def initialize_weights(self, input_size: int) -> None:
        """
        Initializes weights of each neuron in the layer according to input vector size.
        """
        pass

    def get_parameters(self):
        """
        Returns weights of the layer.
        """
        return self.weights    

    def get_neuron_size(self) -> int:
        """
        Returns number of neurons in the layer.
        """
        return self.output.data.shape[0]
    
    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        """
        Calculates the output of the layer given an input vector.
        """
        pass

    def plot_dist(self, is_weight: bool, n_layer: int):
        """
        Plot distribution of weights or gradients in this layer.
        """
        data = []
        if is_weight:
            for i in range (len(self.weights)):
                data.extend(self.weights[i].data.flatten())
        else:
            for i in range (len(self.gradients)):
                data.extend(self.gradients[i].gradient.flatten())
        
        data = np.array(data)
        text = 'Weights' if is_weight else 'Gradients'

        plt.figure()
        sns.histplot(data, kde=True, color='dodgerblue', edgecolor='black')
        plt.xlabel(text)
        plt.ylabel('Frequency')
        plt.title(f'{text} Distribution Plot Of Layer {n_layer}')
        plt.tight_layout()
        plt.show()

class Dense(Layer):
    def __init__(self, neuron_size: int, activation: str = "relu", kernel_initializer: str = "he_normal", input_size: int=None) -> None:
        match activation.lower():
            case "linear":
                self.activation_function = Linear
            case "relu":
                self.activation_function = ReLU
            case "sigmoid":
                self.activation_function = Sigmoid
            case "tanh":
                self.activation_function = HyperbolicTangent
            case "softmax":
                self.activation_function = Softmax
            case "gelu":
                self.activation_function = GELU
            case _:
                raise ValueError(
                    f"Activation function '{activation}' is not supported. "
                    "Supported parameters: 'linear', 'relu'"
                )

        match kernel_initializer.lower():
            case weight_init if isinstance(weight_init, WeightInitializer):
                self.weight_initializer = weight_init
            case "zeros":
                self.weight_initializer = ZeroInitializer()
            case "random_uniform":
                self.weight_initializer = RandomUniformInitializer()
            case "random_normal":
                self.weight_initializer = RandomNormalInitializer()
            case "glorot_uniform":
                self.weight_initializer = GlorotUniformInitializer()
            case "he_normal":
                self.weight_initializer = HeNormalInitializer()
            case _:
                raise ValueError(
                    f"Kernel initializer '{kernel_initializer}' is not supported. "
                    "Supported parameters: 'zeros', 'random_uniform', 'random_normal', "
                    "'glorot_uniform', 'he_normal'"
                )

        self.output = np.zeros(neuron_size)
        self.gradients = [Tensor(np.array([0])) for _ in range(neuron_size)]

        if input_size:
            self.initialize_weights(input_size)
        else:
            self.weights = None

    def initialize_weights(self, input_size: int) -> None:
        weights = self.weight_initializer.initialize_weight(self.get_neuron_size(), input_size + 1)
        self.weights = []
        for weight in weights:
            self.weights.append(Tensor(weight, tensor_type="weight"))

    def forward(self, input: Tensor) -> Tensor:
        if self.weights is None:
            raise ValueError("Weights have not been initialized! Initialize them using initialize_weights() first.")

        nets = []
        for weight in self.weights:
            w_x = weight * input
            net = w_x.sum()
            nets.append(net)

        self.gradients = nets

        res = None
        if self.activation_function is Softmax:
            output = nets[0].concat(nets[1:])
            res = output.compute_activation(self.activation_function)
        else:
            output = []
            for net in nets:
                output.append(net.compute_activation(self.activation_function))
            res = output[0].concat(output[1:])

        return res