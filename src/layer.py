import numpy as np

from src.tensor import Tensor
from src.activation_function import ActivationFunction, Linear, ReLU
from src.weight_initializer import (
    WeightInitializer, ZeroInitializer, RandomUniformInitializer, 
    RandomNormalInitializer, GlorotUniformInitializer, HeNormalInitializer
)


class Dense:
    activation_function: ActivationFunction
    weight_initializer: WeightInitializer
    weight: Tensor
    output: Tensor

    def __init__(self, neuron_size: int, activation: str = "relu", kernel_initializer: str = "he_normal", input_size=None):
        match activation.lower():
            case "linear":
                self.activation_function = Linear
            case "relu":
                self.activation_function = ReLU
            case _:
                raise ValueError(
                    f"Activation function '{activation}' is not supported. "
                    "Supported parameters: 'linear', 'relu'"
                )

        match kernel_initializer.lower():
            case "zeros":
                self.weight_initializer = ZeroInitializer
            case "random_uniform":
                self.weight_initializer = RandomUniformInitializer
            case "random_normal":
                self.weight_initializer = RandomNormalInitializer
            case "glorot_uniform":
                self.weight_initializer = GlorotUniformInitializer
            case "he_normal":
                self.weight_initializer = HeNormalInitializer
            case _:
                raise ValueError(
                    f"Kernel initializer '{kernel_initializer}' is not supported. "
                    "Supported parameters: 'zeros', 'random_uniform', 'random_normal', "
                    "'glorot_uniform', 'he_normal'"
                )

        if input_size:
            self.weight = self.initialize_weights(input_size)
        else:
            self.weight = None

        self.output = np.zeros(neuron_size)
    
    def initialize_weights(self, input_size: int):
        weights = self.weight_initializer.initialize_weight(self.get_neuron_size(), input_size + 1)
        self.weight = []

        for weight in weights:
            self.weight.append(Tensor(weight))
    
    def get_neuron_size(self):
        return self.output.data.shape[0]

    # def forward(input: Tensor):

    #     for weight in 