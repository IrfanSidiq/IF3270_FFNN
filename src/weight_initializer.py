from abc import ABC, abstractmethod
import numpy as np


class WeightInitializer(ABC):
    @abstractmethod
    def initialize_weight(neuron_size: int, input_size: int):
        pass


class ZeroInitializer(WeightInitializer):
    @staticmethod
    def initialize_weight(neuron_size: int, input_size: int):
        return np.zeros((neuron_size, input_size))


class RandomUniformInitializer(WeightInitializer):
    @staticmethod
    def initialize_weight(neuron_size: int, input_size: int, lower_bound: int = -1, upper_bound: int = 1, seed: int = None):
        if not seed:
            seed = np.random.randint(0, 2**31 - 1)
        
        rng = np.random.RandomState(seed)
        return rng.uniform(lower_bound, upper_bound, (neuron_size, input_size))
    

class RandomNormalInitializer(WeightInitializer):
    @staticmethod
    def initialize_weight(neuron_size: int, input_size: int, mean: int = -1, variance: int = 1, seed: int = None):
        if not seed:
            seed = np.random.randint(0, 2**31 - 1)
        
        rng = np.random.RandomState(seed)
        standard_deviation = np.sqrt(variance)

        return rng.normal(loc=mean, scale=standard_deviation, size=(neuron_size, input_size))


class GlorotUniformInitializer(WeightInitializer):
    @staticmethod
    def initialize_weight(neuron_size: int, input_size: int, seed: int = None):
        if not seed:
            seed = np.random.randint(0, 2**31 - 1)
        
        rng = np.random.RandomState(seed)
        limit = np.sqrt(6 / (input_size + neuron_size))

        return rng.uniform(-limit, limit, (neuron_size, input_size))


class HeNormalInitializer(WeightInitializer):
    @staticmethod
    def initialize_weight(neuron_size: int, input_size: int, seed: int = None):
        if not seed:
            seed = np.random.randint(0, 2**31 - 1)
        
        rng = np.random.RandomState(seed)
        standard_deviation = np.sqrt(2 / input_size)

        return rng.normal(loc=0, scale=standard_deviation, size=(neuron_size, input_size))