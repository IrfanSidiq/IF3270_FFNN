from abc import ABC, abstractmethod
import numpy as np


class WeightInitializer(ABC):
    @abstractmethod
    def initialize_weight(neuron_size: int, input_size: int) -> np.ndarray:
        """
        Initializes weight vector and returns it as a NumPy array.
        """
        pass


class ZeroInitializer(WeightInitializer):
    def __init__(self):
        pass

    def initialize_weight(neuron_size: int, input_size: int):
        return np.zeros((neuron_size, input_size))


class RandomUniformInitializer(WeightInitializer):
    def __init__(self, lower_bound: int = -1, upper_bound: int = 1, seed: int = None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.seed = seed

    def initialize_weight(self, neuron_size: int, input_size: int):
        if not self.seed:
            self.seed = np.random.randint(0, 2**31 - 1)
        
        rng = np.random.RandomState(self.seed)
        return rng.uniform(self.lower_bound, self.upper_bound, (neuron_size, input_size))
    

class RandomNormalInitializer(WeightInitializer):
    def __init__(self, mean: int = -1, variance: int = 1, seed: int = None):
        self.mean = mean
        self.variance = variance
        self.seed = seed

    def initialize_weight(self, neuron_size: int, input_size: int):
        if not self.seed:
            self.seed = np.random.randint(0, 2**31 - 1)
        
        rng = np.random.RandomState(self.seed)
        standard_deviation = np.sqrt(self.variance)

        return rng.normal(loc=self.mean, scale=standard_deviation, size=(neuron_size, input_size))


class GlorotUniformInitializer(WeightInitializer):
    def __init__(self, seed: int = None):
        self.seed = seed

    def initialize_weight(self, neuron_size: int, input_size: int):
        if not self.seed:
            self.seed = np.random.randint(0, 2**31 - 1)
        
        rng = np.random.RandomState(self.seed)
        limit = np.sqrt(6 / (input_size + neuron_size))

        return rng.uniform(-limit, limit, (neuron_size, input_size))


class HeNormalInitializer(WeightInitializer):
    def __init__(self, seed: int = None):
        self.seed = seed

    def initialize_weight(self, neuron_size: int, input_size: int):
        if not self.seed:
            self.seed = np.random.randint(0, 2**31 - 1)
        
        rng = np.random.RandomState(self.seed)
        standard_deviation = np.sqrt(2 / input_size)

        return rng.normal(loc=0, scale=standard_deviation, size=(neuron_size, input_size))