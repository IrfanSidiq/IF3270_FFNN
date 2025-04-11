import numpy as np

from src.tensor import Tensor

class RMSNorm:
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        """
        Normalize input tensor using the RMS formula.
        """
        gamma = Tensor(np.ones(len(x.data)))
        rms = np.sqrt(np.mean(x ** 2) + 1e-8)
        normalized = (x.data / rms) * gamma.data
        
        return Tensor(normalized, [x, gamma])

    