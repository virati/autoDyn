import numpy as np
from typing import List


def unity(x):
    return x


def quadratic(x: np.ndarray, params: List):
    return (x - params[0]) * (x - params[1])

def rectify(x):
    return np.maximum(x, 0)