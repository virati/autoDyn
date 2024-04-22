import numpy as np

# Dynamics Functions
def consensus(x, **kwargs):
    new_x = np.zeros_like(x)
    D = kwargs["D"]
    if "W" in kwargs.keys():
        W = kwargs["W"]
    else:
        W = np.eye(D.shape[1])

    new_x = -np.dot(np.dot(D, np.dot(W, D.T)), x)

    return new_x


def lorenz(x, **kwargs):
    new_x = np.zeros_like(x)
    sigma = kwargs["sigma"]
    rho = kwargs["rho"]
    beta = kwargs["beta"]

    new_x[0] = sigma * (x[1] - x[0])
    new_x[1] = x[0] * (rho - x[2]) - x[1]
    new_x[2] = x[0] * x[1] - beta * x[2]

    return new_x


def controlled_hopf(x, g, u, **kwargs):
    new_x = np.zeros_like(x)

    r = x[0]
    theta = x[1]

    c = kwargs["c"]
    w = kwargs["w"]

    new_x[0] = -(r**2) * (r - c) - g * u
    new_x[1] = w

    return new_x


def single_hopf(x, **kwargs):
    new_x = np.zeros_like(x)

    r = x[0]
    theta = x[1]

    c = kwargs["c"]
    w = kwargs["w"]

    new_x[0] = -(r**2) * (r - c)
    new_x[1] = w + 0 * theta

    return new_x
