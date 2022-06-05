import numpy as np


def sigm(x):
    return 1 / (1 + np.exp(-x))


def wc_drift(x, **kwargs):
    e = x[0]  # region number is first, then element inside
    i = x[1]  # region number is first, then element inside

    params = kwargs["params"]

    tau = params["tau"]
    alpha = params["alpha"]
    beta = params["beta"]
    w = params["w"]
    thresh = params["thresh"]
    net_k = params["net_k"]
    L = params["L"]

    e_dot = np.zeros(shape=(x.shape[0], 1))
    i_dot = np.zeros(shape=(x.shape[0], 1))
    e_dot_p = params["T_e"] * (
        -e + sigm(-beta["e"] * (e * w["ee"] - i * w["ei"] - thresh["e"]))
    )
    i_dot_p = params["T_i"] * (
        -i + sigm(-beta["i"] * (-i * w["ii"] + e * w["ie"] - thresh["i"]))
    )

    return np.array([e_dot_p, i_dot_p]).squeeze().T
