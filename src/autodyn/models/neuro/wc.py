import numpy as np


def sigm(x):
    return 1 / (1 + np.exp(-x))


def wc_drift(x, **kwargs):
    e = x[0]
    i = x[1]

    params = kwargs["params"]

    tau = params["tau"]
    alpha = params["alpha"]
    beta = params["beta"]
    w = params["w"]
    thresh = params["thresh"]

    e_dot_p = params["T_e"] * (
        -e + sigm(-beta["e"] * (e * w["ee"] - i * w["ei"] - thresh["e"]))
    )
    i_dot_p = params["T_i"] * (
        -i + sigm(-beta["i"] * (-i * w["ii"] + e * w["ie"] - thresh["i"]))
    )

    new_state = np.array([e_dot_p, i_dot_p])

    return new_state


def wc_input(x, **kwargs):
    e = x[0]
    i = x[1]

    params = kwargs["params"]

    tau = params["tau"]
    alpha = params["alpha"]
    beta = params["beta"]
    w = params["w"]
    thresh = params["thresh"]

    u = kwargs["u"]

    e_dot_p = (
        params["T_e"]
        * (-e + sigm(-beta["e"] * (e * w["ee"] - i * w["ei"] - thresh["e"])))
        + u
    )
    i_dot_p = params["T_i"] * (
        -i + sigm(-beta["i"] * (-i * w["ii"] + e * w["ie"] - thresh["i"]))
    )

    new_state = np.array([e_dot_p, i_dot_p])

    return new_state
