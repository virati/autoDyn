import numpy as np


def sigm(x):
    return 1 / (1 + np.exp(-x))


def wc_drift(params, x, u=0):
    e = np.copy(x[:, 0])  # region number is first, then element inside
    i = np.copy(x[:, 1])  # region number is first, then element inside

    tau = params["tau"]
    alpha = params["alpha"]
    beta = params["beta"]
    w = params["w"]
    thresh = params["thresh"]
    net_k = params["net_k"]
    L = params["L"]

    e_dot = np.zeros(shape=(x.shape[0], 1))
    i_dot = np.zeros(shape=(x.shape[0], 1))

    for nn in range(x.shape[0]):
        e_dot_p = params["T_e"] * (
            -e[nn]
            + sigm(
                -beta["e"]
                * (
                    e[nn] * w["ee"]
                    - i[nn] * w["ei"]
                    - thresh["e"]
                    + net_k * np.dot(L[nn, :], e)
                )
            )
            + u
        )
        i_dot_p = params["T_i"] * (
            -i[nn]
            + sigm(-beta["i"] * (-i[nn] * w["ii"] + e[nn] * w["ie"] - thresh["i"]))
        )

        e_dot[nn] = e_dot_p
        i_dot[nn] = i_dot_p

    return np.array([e_dot, i_dot]).squeeze().T
