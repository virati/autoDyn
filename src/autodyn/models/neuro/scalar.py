from numpy import cos


def kuramoto(x, g=None, u=0, **kwargs):
    D = kwargs["params"]["D"]
    w = kwargs["params"]["w"]

    new_x = w - D @ cos(D.T @ x)

    return new_x


def delay_kuramoto(x, g=None, u=0, **kwargs):
    raise NotImplemented
