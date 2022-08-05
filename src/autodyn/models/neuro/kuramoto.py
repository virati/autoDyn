from numpy import cos


def kuramoto(x, g, u, **kwargs):
    D = kwargs["D"]

    new_x = -D @ cos(D.T @ x)

    return new_x


def delay_kuramoto(x, g, u, **kwargs):
    raise NotImplemented
