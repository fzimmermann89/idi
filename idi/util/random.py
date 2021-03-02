from numpy import pi
import numpy as _np
import numexpr as _ne


def rndgennorm(mu, fwhm, rho, N, rng=None):
    """
    samples from the generalised random normal distribution with given shape parameter rho, mean mu and fwhm.
    for rho=2 this is a normal distribution, for higher rho it approaches a uniform distribution between -fwhm/2 and fwhm/2.
    mu, rho, fwhm must be broadcastable to the number of samples N.
    """
    if rng is None:
        rng = _np.random.default_rng()
    rho = _np.asarray(rho)
    if _np.isscalar(N):
        if not _np.isscalar(fwhm):
            N = (N, len(fwhm))
        elif not _np.isscalar(rho):
            N = (N, len(rho))

    # https://sccn.ucsd.edu/wiki/Generalized_Gaussian_Probability_Density_Function
    # https://en.wikipedia.org/wiki/Generalized_normal_distribution
    # ret=mu + fwhm / 2 * (rng.gamma(1 / rho, 1, N) / _np.log(2)) ** (1 / rho) * rng.choice((-1, 1), N)
    c = rng.choice(_np.array([-1, 1], _np.int8), N) # noqa
    ret = rng.gamma(1 / rho, 1, N)
    _ne.evaluate('mu + fwhm / 2 * (ret / log(2)) ** (1 / rho) * c', out=ret)
    return ret


def rndstr(N):
    """
    random string of N lower case ascii letters and numbers
    """
    import random
    import string

    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=N))


def rndSphere(r, N, rng=None):
    if rng is None:
        rng = _np.random.default_rng()
    ret = _np.zeros((N, 3))
    t = rng.uniform(-1, 1, N)
    p = rng.uniform(0, 2 * pi, N)
    u = rng.uniform(0, 1, N)
    _ne.evaluate('r*u**(1/3)', out=u)
    _ne.evaluate('u * t', out=ret[:, 0])
    _ne.evaluate('sqrt(1 - t**2)', out=t)
    _ne.evaluate('u * t * sin(p)', out=ret[:, 1])
    _ne.evaluate('u * t * cos(p)', out=ret[:, 2])
    return ret


def rndConvexpoly(N, n, x, rotmatrix, r, ratio=2):
    if rotmatrix is not None:
        n = _np.matmul(rotmatrix, n.T, order='F').T
        x = _np.matmul(rotmatrix, x.T, order='F').T
    found = 0
    ret = []
    criterion = _np.einsum('ij,ij->i', n, x)
    while found < N:
        points = rndSphere(1, int(_np.clip(ratio * (N - found), 1024, N)))  # steps smalller 1024 are just wastefull, limit memory usage
        inside = _np.ones(len(points), bool)
        for i in range(len(n)):  # loop instead of full vectorisation to decrease memory allocation
            inside &= (points @ n[i]) > criterion[i]
        # inside=_np.all(_np.squeeze(_np.matmul(points[None,...],n[...,None]))>_np.einsum('ij,ij->i',n,x)[...,None],axis=0)
        points = points[inside] * r
        found += len(points)
        ret.append(points)
    return _np.concatenate(ret)[: int(N)]


def rndIcosahedron(r, N, rotmatrix=None):

    phi = (1 + _np.sqrt(5)) / 2
    # surface normals
    n = _np.array(
        [
            [0, -1, 2 - phi],
            [0, -1, phi - 2],
            [2 - phi, 0, -1],
            [phi - 2, -0, -1],
            [phi - 2, 0, 1],
            [2 - phi, 0, 1],
            [0, 1, phi - 2],
            [0, 1, 2 - phi],
            [1, phi - 2, 0],
            [1, 2 - phi, 0],
            [-1, phi - 2, 0],
            [-1, 2 - phi, 0],
            [1, -1, -1],
            [-1, -1, -1],
            [1, -1, 1],
            [-1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
            [1, 1, -1],
            [-1, 1, -1],
        ]
    )
    # point on each surface for r=1
    x = _np.array(
        [
            [0, 1, -phi],
            [0, 1, phi],
            [0, 1, phi],
            [0, 1, phi],
            [0, 1, -phi],
            [0, 1, -phi],
            [0, -1, phi],
            [0, -1, -phi],
            [-1, phi, 0],
            [-1, -phi, 0],
            [1, phi, 0],
            [1, -phi, 0],
            [0, 1, phi],
            [0, 1, phi],
            [0, 1, -phi],
            [0, 1, -phi],
            [0, -1, -phi],
            [0, -1, -phi],
            [0, -1, phi],
            [0, -1, phi],
        ]
    ) * (1 / _np.sqrt(1 + phi ** 2))

    return rndConvexpoly(N, n, x, rotmatrix, r)


def random_rotation(rng=None, amount=1):
    if rng is None:
        rng = _np.random.default_rng()
    # https://doc.lagout.org/Others/Game%20Development/Programming/Graphics%20Gems%203.pdf
    theta, phi, z = rng.uniform(0, (2.0 * pi * amount, 2.0 * pi, amount))
    V = (_np.cos(phi) * _np.sqrt(z), _np.sin(phi) * _np.sqrt(z), _np.sqrt(1.0 - z))
    sint, cost = _np.sin(theta), _np.cos(theta)
    R = _np.array(((cost, sint, 0), (-sint, cost, 0), (0, 0, 1)))
    M = _np.dot(2 * _np.outer(V, V) - _np.eye(3), R)
    return M
