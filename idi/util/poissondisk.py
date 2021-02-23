import math as _math
import numpy as _np
import numba as _numba
from scipy.spatial import cKDTree as _cKDTree


def poisson_disc_sample(r, d, N=_np.inf, ndim=3, k=10, method='auto', rng=None):
    """
    Random points with minimum distance in between inside n-sphere
    Parameters:
    r: radius n-sphere
    d: minidmum distance
    ndim: dimension of n-sphere
    N: number of samples
    method: string
        'darts': throw darts and check if darts fit
        'bridson' use bridson algorithm
        'bridson_dense': use bridson algorithm with minimum distance between sample canditates. will result in denser samples
        'auto': will use darts for small N and bridson only for big N
    k: fidelity parameter, should be >10 for good results
    """
    if rng is None:
        rng = _np.random.default_rng(seed=_np.random.randint(2 ** 63))

    ndim = int(ndim)
    if ndim < 1:
        raise ValueError('ndim should be integer >=1')
    if r < 0 or d < 0:
        raise ValueError('r and d should be positive')
    if d == 0:
        if 'bridson' in method:
            raise ValueError('bridson needs d>0')
        if not _np.isfinite(N):
            raise ValueError('Zero distance would cause infinite number of spheres. Limit N or set distance >0')
        method = 'darts'
    if (method == 'auto' and (N > 0.75 * (1.33 * r / d) ** ndim or N > 1e6)) or (method == 'bridson' or method == 'bridson_dense'):
        fixd = method == 'bridson_dense'
        cellsize = d / _math.sqrt(ndim)
        grid_shape = int(_np.ceil((2 * r + 2 * d) / cellsize))
        grid = _np.zeros(ndim * [grid_shape], dtype=_np.int64)
        points = _poisson_disc_sample_bridson(grid, r, d, ndim, k, fixd=fixd, seed=rng.integers(2 ** 63))
        points = points[(_np.einsum('ij,ij->i', points, points)) < r ** 2]
        if len(points) > N:
            points = points[rng.choice(range(len(points)), int(N)), :]
    elif method == 'darts' or method == 'auto':
        points = _poisson_disc_sample_darts(r, d, N, ndim, k=k)
    else:
        raise NotImplementedError(f'method {method} unknown')
    return points


@_numba.njit()
def _poisson_disc_sample_bridson(grid, r, d, ndim=3, k=10, fixd=False, seed=0):
    """
    using the nd-bridson algorithm. the grid has to be preallocated
    """
    if seed != 0:
        _np.random.seed(seed)

    cellsize = d / _math.sqrt(ndim)
    grid_shape = len(grid)
    gstride = (_np.array(grid.strides) / grid.itemsize).astype(_np.int64)
    p = _np.zeros(ndim)
    while _np.sum((p - r - d) ** 2) > r ** 2:
        p = 2 * r * _np.random.rand(ndim) + d - r
    if grid_shape < 2:
        return p.reshape(1, -1)
    points = _np.zeros((1024, ndim))
    points[0, :] = p[:]
    queue = [0]
    coords = (p / cellsize).astype(_np.int64)
    n = 0
    flatcoord = _np.sum(gstride * coords)
    grid.ravel()[flatcoord] = n + 1

    while len(queue):
        q = points[queue.pop(_np.random.randint(0, len(queue))), :]
        rand = _np.random.randn(k, ndim)
        norm = _np.sum(rand ** 2, axis=-1) ** (1 / 2)
        rs = _np.ones(k) * d if fixd else d * (1 + (2 ** ndim - 1) * _np.random.rand(k)) ** (1 / ndim)
        ps = rand * (rs / norm).reshape(-1, 1) + q
        for ip in range(len(ps)):
            p = ps[ip, :]
            coord = (p / cellsize).astype(_np.int64)
            if not _fits_bridson(p, d, points, grid, coord):
                continue
            else:
                n += 1
                grid.ravel()[_np.sum(gstride * coord)] = n + 1
                if n >= len(points):
                    tmp = _np.zeros((2 * len(points), ndim))
                    tmp[: len(points)] = points
                    points = tmp
                points[n, :] = p[:]
                if _np.sum((p - r - d) ** 2) < (r + 2 * d) ** 2:
                    queue.append(n)
    return points[: n + 1, :] - r - d


@_numba.njit()
def _fits_bridson(p, d, points, grid, coords):
    """
    does p fit?
    """
    if coords[0] >= len(grid) or coords[0] < 0:
        return False
    elif grid.ndim == 2:  # fastpath for ndim==2
        if coords[1] >= grid.shape[1] or coords[1] < 0:
            return False
        for i in range(max(coords[0] - 2, 0), min(coords[0] + 3, grid.shape[0])):
            for j in range(max(coords[1] - 2, 0), min(coords[1] + 3, grid.shape[1])):
                if grid[i, j] != 0 and _np.sum((p - points[grid[i, j] - 1]) ** 2) <= d ** 2:
                    return False
        return True

    elif grid.ndim > 1:  # recurse in the dimensions
        for i in range(max(coords[0] - 2, 0), min(coords[0] + 3, len(grid))):
            if not _fits_bridson(p, d, points, grid[i, ...], coords[1:]):
                return False
    else:  # last dimension
        for i in range(max(coords[0] - 2, 0), min(coords[0] + 3, len(grid))):
            if grid[i] != 0 and _np.sum((p - points[grid[i] - 1]) ** 2) <= d ** 2:
                return False
    return True


def _poisson_disc_sample_darts(r, mindistance, N, d=3, m=None, k=10, rng=None):
    """
    throwing darts and checking using ndtree
    """
    if rng is None:
        rng = _np.random.default_rng(seed=_np.random.randint(2**63))
    N = _np.clip(N, 1, (1.35 * r / mindistance) ** d) if mindistance > 0 else N
    if m is None:
        m = max(1, N / 4)
    m = int(min(N, m))
    N = int(N)
    found = 0
    points = _np.zeros((0, d))
    maxtries = k * N / m
    for k in range(int(maxtries)):
        rand = rng.standard_normal((m, d + 2))
        newpoints = rand[:, :d] * ((r + mindistance) / _np.sqrt(_np.einsum('ij,ij->i', rand, rand)))[:, None]
        tree = _cKDTree(newpoints)
        n = tree.query_ball_tree(tree, mindistance)
        nn = [len(el) for el in n]
        mask = _np.ones(m, bool)
        for i in range(len(n)):
            if nn[i] > 1:
                mask[i] = False
                for j in n[i][1:]:
                    nn[j] -= 1
        newpoints = newpoints[mask, :]
        if len(points) > 0:
            newpoints = newpoints[_cKDTree(points, balanced_tree=False).query(newpoints, n_jobs=4, distance_upper_bound=mindistance)[0] > mindistance]
        found += _np.sum((_np.einsum('ij,ij->i', newpoints, newpoints)) < r ** 2)
        points = _np.vstack((points, newpoints))
        if found > N:
            break
    points = points[(_np.einsum('ij,ij->i', points, points)) < r ** 2]
    if len(points) > N:
        points = points[:N]
    return points
