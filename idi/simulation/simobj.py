from __future__ import division as _future_div, print_function as _future_printf
import numpy as _np
import numexpr as _ne
from six import print_ as print
from numpy import pi
import numba as _numba


class atoms:
    def __init__(self, E, pos):
        self._N = len(pos)
        self._E = _np.ones(self._N) * E
        self._pos = pos
        self.rndPhase = False

    def get(self):
        raise NotImplementedError("abstract method")

    @property
    def N(self):
        return self._N

    @property
    def E(self):
        return self._E

    def get(self, rndPhase=True):
        k = 2 * pi / (1.24 / self._E)  # in 1/um
        z = self._pos[2, ...]
        rnd = _np.random.rand(self._N) if rndPhase else 0
        phase = _ne.evaluate('(k*z+rnd*2*pi)%(2*pi)')
        ret = _np.concatenate((self._pos, phase[_np.newaxis, :]))
        return ret


class sphere(atoms):
    def __init__(self, N, r, E):
        pos = self._rndSphere(r, N)
        atoms.__init__(self, E, pos)
        self._r = r
        self.rndPos = False

    @staticmethod
    def _rndSphere(r, N):
        rnd = _np.random.rand(N)
        t = _ne.evaluate('arcsin(2.*rnd-1.)')
        rnd = _np.random.rand(N)
        p = _ne.evaluate('2.*pi*rnd')
        rnd = _np.random.rand(N)
        r = _ne.evaluate('r*rnd**(1./3)')
        x = _ne.evaluate('r * cos(t) * cos(p)')
        y = _ne.evaluate('r * cos(t) * sin(p)')
        z = _ne.evaluate('r * sin(t)')
        return _np.array((x, y, z))

    def get(self, rndPhase=True, rndPos=False):
        k = 2 * pi / (1.24 / self._E)  # in 1/um
        if rndPos:
            self._pos = self._rndSphere(self._r, self._N)
        return atoms.get(self, rndPhase)


class xyzgrid(atoms):
    def __init__(self, filename, atomname, rotangles, E):
        xyz = _np.genfromtxt('Downloads/1010527-5.xyz', dtype=None, skip_header=2)
        pos = _np.array([[x[1], x[2], x[3]] for x in xyz if x[0]==atomname])
        if _np.any(rotangles):
            self._rotmatrix = grid._rotation(*rotangles)
            pos = _np.matmul(self._rotmatrix,pos)
        else:
            self._rotmatrix = None
        atoms.__init__(self, E, pos)

class grid(atoms):
    def __init__(self, lconst, langle, unitcell, Ns, rotangles, E):
        pos = grid._lattice(lconst, langle, unitcell, Ns)
        if _np.any(rotangles):
            self._rotmatrix = grid._rotation(*rotangles)
            pos = _np.matmul(self._rotmatrix, pos)
        else:
            self._rotmatrix = None
        atoms.__init__(self, E, pos)

    @staticmethod
    def _lattice(lconst, langle, unitcell, repeats):
        cosa, cosb, cosc = _np.cos(_np.array(langle) * pi / 180.0)
        sina, sinb, sinc = _np.sin(_np.array(langle) * pi / 180.0)
        basis = _np.array(
            [
                [1, 0, 0],
                [cosc, sinc, 0],
                [cosb, (cosa - cosb * cosc) / sinc, _np.sqrt(sinb ** 2 - ((cosa - cosb * cosc) / sinc) ** 2)],
            ]
        ) * _np.expand_dims(lconst, 1)
        atoms = unitcell

        tmpatoms = []
        for i in range(repeats[0]):
            offset = basis[0] * i
            tmpatoms.append(atoms + offset[_np.newaxis, :])
        atoms = _np.concatenate(tmpatoms)
        tmpatoms = []
        for j in range(repeats[1]):
            offset = basis[1] * j
            tmpatoms.append(atoms + offset[_np.newaxis, :])
        atoms = _np.concatenate(tmpatoms)
        tmpatoms = []
        for k in range(repeats[2]):
            offset = basis[2] * k
            tmpatoms.append(atoms + offset[_np.newaxis, :])
        atoms = _np.concatenate(tmpatoms)
        return (atoms - _np.max(atoms, axis=0) / 2.0).T.copy()

    @staticmethod
    @_numba.njit
    def _rotation(alpha, beta, gamma):
#         #euler angles
#         M = _np.array([[cos(alpha)*cos(gamma)-sin(alpha)*cos(beta)*sin(gamma),
#                        sin(alpha)*cos(gamma)+cos(alpha)*cos(beta)*sin(gamma),
#                        sin(beta)*sin(gamma)],
#                       [-cos(alpha)*sin(gamma)-sin(alpha)*cos(beta)*cos(gamma),
#                        -sin(alpha)*sin(gamma)+cos(alpha)*cos(beta)*cos(gamma),
#                        sin(beta)*cos(gamma)],
#                       [sin(alpha)*sin(beta),
#                        -cos(alpha)*sin(beta),
#                        cos(beta)]])
        # yaw pitch roll
        M = _np.array(
            [
                [
                    cos(beta) * cos(gamma),
                    sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma),
                    cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma),
                ],
                [
                    cos(beta) * sin(gamma),
                    sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma),
                    cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma),
                ],
                [
                    -sin(beta), 
                    sin(alpha) * cos(beta), 
                    cos(alpha) * cos(beta)
                ]
            ]
        )
        return M

    def get(self, rndPhase=True, rndOrientation=False):
        if rndOrientation:
            m = grid._rotation(*2 * pi * _np.random.rand(3))
            self._pos = _np.matmul(m, self._pos)
        return atoms.get(self, rndPhase)

    @property
    def n(self):
        return self._n


class gridsc(grid):
    def __init__(self, N, a, E, rotangles):
        if (_np.array(N)).size == 1:
            N = int(_np.rint(N ** (1 / 3.0)))
            N = [N, N, N]
        lconst = [a, a, a]
        unitcell = [[0, 0, 0]]
        langle = [90, 90, 90]
        grid.__init__(self, lconst, langle, unitcell, N, rotangles, E)


class gridfcc(grid):
    def __init__(self, N, a, E, rotangles):
        if (_np.array(N)).size == 1:
            N = int(_np.rint((N / 4) ** (1 / 3.0)))
            N = [N, N, N]
        lconst = [a, a, a]
        unitcell = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        langle = [90, 90, 90]
        grid.__init__(self, lconst, langle, unitcell, N, rotangles, E)


class gridcuso4(grid):
    def __init__(self, N, E, rotangles):
        N = int(_np.rint((N / 2) ** (1 / 3.0)))
        unitcell = [[0, 0, 0.5], [0, 0.5, 0]]
        lconst = _np.array([0.60, 0.61, 1.07]) * 1e-4
        langle = _np.array([77.3, 82.3, 72.6])
        Ns = [N, N, N]
        grid.__init__(self, lconst, langle, unitcell, Ns, rotangles, E)
