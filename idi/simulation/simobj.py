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

    @property
    def N(self):
        return self._N

    @property
    def E(self):
        return self._E

    def get(self, rndPhase=True):
        k = 2 * pi / (1.24 / self._E)  # in 1/um
        z = self._pos[..., 2]
        rnd = _np.random.rand(self._N) if rndPhase else 0
        phase = _ne.evaluate('(k*z+rnd*2*pi)%(2*pi)')
        ret = _np.concatenate((self._pos, phase[:, _np.newaxis]),axis=1)
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
        r = _ne.evaluate('r*rnd**(1./3.)')
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
        import re
        with open(filename, 'r') as file:
            data = file.read()
        lines=re.findall("^"+atomname+"\d*\s*[\d,\.]*\s*[\d,\.]*\s*[\d,\.]*",data,re.IGNORECASE | re.MULTILINE)
        pos=_np.genfromtxt(lines)[:,1:]
        if _np.any(rotangles):
            self._rotmatrix = grid._rotation(*rotangles)
            pos = _np.matmul(pos, self._rotmatrix)
        else:
            self._rotmatrix = None
        atoms.__init__(self, E, pos)


class grid(atoms):
    def __init__(self, lconst, langle, unitcell, Ns, rotangles, E):
        pos = grid._lattice(lconst, langle, unitcell, Ns)
        if _np.any(rotangles):
            self._rotmatrix = grid._rotation(*rotangles)
            pos2 = _np.matmul(pos,self._rotmatrix)
        else:
            self._rotmatrix = None
        atoms.__init__(self, E, pos)

    @staticmethod
    def _lattice(lconst, langle, unitcell, repeats):
        cosa, cosb, cosc = _np.cos(_np.array(langle))
        sina, sinb, sinc = _np.sin(_np.array(langle))
        basis = _np.array(
            [
                [1, 0, 0],
                [cosc, sinc, 0],
                [cosb, (cosa - cosb * cosc) / sinc, _np.sqrt(sinb ** 2 - ((cosa - cosb * cosc) / sinc) ** 2)],
            ]
        ) * _np.expand_dims(lconst, 1)
        atoms = unitcell * _np.array(lconst)
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
        return (atoms - _np.max(atoms, axis=0) / 2.0)

    @staticmethod
    @_numba.njit
    def _rotation(alpha, beta, gamma):
        cosa, cosb, cosg = _np.cos(_np.array((alpha, beta, gamma)))
        sina, sinb, sing = _np.sin(_np.array((alpha, beta, gamma)))
        
        # # euler angles
        # M = _np.array(
        #     [
        #         [
        #             cosa * cosg - sina * cosb * sing,
        #            sina * cosg + cosa * cosb * sing,
        #             sinb * sing
        #         ],
        #         [
        #             -cosa * sing - sina * cosb * cosg,
        #             -sina * sing + cosa * cosb * cosg,
        #             sinb*cosg
        #         ],
        #         [
        #             sina * sinb,
        #             -cosa * sinb,
        #             cosb
        #         ]
        #     ]
        # )

        # yaw pitch roll
        M = _np.array(
            [
                [
                    cosb * cosg,
                    sina * sinb * cosg - cosa * sing,
                    cosa * sinb * cosg + sina * sing,
                ],
                [
                    cosb * sing,
                    sina * sinb * sing + cosa * cosg,
                    cosa * sinb * sing - sina * cosg,
                ],
                [
                    -sinb,
                    sina * cosb,
                    cosa * cosb
                ]
            ]
        )
        return M

    def get(self, rndPhase=True, rndOrientation=False):
        if rndOrientation:
            rotmatrix = grid._rotation(*2 * pi * _np.random.rand(3))
            self._pos = _np.matmul(self._pos, rotmatrix)
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
        langle = _np.array([90, 90, 90]) * pi / 180.0
        grid.__init__(self, lconst, langle, unitcell, N, rotangles, E)


class gridfcc(grid):
    def __init__(self, N, a, E, rotangles):
        if (_np.array(N)).size == 1:
            N = int(_np.rint((N / 4.0) ** (1 / 3.0)))
            N = [N, N, N]
        lconst = [a, a, a]
        unitcell = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        langle = _np.array([90, 90, 90]) * pi / 180.0
        grid.__init__(self, lconst, langle, unitcell, N, rotangles, E)


class gridcuso4(grid):
    def __init__(self, N, E, rotangles):
        N = int(_np.rint((N / 2.0) ** (1 / 3.0)))
        unitcell = [[0, 0, 0.5], [0, 0.5, 0]]
        lconst = _np.array([0.60, 0.61, 1.07]) * 1e-3
        langle = _np.array([77.3, 82.3, 72.6]) * pi / 180.0
        Ns = [N, N, N]
        grid.__init__(self, lconst, langle, unitcell, Ns, rotangles, E)