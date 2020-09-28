from __future__ import division as _future_div, print_function as _future_printf
import numpy as _np
import numexpr as _ne
from six import print_ as _print
from numpy import pi
import numba as _numba
import math as _math
import random as _random

class atoms:
    def __init__(self, E, pos):
        self._N = len(pos)
        self._E = E  # _np.ones(self._N) * E
        self._pos = pos
        self.rndPhase = True

    @property
    def N(self):
        return self._N

    @property
    def E(self):
        return self._E

    def get(self):
        k = 2 * pi / (1.24 / self._E)  # in 1/um
        z = self._pos[..., 2]
        rnd = _np.random.rand(self._N) if self.rndPhase else 0
        phase = _ne.evaluate('(k*z+rnd*2*pi)%(2*pi)')
        ret = _np.concatenate((self._pos, phase[:, _np.newaxis]), axis=1)
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
        return _np.stack((x, y, z), axis=1)

    def get(self):
        if self.rndPos:
            self._pos = self._rndSphere(self._r, self._N)
        return atoms.get(self)


from ..util import poisson_disc_sample as _pds
class multisphere(atoms):
    @staticmethod
    @_numba.njit()
    def _staggeredadd(pos1, pos2, n):
        """
        repeat pos1[i] n[i] times and add it to pos2
        """
        pos2[: n[0], :] += pos1[0, :]
        for i in range(1, len(n)):
            pos2[n[i - 1] : min(len(pos2), n[i]), :] += pos1[i, :]
        pos2[n[-1] :, :] += pos1[-1, :]

    def __init__(self, E, Natoms=1e6, rsphere=10, fwhmfocal=200, spacing=1, Nspheres=_np.inf):
        """
        Multiple Spheres
        Parameters:
        Natoms: total number of excited atoms
        rsphere: radius of each spehre
        fwhmfocal: the number of atoms per sphere scales gaussian with the distance from the center. this sets the fhwm of the gaussian.
        spacing: thickness of atom-free layer around each sphere
        Nspheres: max. total number of particles in volume
        """
        self._N = Natoms
        self._Nspheres = Nspheres
        self.rsphere = rsphere
        self.fwhmfocal = fwhmfocal
        self.spacing = spacing
        atoms.__init__(self, E, self._atompos())
        self.rndPos = True
        self._debug = None

    def _atompos(self):
        """
        calculate positin of atoms
        """
        self._posspheres = _pds(1.2 * self.fwhmfocal, 2 * (self.rsphere + self.spacing),ndim=3, N=self._Nspheres)
        r = _np.sqrt(_np.sum(self._posspheres ** 2, axis=1))
        p = _np.exp(-_np.square((r) / (0.4 * self.fwhmfocal)) / 2.0)
        n = _np.random.poisson(p / _np.sum(p) * self._N)
        missing = self._N - _np.sum(n)
        if missing > 0:
            n[_np.argsort(n)[: -int(missing) - 1 : -1]] += 1
        nc = _np.cumsum(n)
        posatoms = sphere._rndSphere(self.rsphere, int(self._N))
        multisphere._staggeredadd(self._posspheres, posatoms, nc)
        self._debug = (len(self._posspheres),_np.min(n),_np.max(n),_np.mean(n))
        return posatoms

    def get(self):
        if self.rndPos:
            self._pos = self._atompos()
        return atoms.get(self)

    
   
    



class xyzgrid(atoms):
    def __init__(self, filename, atomname, rotangles, E):
        import re

        with open(filename, 'r') as file:
            data = file.read()
        lines = re.findall("^" + atomname + "\d*\s*[\d,\.]+\s+[\d,\.]+\s+[\d,\.]+", data, re.IGNORECASE | re.MULTILINE)
        pos = _np.genfromtxt(lines)[:, 1:] * 1e-4
        if _np.any(rotangles):
            self._rotmatrix = grid._rotation(*rotangles)
            pos = _np.matmul(pos, self._rotmatrix)
        else:
            self._rotmatrix = None
        pos = pos - (_np.max(pos, axis=0) / 2.0)
        atoms.__init__(self, E, pos)


class grid(atoms):
    def __init__(self, lconst, langle, unitcell, Ns, rotangles, E):
        pos = grid._lattice(lconst, langle, unitcell, Ns)
        if _np.any(rotangles):
            self._rotmatrix = grid._rotation(*rotangles)
            pos = _np.matmul(pos, self._rotmatrix)
        else:
            self._rotmatrix = None
        atoms.__init__(self, E, pos)
        self.rndOrientation = False

    @staticmethod
    def _lattice(lconst, langle, unitcell, repeats, sigma=0):
        cosa, cosb, cosc = _np.cos(_np.array(langle))
        sina, sinb, sinc = _np.sin(_np.array(langle))
        basis = (
            _np.array(
                [
                    [1, 0, 0],
                    [cosc, sinc, 0],
                    [cosb, (cosa - cosb * cosc) / sinc, _np.sqrt(sinb ** 2 - ((cosa - cosb * cosc) / sinc) ** 2)],
                ]
            )
            * _np.expand_dims(lconst, 1)
        )
        atoms = _np.dot(unitcell, basis)
        atoms += sigma * _np.random.rand(*atoms.shape)

        tmpatoms = []
        for i in range(repeats[0]):
            offset = basis[0] * i
            tmpatoms.append(atoms + offset[_np.newaxis, :])
        atoms = _np.concatenate(tmpatoms)
        atoms += sigma * _np.random.rand(*atoms.shape)

        tmpatoms = []
        for j in range(repeats[1]):
            offset = basis[1] * j
            tmpatoms.append(atoms + offset[_np.newaxis, :])
        atoms = _np.concatenate(tmpatoms)
        atoms += sigma * _np.random.rand(*atoms.shape)

        tmpatoms = []
        for k in range(repeats[2]):
            offset = basis[2] * k
            tmpatoms.append(atoms + offset[_np.newaxis, :])
        atoms = _np.concatenate(tmpatoms)
        atoms += sigma * _np.random.rand(*atoms.shape)

        return atoms - _np.max(atoms, axis=0) / 2.0

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
                [-sinb, sina * cosb, cosa * cosb],
            ]
        )
        return M

    @staticmethod
    @_numba.njit
    def _random_rotation(amount=1):
        deflection = 1
        # https://doc.lagout.org/Others/Game%20Development/Programming/Graphics%20Gems%203.pdf
        theta, phi, z = _np.random.rand(3) * _np.array((2.0 * _np.pi * amount, 2.0 * _np.pi, amount))
        V = (_np.cos(phi) * _np.sqrt(z), _np.sin(phi) * _np.sqrt(z), _np.sqrt(1.0 - z))
        sint, cost = _np.sin(theta), _np.cos(theta)
        R = _np.array(((cost, sint, 0), (-sint, cost, 0), (0, 0, 1)))
        M = _np.dot(2 * _np.outer(V, V) - _np.eye(3), R)
        return M

    def get(self):
        if self.rndOrientation:
            rotmatrix = grid._random_rotation()
            self._pos = _np.matmul(self._pos, rotmatrix)
        return atoms.get(self)


#     @property
#     def n(self):
#         return self._n


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


class gridhcp(grid):
    def __init__(self, N, a, E, rotangles):
        if (_np.array(N)).size == 1:
            N = int(_np.rint((N / 2.0) ** (1 / 3.0)))
            N = [N, N, N]
        lconst = [a, a, 1.633 * a]
        unitcell = [[0, 0, 0], [1.0 / 3, 2.0 / 3, 1.0 / 2]]
        langle = _np.array([90, 90, 120]) * pi / 180.0
        grid.__init__(self, lconst, langle, unitcell, N, rotangles, E)


class gridcuso4(grid):
    def __init__(self, N, E, rotangles):
        N = int(_np.rint((N / 2.0) ** (1 / 3.0)))
        # https://doi.org/10.1524%2Fzkri.1975.141.5-6.330
        unitcell = [[0, 0, 0], [0.5, 0.5, 0]]
        lconst = _np.array([6.141, 10.736, 5.986]) * 1e-4
        langle = _np.array([82.27, 107.43, 102.67]) * pi / 180.0
        Ns = [N, N, N]
        grid.__init__(self, lconst, langle, unitcell, Ns, rotangles, E)


class hcpspheres(atoms):
    def __init__(self, Nhcp, Nsphere, a, r, E, rotangles, sigma=0):
        if (_np.array(Nhcp)).size == 1:
            Nhcp = int(_np.rint((Nhcp / 2.0) ** (1 / 3.0)))
            Nhcp = [Nhcp, Nhcp, Nhcp]
        lconst = [a, a, 1.633 * a]
        unitcell = [[0, 0, 0], [1.0 / 3, 2.0 / 3, 1.0 / 2]]
        langle = _np.array([90, 90, 120]) * pi / 180.0
        self._hcppos = grid._lattice(lconst, langle, unitcell, Nhcp, sigma)
        if _np.any(rotangles):
            self._rotmatrix = grid._rotation(*rotangles)
            self._hcppos = _np.matmul(self._hcppos, self._rotmatrix)
        else:
            self._rotmatrix = None
        import random

        pos = []
        for p in self._hcppos:
            rr = random.gauss(r, 0.1 * r)
            pos.append(sphere._rndSphere(rr, Nsphere) + p)
        pos = _np.concatenate(pos)
        atoms.__init__(self, E, pos)
        self.rndPos = False
        self.rndOrientation = False
        self._Nhcp, self._Nsphere, self._r, self._a, self._sigma = Nhcp, Nsphere, r, a, sigma
        self._lconst, self._langle, self._unitcell = lconst, langle, unitcell

    def get(self):
        if self._sigma != 0 and self.rndPos:
            self._hcppos = grid._lattice(self._lconst, self._langle, self._unitcell, self._Nhcp, self._sigma)
        if self.rndOrientation:
            rotmatrix = grid._random_rotation()
            self._hcppos = _np.matmul(self._hcppos, rotmatrix)
        if self.rndOrientation or self.rndPos:
            import random

            pos = []
            for p in self._hcppos:
                rr = random.gauss(self._r, 0.2 * self._r)
                pos.append(sphere._rndSphere(rr, self._Nsphere) + p)
            self._pos = _np.concatenate(pos)
        return atoms.get(self)
