import abc as _abc
from warnings import warn as _warn
import numba as _numba
import numexpr as _ne
import numpy as _np
from ..util import angles as _angles
from ..util import axisrotation as _axisrotation
from ..util import fastlen as _fastlen
from ..util import poisson_disc_sample as _pds
from ..util import random_rotation as _random_rotation
from ..util import rndgennorm as _rndgennorm
from ..util import rndIcosahedron as _rndIcosahedron
from ..util import rndSphere as _rndSphere
from ..util import rotation as _rotation
from ..util import gnorm as _gnorm

from numpy import pi
from gc import collect as _gc

"""
simulation objects
"""


class simobj(_abc.ABC):
    """
    baseclass
    """

    rng = _np.random.default_rng(_np.random.randint(2 ** 63))
    _pos = None
    _resetproperties = []
    _rotmatrix = None

    def __init__(self, E, N):
        self.E = E  # _np.ones(self._N) * E
        self.rndPhase = True
        self.rndPos = True
        self._N = int(N)

    @property
    def N(self):
        return self._N

    @property
    def k(self):
        return 2 * pi / (1.24 / self.E)

    def get2(self):
        if self.rndPos or self._pos is None:
            self._pos = None
            # _gc()
            self.updatePos()

        if self.rndPhase:
            phase = self.rng.uniform(0, 2 * pi, (self.N, 1))
        else:
            phase = _np.zeros((self.N, 1), self._pos.dtype)
        pos = self._pos
        return pos, phase

    def get(self):
        return _np.hstack(self.get2())

    def getImg(self, dx, ndim=2):
        if not 0 < ndim <= 3:
            raise ValueError
        pos, phase = self.get2()
        ind = _np.rint((pos[:, :ndim] - _np.min(pos[:, :ndim], axis=0, keepdims=True)) / dx).astype(int)
        s = _np.array(ind.max(0)) + 1
        pads = _np.array([_fastlen(i) for i in s])
        ind += (pads - s) // 2
        img = _np.zeros(pads, _np.complex128)
        _np.add.at(img, tuple(ind.T), _np.exp(1j * _np.squeeze(phase)))
        return img

    @_abc.abstractmethod
    def updatePos(self):
        pass

    @property
    def rotangles(self):
        if self._rotmatrix is None:
            _warn('Rotation not used for this object')
            return [0, 0, 0]
        elif self._rotmatrix is False:
            return [0, 0, 0]
        else:
            return _angles(self._rotmatrix)

    @rotangles.setter
    def rotangles(self, value):
        if self._rotmatrix is None:
            _warn('Rotation not used for this object')
        elif not _np.any(value):
            self._rotmatrix = False
        elif self._rotmatrix is False:
            newmatrix = _rotation(*value)
            self._rotmatrix = newmatrix
        else:
            newmatrix = _rotation(*value)
            self._rotate(newmatrix @ self._rotmatrix.T)
            self._rotmatrix = newmatrix

    def __setattr__(self, name, value):
        # reset cached values if property changes
        if any(name == n for n in self._resetproperties):
            self._pos = None
        # _gc()
        super().__setattr__(name, value)

    def _rotate(self, rotmatrix):
        pass


class sphere(simobj):
    """
    a sphere with random positions inside
    """

    def __init__(self, E, N, r):
        self.r = r
        super().__init__(E, N)
        self._resetproperties = ['r']

    def updatePos(self):
        self._pos = _rndSphere(self.r, self.N)


class icosahedron(simobj):
    """
    icosahedron with random positions inside
    """

    def __init__(self, E, N, r, rotangles=(0, 0, 0)):
        self.r = r
        self.rndOrientation = False
        self._rotmatrix = _rotation(*rotangles)
        super().__init__(E, N)
        self._resetproperties = ['r']

    def updatePos(self):
        if self.rndOrientation:
            self._rotmatrix = _random_rotation(self.rng)
        self._pos = _rndIcosahedron(self.r, self._N, self._rotmatrix)

    def _rotate(self, rotmatrix):
        self._pos = _np.matmul(rotmatrix, self._pos.T, order='F').T


class gnorm(simobj):
    """
    a generalised normal shaped volume with random positions inside
    """

    def __init__(self, E, N, fwhm, rho=2, rotangles=(0, 0, 0)):
        self.fwhm, self.rho = _np.array(fwhm), _np.array(rho)
        self._rotmatrix = _rotation(*rotangles) if _np.any(rotangles) else False
        super().__init__(E, N)
        self._resetproperties = ['fwhm', 'rho']

    def updatePos(self):
        _gc()
        if _np.all(self.rho == 2):
            self._pos = self.rng.normal(scale=(self.fwhm / 2.355), size=(self.N, 3))
        else:
            self._pos = _rndgennorm(0, self.fwhm, self.rho, (self.N, 3), self.rng)
        if self._rotmatrix is not False:
            self._rotate(self._rotmatrix)

    def _rotate(self, rotmatrix):
        self._pos = _np.matmul(rotmatrix, self._pos.T, order='F').T


class gauss(gnorm):
    """
    a gaussian shaped volume with random positions inside
    """

    def __init__(self, E, N, fwhm):
        super().__init__(E, N, fwhm, 2)


class multisphere(simobj):
    """
    multiple, randomly positioned spheres
    """

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

    def __init__(self, E, Natoms=1e6, rsphere=10, fwhm=200, rho=2, spacing=1, Nspheres=_np.inf):
        """
        Multiple Spheres
        Parameters:
        Natoms: total number of excited atoms, if negative: mean number per sphere
        rsphere: radius of each sphere
        fwhm: the number of atoms per sphere scales generalised gaussian with the distance from the center. fhwm of the generalised gaussian.
        rho: parameter for generalised gaussian, rho=2 is gaussian in all directions.
        spacing: thickness of atom-free layer around each sphere
        Nspheres: max. total number of particles in volume
        """

        self._Nspheres, self.rsphere, self.spacing = Nspheres, rsphere, spacing
        self.fwhm, self.rho = _np.array(fwhm), _np.array(rho)
        self.rndPos = True
        self._debug = None
        N = int(-Natoms * _np.mean([len(self._spherepos()) for i in range(10)])) if Natoms < 0 else Natoms
        super().__init__(E, N)
        self._resetproperties = ['rndPos', 'rsphere', 'fwhm', 'spacing', 'rho', '_Nspheres']

    def _spherepos(self):
        return _pds(1.2 * self.fwhm, 2 * (self.rsphere + self.spacing), ndim=3, N=self._Nspheres)

    def updatePos(self):
        """
        calculate position of atoms
        """
        posspheres = self._spherepos()
        p = _np.exp(
            _ne.evaluate('sum(-pos**rho*s, axis=1)', local_dict={'s': _np.log(2) * (2 / self.fwhm) ** self.rho, 'pos': posspheres, 'rho': self.rho})
        )
        n = self.rng.poisson(p * (self._N / _np.sum(p)))
        missing = self._N - _np.sum(n)
        while missing:
            ids = self.rng.choice(len(n), int(abs(missing)), replace=True, p=p / (_np.sum(p)))
            _np.add.at(n, ids, _np.sign(missing))
            n[n < 0] = 0
            missing = self._N - _np.sum(n)
        nc = _np.cumsum(n)
        if self.rsphere > 0:
            posatoms = _rndSphere(self.rsphere, int(self._N))
        else:
            posatoms = _np.zeros((self._N, 3))
        multisphere._staggeredadd(posspheres, posatoms, nc)
        self._debug = (len(posspheres), _np.min(n), _np.max(n), _np.mean(n))
        self._pos = posatoms


class hcpsphere(multisphere):
    """
    densly hcp packed spheres
    """

    def __init__(self, E, Natoms=1e6, rsphere=10, fwhm=200, rho=2, a=20, rotangles=(0, 0, 0), sigma=0):
        self.Nhcp = None
        self.rndOrientation = False
        self.rsphere, self.a, self.sigma, self.rsphere = rsphere, a, sigma, rsphere
        self.fwhm, self.rho = _np.array(fwhm), _np.array(rho)
        self._rotmatrix = _rotation(*rotangles) if _np.any(rotangles) else False
        self._resetproperties = ['rsphere', 'fwhm', 'a', 'sigma', 'rho']
        super().__init__(E, Natoms)

    def __setattr__(self, name, value):
        if any(name == n for n in ['sigma', 'a', 'fwhm']):
            self._hcp = None
        super().__setattr__(name, value)

    def _rotate(self, rotmatrix):
        self._hcp = _np.matmul(rotmatrix, self._hcp.T, order='F').T
        self._pos = _np.matmul(rotmatrix, self._pos.T, order='F').T

    def _spherepos(self):
        if self._hcp is None or self.sigma:
            Nhcp = self.Nhcp or _np.ceil(5 * self.fwhm / (self.a * _np.array([2.0, 0.86, 1.63]))).astype(int)
            lconst = [self.a, self.a, 1.633 * self.a]
            unitcell = [[0, 0, 0], [1.0 / 3, 2.0 / 3, 1.0 / 2]]
            langle = _np.array([90, 90, 120]) * pi / 180.0
            hcp = crystal._lattice(lconst, langle, unitcell, Nhcp, self.sigma, self.rng)
            hcp -= _np.mean(hcp, axis=0)
            self._hcp = hcp

        if _np.any(self.rndOrientation):
            if _np.size(self.rndOrientation) == 3:
                rotmatrix = _axisrotation(_np.array(self.rndOrientation, dtype=_np.float64), self.rng.uniform(0, 2 * pi))
            else:
                rotmatrix = _random_rotation(self.rng)
            if self._rotmatrix is not False:
                self._rotmatrix = rotmatrix @ self._rotmatrix
            else:
                self._rotmatrix = rotmatrix
        if self._rotmatrix is not False:
            self._hcp = _np.matmul(self._rotmatrix, self._hcp.T, order='F').T
        return self._hcp


class xyz(simobj):
    """
    atom positions specified by an xyz file
    """

    def __init__(self, E, filename, atomname, rotangles=(0, 0, 0), scale=1e-4):
        import re

        with open(filename, 'r') as file:
            data = file.read()
        lines = re.findall("^" + atomname + r"\d*\s*[\d,\.]+\s+[\d,\.]+\s+[\d,\.]+", data, re.IGNORECASE | re.MULTILINE)
        pos = _np.genfromtxt(lines)[:, 1:] * scale
        self._pos = pos - (_np.max(pos, axis=0) / 2.0)
        self._rotmatrix = _rotation(*rotangles)
        self.rndOrientation = False
        if _np.any(rotangles):
            pos = _np.matmul(self._rotmatrix, pos.T, order='F').T
        super().__init__(E, len(pos))

    def _rotate(self, rotmatrix):
        self._pos = _np.matmul(rotmatrix, self._pos.T, order='F').T

    def updatePos(self):
        if self.rndOrientation:
            self._rotmatrix = _random_rotation(rng=self.rng)
            self._rotate(self._rotmatrix)


class crystal(simobj):
    """
    a crystalline structure
    """

    def __init__(self, E, lconst, langle, unitcell, N, repeats=None, rotangles=(0, 0, 0), fwhm=None, rho=2):
        if fwhm is not None:
            cosa, cosb, cosc = _np.cos(_np.array(langle))
            sina, sinb, sinc = _np.sin(_np.array(langle))
            neededrepeats = _np.ceil(
                2 * _np.array(fwhm) / (_np.array([1, sinc, _np.sqrt(sinb ** 2 - ((cosa - cosb * cosc) / sinc) ** 2)]) * _np.array(lconst))
            ).astype(int)
            if repeats is not None:
                if _np.any(neededrepeats > repeats):
                    _warn('Number of repeats small for choosen fwhm')
            else:
                repeats = neededrepeats
        if repeats is None:
            repeats = 3 * [int(_np.rint((N / len(unitcell)) ** (1 / 3.0)))]
        if _np.prod(repeats) * len(unitcell) < N:
            _warn('Number of atoms high for atoms in focus')
        allpos = crystal._lattice(lconst, langle, unitcell, repeats, rng=self.rng)

        if _np.any(rotangles):
            self._rotmatrix = _rotation(*rotangles)
            allpos = _np.matmul(self._rotmatrix, allpos.T, order='F').T
        else:
            self._rotmatrix = False

        self._allpos = allpos
        self.rndOrientation = False
        self._p = None
        self.fwhm = None if fwhm is None else _np.array(fwhm)
        self.rho = _np.array(2) if rho is None else _np.array(rho)

        super().__init__(E, N)
        self._resetproperties = ['rho', 'fwhm']

    @staticmethod
    def _lattice(lconst, langle, unitcell, repeats, sigma=0, rng=None):
        if rng is None:
            rng = _np.random.default_rng(_np.random.randint(2 ** 63))
        cosa, cosb, cosc = _np.cos(_np.array(langle))
        sina, sinb, sinc = _np.sin(_np.array(langle))
        basis = _np.array(
            [[1, 0, 0], [cosc, sinc, 0], [cosb, (cosa - cosb * cosc) / sinc, _np.sqrt(sinb ** 2 - ((cosa - cosb * cosc) / sinc) ** 2)]]
        ) * _np.expand_dims(lconst, 1)

        atoms = _np.dot(unitcell, basis).astype(_np.float32)
        if sigma != 0:
            atoms += rng.uniform(0, sigma, atoms.shape).astype(_np.float32)

        for j in range(3):
            atoms = _np.concatenate([atoms + (basis[j] * k).astype(_np.float32)[_np.newaxis, :] for k in range(repeats[j])])
            if sigma != 0:
                atoms += rng.uniform(0, sigma, atoms.shape).astype(_np.float32)

        return atoms - _np.max(atoms, axis=0) / 2.0

    def updatePos(self):
        pos = self._allpos

        if _np.any(self.rndOrientation):
            if _np.size(self.rndOrientation) == 3:
                rotmatrix = _axisrotation(_np.array(self.rndOrientation, dtype=_np.float64), self.rng.uniform(0, 2 * pi))
            else:
                rotmatrix = _random_rotation(self.rng)
            if self._rotmatrix is not False:
                self._rotmatrix = rotmatrix @ self._rotmatrix
            else:
                self._rotmatrix = rotmatrix
            pos = _np.matmul(self._rotmatrix, pos.T, order='F').T
            self._p = None

        if self.fwhm is not None and self._p is None:
            p = _np.exp(
                _ne.evaluate('sum(-pos**rho*s, axis=1)', local_dict={'s': _np.log(2) * (2 / self.fwhm) ** self.rho, 'pos': pos, 'rho': self.rho})
            )
            p /= _np.sum(p)
            self._p = p
        else:
            p = self._p
        missing = self.N
        idx = []

        while missing:
            new = self.rng.choice(len(self._allpos), min(missing, len(self._allpos)), replace=False, p=p)
            idx.append(new)
            missing -= len(new)
        idx = _np.concatenate(idx)
        self._pos = pos[idx]

    def __setattr__(self, name, value):
        # reset cached values if property changes
        if any(name == n for n in self._resetproperties):
            self._p = None
        super().__setattr__(name, value)

    def _rotate(self, rotmatrix):
        self._allpos = _np.matmul(rotmatrix, self._allpos.T, order='F').T


class sc(crystal):
    """
    a sc crystal
    """

    def __init__(self, E, N, a, repeats=None, rotangles=(0, 0, 0), fwhm=None, rho=None):
        lconst = [a, a, a]
        unitcell = [[0, 0, 0]]
        langle = _np.array([90, 90, 90]) * pi / 180.0
        super().__init__(E, lconst, langle, unitcell, N, repeats, rotangles, fwhm, rho)


class fcc(crystal):
    """
    a fcc crystal
    """

    def __init__(self, E, N, a, repeats=None, rotangles=(0, 0, 0), fwhm=None, rho=None):
        lconst = [a, a, a]
        unitcell = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        langle = _np.array([90, 90, 90]) * pi / 180.0
        super().__init__(E, lconst, langle, unitcell, N, repeats, rotangles, fwhm, rho)


class hcp(crystal):
    """
    a hcp crystal
    """

    def __init__(self, E, N, a, repeats=None, rotangles=(0, 0, 0), fwhm=None, rho=None):
        lconst = [a, a, 1.633 * a]
        unitcell = [[0, 0, 0], [1.0 / 3, 2.0 / 3, 1.0 / 2]]
        langle = _np.array([90, 90, 120]) * pi / 180.0
        super().__init__(E, lconst, langle, unitcell, N, repeats, rotangles, fwhm, rho)


class cuso4(crystal):
    """
    cuso4 crsystal with fixed lattice constant
    """

    def __init__(self, N, E, repeats=None, rotangles=(0, 0, 0), fwhm=None, rho=None):
        # https://doi.org/10.1524%2Fzkri.1975.141.5-6.330
        unitcell = [[0, 0, 0], [0.5, 0.5, 0]]
        lconst = _np.array([6.141, 10.736, 5.986]) * 1e-4
        langle = _np.array([82.27, 107.43, 102.67]) * pi / 180.0
        super().__init__(E, lconst, langle, unitcell, N, repeats, rotangles, fwhm, rho)


class grating(simobj):
    def __init__(self, E, N, linewidth, spacingwidth, fwhm, rho=2, rholine=10, rotangles=(0, 0, 0)):
        self.linewidth = linewidth
        self.spacingwidth = spacingwidth
        self.fwhm = fwhm
        self.rho = rho
        self.rholine = rholine
        self.__resetproperties = ['linewidth', 'spacingwidth', 'fwhm', 'rho', 'rholine']
        if _np.any(rotangles):
            self._rotmatrix = _rotation(*rotangles)
        else:
            self._rotmatrix = False
        super().__init__(E, int(N))

    def updatePos(self):
        def _lines(x, a, b, rho):
            s = _np.log(2) * (2 / a) ** rho  # noqa
            return _ne.evaluate('exp(-abs((x%(a+b))-(a+b)/2)**rho*s)')

        if self._pos is None:
            rhofocusx = self.rho if _np.isscalar(self.rho) else self.rho[0]
            fwhmx = self.fwhm if _np.isscalar(self.fwhm) else self.fwhm[0]
            self._x = _np.arange(-(0.5 + 2 / rhofocusx) * fwhmx, (0.5 + 2 / rhofocusx) * fwhmx, min(self.linewidth, self.spacingwidth) / 10)
            p = _lines(self._x, self.linewidth, self.spacingwidth, self.rholine) * _gnorm(self._x, fwhmx, rhofocusx)
            self._c = _np.cumsum(p)
            self._c = self._c / self._c[-1]
        rhofocusyz = self.rho if _np.isscalar(self.rho) else _np.array(self.rho[1:])
        fwhmyz = self.fwhm if _np.isscalar(self.fwhm) else _np.array(self.fwhm[1:])
        self._pos = _np.zeros((self.N, 3))
        self._pos[:, 0] = _np.interp(self.rng.uniform(size=self.N), self._c, self._x)
        self._pos[:, 1:] = _rndgennorm(0, fwhmyz, rhofocusyz, (self.N, 2), self.rng)

        if self._rotmatrix is not False:
            self._rotate(self._rotmatrix)

    def _rotate(self, rotmatrix):
        self._pos = _np.matmul(rotmatrix, self._pos.T, order='F').T
