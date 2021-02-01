from __future__ import division as _future_div, print_function as _future_printf
import numpy as _np
import numexpr as _ne
from numpy import pi
import numba as _numba
import math as _math
import random as _random
from ..util import poisson_disc_sample as _pds, rndgennorm as _rndgennorm, fastlen as _fastlen


'''
simulation objects
'''


class atoms:
    '''
    baseclass - a bunch of atoms
    '''

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
    
    @property
    def k(self):
        return 2 * pi / (1.24 / self._E)

    def get(self): 
        if self.rndPhase:
            phase = _np.random.rand(self._N)*(2*pi)
        else:
            phase=_np.zeros(self._N)
                        
        ret = _np.concatenate((self._pos, phase[:, _np.newaxis]), axis=1)
        return ret
    
    def getImg(self, dx, ndim=2):  
        pos = self.get()
        ind = _np.rint((pos[:, :ndim] - _np.min(pos[:, :ndim], axis=0, keepdims=True)) / dx).astype(int)
        s = _np.array(ind.max(0)) + 1
        pads = _np.array([_fastlen(i) for i in s])
        ind += (pads - s) // 2
        img = _np.zeros(pads, _np.complex128)
        _np.add.at(img, tuple(ind.T), _np.exp(1j * pos[:, -1]))
        return img

class sphere(atoms):
    '''
    a sphere with random positions inside
    '''

    def __init__(self, E, N, r):
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


class gnorm(atoms):
    '''
    a generalised normal shaped volume with random positions inside
    '''

    def __init__(self, E, N, fwhm, rho=2):
        self._fwhm = _np.array(fwhm)
        self._rho = _np.array(rho)
        pos = self._rndGNorm(self._fwhm, self._rho, N)
        atoms.__init__(self, E, pos)
        self.rndPos = True

    @staticmethod
    def _rndGNorm(fwhm, rho, N):
        if _np.all(rho==2):
            return _np.random.randn(int(N),3)*(fwhm/2.355)
        return _rndgennorm(0, fwhm, rho, (int(N),3))

    def get(self):
        if self.rndPos:
            self._pos = self._rndGNorm(self._fwhm, self._rho, self.N)
        return atoms.get(self)

    
class gauss(gnorm):
    '''
    a gaussian shaped volume with random positions inside
    '''
    def __init__(self, E, N, fwhm):
        gnorm.__init__(self, E, N, fwhm, 2)
        

class multisphere(atoms):
    '''
    multiple, randomly positioned spheres
    '''

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
        Natoms: total number of excited atoms, if negative: mean number per sphere
        rsphere: radius of each spehre
        fwhmfocal: the number of atoms per sphere scales gaussian with the distance from the center. this sets the fhwm of the gaussian.
        spacing: thickness of atom-free layer around each sphere
        Nspheres: max. total number of particles in volume
        """
       
        self._Nspheres = Nspheres
        self.rsphere = rsphere
        self.fwhmfocal = fwhmfocal
        self.spacing = spacing
        self.rndPos = True
        self._debug = None
        self._N = int(-Natoms*_np.mean([len(self._spherepos()) for i in range(10)])) if Natoms<0 else Natoms
        atoms.__init__(self, E, self._atompos())

    def _spherepos(self):
        return _pds(1.2 * self.fwhmfocal, 2 * (self.rsphere + self.spacing), ndim=3, N=self._Nspheres)

    def _atompos(self):
        """
        calculate position of atoms
        """
        posspheres = self._spherepos()
        r2 = (_np.sum(posspheres ** 2, axis=1))
        p = _np.exp(-r2*(1/(0.4 * self.fwhmfocal))**2 / 2.0)
        n = _np.random.poisson(p / _np.sum(p) * self._N)
        missing = self._N - _np.sum(n)
        while missing:
            ids=_np.random.choice(len(n),int(abs(missing)),replace=True,p=p/(_np.sum(p)))
            _np.add.at(n,ids,_np.sign(missing))
            n[n<0]=0
            missing = self._N - _np.sum(n)
        nc = _np.cumsum(n)
        if self.rsphere>0:
            posatoms = sphere._rndSphere(self.rsphere, int(self._N))
        else:
            posatoms = _np.zeros((self._N,3))
        multisphere._staggeredadd(posspheres, posatoms, nc)
        self._debug = (len(posspheres), _np.min(n), _np.max(n), _np.mean(n))
        return posatoms

    def get(self):
        if self.rndPos or self._pos is None:
            self._pos = self._atompos()
        return atoms.get(self)

    def __setattr__(self, name, value):
        # reset cached values if property changes
        if any(name == n for n in ['rndPos', 'rsphere', 'fwhmfocal', 'spacing', '_Nspheres']):
            self._pos = None
        super(multisphere, self).__setattr__(name, value)


class hcpsphere(multisphere):
    '''
    densly hcp packed spheres
    '''

    def __init__(self, E, Natoms=1e6, rsphere=10, fwhmfocal=200, a=20, rotangles=None, sigma=0):
        self.Nhcp = None
        self.rndPos = False
        self.rndOrientation = False
        self._N, self.rsphere, self.fwhmfocal, self.a, self.sigma, self.rsphere = Natoms, rsphere, fwhmfocal, a, sigma, rsphere
        self.rotangles = rotangles
        atoms.__init__(self, E, self._atompos())

    def __setattr__(self, name, value):
        # reset cached values if property changes
        if any(name == n for n in ['rndPos', 'rndOrientation', 'rsphere', 'fwhmfocal', 'a', 'sigma', 'rotangles']):
            self._hcp = None
            self._pos = None
        if name == 'rotangles':
            self._rotmatrix = None if value is None else crystal._rotation(*value)
        super(hcpsphere, self).__setattr__(name, value)

    def _spherepos(self):
        if (self.sigma and self.rndPos) or self._hcp is None:
            Nhcp = self.Nhcp or _np.ceil(5 * self.fwhmfocal / (self.a * _np.array([2.0, 0.86, 1.63]))).astype(int)
            lconst = [self.a, self.a, 1.633 * self.a]
            unitcell = [[0, 0, 0], [1.0 / 3, 2.0 / 3, 1.0 / 2]]
            langle = _np.array([90, 90, 120]) * pi / 180.0
            hcp = crystal._lattice(lconst, langle, unitcell, Nhcp, self.sigma)
            hcp -= _np.mean(hcp, axis=0)
            self._hcp = hcp
        if self.rndOrientation:
            self._rotmatrix = crystal._random_rotation()
        if self._rotmatrix:
            return _np.matmul(self._rotmatrix, self._hcp.T, order='F').T
        else:
            return self._hcp


class xyz(atoms):
    '''
    atom positions specified by an xyz file
    '''

    def __init__(self, E, filename, atomname, rotangles, scale=1e-4):
        import re

        with open(filename, 'r') as file:
            data = file.read()
        lines = re.findall("^" + atomname + "\d*\s*[\d,\.]+\s+[\d,\.]+\s+[\d,\.]+", data, re.IGNORECASE | re.MULTILINE)
        pos = _np.genfromtxt(lines)[:, 1:] * scale
        if _np.any(rotangles):
            self._rotmatrix = crystal._rotation(*rotangles)
            pos = _np.matmul(self._rotmatrix, pos.T, order='F').T
        else:
            self._rotmatrix = None
        pos = pos - (_np.max(pos, axis=0) / 2.0)
        atoms.__init__(self, E, pos)


class crystal(atoms):
    '''
    a crystalline structure
    '''

    def __init__(self, E, lconst, langle, unitcell, N, repeats = None, rotangles = [0,0,0], fwhm = None):
        if repeats is None :
            repeats = 3 * [int(_np.rint((N / len(unitcell)) ** (1 / 3.0)))]
        pos = crystal._lattice(lconst, langle, unitcell, repeats)
        if _np.any(rotangles):
            self._rotmatrix = crystal._rotation(*rotangles)
            pos = _np.matmul(self._rotmatrix, pos.T, order='F').T
        else:
            self._rotmatrix = None
        atoms.__init__(self, E, pos)
        self._allpos = pos
        self._pos = None
        self.rndOrientation = False
        self._N = int(N)
        self._fwhm = fwhm
        self._p = None
        
    @staticmethod
    def _lattice(lconst, langle, unitcell, repeats, sigma=0):
        cosa, cosb, cosc = _np.cos(_np.array(langle))
        sina, sinb, sinc = _np.sin(_np.array(langle))
        basis = _np.array([[1, 0, 0], [cosc, sinc, 0], [cosb, (cosa - cosb * cosc) / sinc, _np.sqrt(sinb ** 2 - ((cosa - cosb * cosc) / sinc) ** 2)],]) * _np.expand_dims(lconst, 1)
        
        atoms = _np.dot(unitcell, basis).astype(_np.float32)
        if sigma!=0: atoms += (sigma * _np.random.rand(*atoms.shape)).astype(_np.float32)

        for j in range(3):
            atoms = _np.concatenate([atoms + (basis[j] * k).astype(_np.float32)[_np.newaxis, :] for k in range(repeats[j])])
            if sigma!=0: atoms += sigma * _np.random.rand(*atoms.shape).astype(_np.float32)

        return atoms - _np.max(atoms, axis=0) / 2.0

    @staticmethod
    @_numba.njit
    def _rotation(alpha, beta, gamma):
        cosa, cosb, cosg = _np.cos(_np.array((alpha, beta, gamma)))
        sina, sinb, sing = _np.sin(_np.array((alpha, beta, gamma)))

        # yaw pitch roll
        M = _np.array(
            [
                [cosb * cosg, sina * sinb * cosg - cosa * sing, cosa * sinb * cosg + sina * sing,],
                [cosb * sing, sina * sinb * sing + cosa * cosg, cosa * sinb * sing - sina * cosg,],
                [-sinb, sina * cosb, cosa * cosb],
            ]
        )
        return M

    @staticmethod
    @_numba.njit
    def _random_rotation(amount=1):
        # https://doc.lagout.org/Others/Game%20Development/Programming/Graphics%20Gems%203.pdf
        theta, phi, z = _np.random.rand(3) * _np.array((2.0 * pi * amount, 2.0 * pi, amount))
        V = (_np.cos(phi) * _np.sqrt(z), _np.sin(phi) * _np.sqrt(z), _np.sqrt(1.0 - z))
        sint, cost = _np.sin(theta), _np.cos(theta)
        R = _np.array(((cost, sint, 0), (-sint, cost, 0), (0, 0, 1)))
        M = _np.dot(2 * _np.outer(V, V) - _np.eye(3), R)
        return M
    
    @staticmethod
    @_numba.njit
    def _axisrotation(axis, theta):
        u = axis / _np.linalg.norm(axis.astype(_np.float64))
        return (
            _np.cos(theta) * _np.identity(3)
            + _np.sin(theta) * _np.cross(_np.identity(3), u)
            + (1 - _np.cos(theta)) * _np.outer(u, u)
        )


    def get(self):
        pos = self._allpos
        if _np.any(self.rndOrientation):
            if _np.size(self.rndOrientation) == 3:
                rotmatrix = crystal._axisrotation(_np.array(self.rndOrientation, dtype=_np.float64), _random.random() * 2 * pi)
            else:
                rotmatrix = crystal._random_rotation()
            pos = _np.matmul(rotmatrix, pos.T, order='F').T
            self.p = None   
        if self._fwhm is not None and self._p is None:
            p= _np.exp(_ne.evaluate('sum(-pos**2 / (2*sigma**2), axis=1)',
                                    local_dict={'sigma': self._fwhm/2.35, 'pos': pos}
                                   )
                      )
            p /= _np.sum(p)
            if not _np.any(self.rndOrientation): self._p = p
        else:
            p = self._p       
        missing = self.N
        idx = []
        r = _np.random.default_rng(_np.random.randint(2**31)) #to be able to use np.seed() to seed this as well
        while missing:
            new = r.choice(len(self._allpos), min(missing, len(self._allpos)), replace=False, p=p)
            idx.append(new)
            missing -= len(new)
        idx = _np.concatenate(idx)
        self._pos = pos[idx]
        return atoms.get(self)



class sc(crystal):
    '''
    a sc crystal
    '''

    def __init__(self, E, N, a, repeats=None, rotangles=[0,0,0], fwhm=None):
        lconst = [a, a, a]
        unitcell = [[0, 0, 0]]
        langle = _np.array([90, 90, 90]) * pi / 180.0
        crystal.__init__(self, E, lconst, langle, unitcell, N, repeats, rotangles, fwhm)


class fcc(crystal):
    '''
    a fcc crystal
    '''

    def __init__(self, E, N, a, repeats=None, rotangles=[0,0,0], fwhm=None):
        lconst = [a, a, a]
        unitcell = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        langle = _np.array([90, 90, 90]) * pi / 180.0
        crystal.__init__(self, E, lconst, langle, unitcell, N, repeats, rotangles, fwhm)


class hcp(crystal):
    '''
    a hcp crystal
    '''

    def __init__(self, E, N, a, repeats=None, rotangles=[0,0,0], fwhm=None):
        lconst = [a, a, 1.633 * a]
        unitcell = [[0, 0, 0], [1.0 / 3, 2.0 / 3, 1.0 / 2]]
        langle = _np.array([90, 90, 120]) * pi / 180.0
        crystal.__init__(self, E, lconst, langle, unitcell, N, repeats, rotangles, fwhm)


class cuso4(crystal):
    '''
    cuso4 crsystal with fixed lattice constant
    '''

    def __init__(self, N, E, repeats=None, rotangles=[0,0,0], fwhm=None):
        # https://doi.org/10.1524%2Fzkri.1975.141.5-6.330
        unitcell = [[0, 0, 0], [0.5, 0.5, 0]]
        lconst = _np.array([6.141, 10.736, 5.986]) * 1e-4
        langle = _np.array([82.27, 107.43, 102.67]) * pi / 180.0
        crystal.__init__(self, E, lconst, langle, unitcell, N, repeats, rotangles, fwhm)
