import unittest

import numpy as np
import numpy.testing as testing


class basic(unittest.TestCase):
    def test_import(self):
        import idi.util

    def test_angles(self):
        from idi.util import angles, rotation

        testing.assert_allclose(angles(np.eye(3)), (0, 0, 0), atol=1e-15)
        testing.assert_allclose(rotation(0, 0, 0), np.eye(3), atol=1e-15)
        for case in [(np.pi, 0, 0), (0, np.pi, 0), (0, 0, np.pi), (1, 1, 1)]:
            rm = rotation(*case)
            testing.assert_allclose(rotation(*angles(rm)), rm, atol=1e-15, err_msg=f'{case}: rotation(*angles({rm})) != {rm}')

    def test_fastlen(self):
        from idi.util import fastlen

        for x in [1, 2, 10, 17, 101, 5000]:
            self.assertGreaterEqual(fastlen(x), x)
            self.assertGreaterEqual(fastlen(x, factors=[2]), x)
            self.assertGreaterEqual(fastlen(x, factors=[2, 3]), x)

        xs = np.array([1, 11, 5001])
        for x, f in zip(xs, fastlen(xs, factors=[2, 3])):
            self.assertGreaterEqual(f, x)
        for x, f in zip(xs, fastlen(xs)):
            self.assertGreaterEqual(f, x)

        self.assertEqual(19, fastlen(19, [2, 19]))
        self.assertEqual(24, fastlen(19, [2, 3]))
        self.assertEqual(32, fastlen(19, [2]))
        self.assertEqual(6048, fastlen(6001, factors=[2, 3, 5, 7, 11]))

    def test_radial_profile(self):
        from idi.util import radial_profile

        # basic
        t = np.ones((100, 100))
        r, s = radial_profile(t, calcStd=True)
        testing.assert_allclose(r, 1, atol=1e-15)
        testing.assert_allclose(s, 0, atol=1e-15)
        self.assertEqual(len(r), 72)
        self.assertEqual(True, True)
        # center
        t = np.zeros((10, 10))
        t[2:5, :3] = 1
        r = radial_profile(t, center=(3, 1))
        testing.assert_allclose(r, np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        # random
        t = np.random.default_rng(0).random((50, 50))
        r, s = radial_profile(t, calcStd=True)
        testing.assert_array_less(r, 1)
        testing.assert_array_less(0, r)
        testing.assert_array_less(0, s[1:])
        testing.assert_allclose(r.mean(), 0.5, atol=0.1)
        # nan
        t[t > 0.8] = np.nan
        t[~np.isnan(t)] = 1
        r = radial_profile(t)
        testing.assert_allclose(r, 1, atol=1e-15)
        self.assertFalse(np.any(np.isnan(r)))


    def test_abs(self):
        from idi.util import abs2, abs2c

        self.assertAlmostEquals(abs2(1j + 1), 2)
        t = np.array(1 - 1j, dtype=np.complex)
        self.assertAlmostEqual(abs2c(t), 2.0)
        self.assertFalse(np.iscomplex(abs2c(t)), 2.0)
        self.assertTrue(np.iscomplexobj(abs2c(t)), 2.0)

    def test_shortsci(self):
        from idi.util import shortsci

        for n, s in zip([1, 0.1, -0.1, 10, 1e17, -1e-17, 0.5, 1.5, -15], ['1e0', '1e-1', '-1e-1', '1e1', '1e17', '-1e-17', '5e-1', '2e0', '-2e1']):
            self.assertEqual(shortsci(n), s)

        for n, s in zip([1, 0.555, 1.51, -1.234e17], ['1.0e0', '5.6e-1', '1.5e0', '-1.2e17',]):
            self.assertEqual(shortsci(n, decimals=1), s)


class random(unittest.TestCase):
    '''
    TODO:
    random_rotation
    poisson_disc_sample
    rndConvexpoly
    rndIcosahedron
    rndSphere
    rndgennorm
    rndstr
    gnorm
    '''


def test_something(self):
    self.assertEqual(True, True)


class h5util(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


class funchelper(unittest.TestCase):
    '''
    TODO:
    asgen
    aslengen
    aslist
    '''


def test_something(self):
    self.assertEqual(True, True)


class array(unittest.TestCase):
    def test_cutnan(self):
        from idi.util import cutnan

        t = np.zeros((10, 20))
        t[:5, :] = np.nan
        t[-5, :] = np.nan
        t[:, :2] = np.nan
        t[:, 15:17] = np.nan
        self.assertEqual(cutnan(t).shape, (4, 16))

        t = np.zeros((1, 20))
        t[:] = np.nan
        self.assertEqual(cutnan(t).shape, (0, 0))

    def test_centeredpart(self):
        from idi.util import centered_part

        t = np.arange(20).reshape(4, 5)
        testing.assert_array_equal(centered_part(t, (2, 2)), np.array([[6, 7], [11, 12]]))

    '''
    TODO:
    bin
    create_mask
    diffdist
    fastlen
    fftfilter_mean
    fftfilter_std
    fill
    filter_std
    arrayfromiter
    atleastnd
    split
    '''


def test_something(self):
    self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
