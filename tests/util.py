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

        self.assertAlmostEqual(abs2(1j + 1), 2)
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
    rndgennorm
    rndstr
    gnorm
    '''

    def test_rndSphere(self):
        from idi.util import rndSphere

        R = 10
        N = int(1e6)
        t = rndSphere(R, N)
        r = np.linalg.norm(t, axis=1)
        self.assertTupleEqual(t.shape, (N, 3))
        self.assertGreater(r.max(), R * 0.95)
        self.assertLessEqual(r.max(), R)
        hist = np.histogram(r, bins=np.linspace(0, 1, int(N ** (1 / 3) * 0.2)) ** (1 / 3) * R)[0]
        self.assertLess(np.ptp(hist) / np.mean(hist), 0.05, msg='r deviation higher than 5%')
        d, b = np.histogramdd(t, density=True, bins=11)
        br = np.linalg.norm(np.meshgrid(*((c[:-1] + c[1:]) / 2 for c in b)), axis=0)
        f = d[br < 0.9 * R]
        self.assertLess(f.std() / f.mean(), 0.05, msg='density deviation higher than 5%')

    def test_rndstr(self):
        from idi.util import rndstr

        self.assertEqual(len(rndstr(0)), 0)
        self.assertEqual(len(rndstr(1)), 1)
        self.assertEqual(len(rndstr(1000)), 1000)
        s = set()
        for i in range(100):
            c = rndstr(5)
            s.add(c)
            self.assertTrue(c.isascii())
        self.assertEqual(len(s), 100, msg='repeats found')


class h5util(unittest.TestCase):
    def setUp(self):
        import h5py

        self.f1 = h5py.File(name='f1', driver='core', backing_store=False, mode='w')

    def tearDown(self):
        self.f1.close()

    def test_append(self):
        import idi.util.h5util as h5util

        h5util.appenddata(self.f1, 't1', np.zeros((1, 10, 10)))
        t = self.f1['t1']
        self.assertTupleEqual(t.shape, (1, 10, 10))
        self.assertRaises(KeyError, h5util.appenddata, t, 'blub', np.zeros((3, 10, 10)))
        h5util.appenddata(t, '/', np.zeros((3, 10, 10)))
        self.assertTupleEqual(t.shape, (4, 10, 10))
        self.assertEqual(t.compression, 'lzf')
        self.assertEqual(t.shuffle, True)
        self.assertTupleEqual(t.chunks, (1, 10, 10))

        h5util.appenddata(self.f1, 'string', 's1')
        h5util.appenddata(self.f1, 'string', 's2_ismuchlonger')
        h5util.appenddata(
            self.f1, 'string', np.array(self.f1['string']),
        )
        t = self.f1['string']
        self.assertEqual(len(t), 4)
        self.assertEqual(t[-1], b's2_ismuchlonger')
        self.assertEqual(np.array(t)[-2], b's1')


class funchelper(unittest.TestCase):
    def test_asgen(self):
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

    def test_list2array(self):
        from idi.util import list2array

        self.assertRaises(TypeError, list2array, 1)

        l = [1, 2, 3]
        testing.assert_array_equal(list2array(l), np.array([1, 2, 3]))

        l = [[1], [1, 2], [1, 2, 3], [1]]
        a = list2array(l)
        self.assertEqual(a[0, 0], l[0][0])
        self.assertEqual(a[1, 1], l[1][1])
        self.assertEqual(a[2, 2], l[2][2])
        self.assertEqual(np.sum(a), 11)
        self.assertEqual(len(a), len(l))

        l = [[1.0], ['a', 'b'], ['a', 'b', 'c'], ['a']]
        a = list2array(l)
        self.assertEqual(float(a[0, 0]), l[0][0])
        self.assertEqual(a[1, 1], l[1][1])
        self.assertEqual(a[2, 2], l[2][2])
        self.assertEqual(len(a), len(l))

    def test_fill(self):
        from idi.util import fill

        t = np.arange(20).reshape(4, 5).astype(float)
        t[1:4, 1:4] = 1
        t[2, 2] = np.nan
        t[0, 0] = np.nan
        t[-1, 0] = np.nan
        f = fill(t)
        good = np.copy(t)
        good[2, 2] = 1
        good[0, 0] = 5
        good[-1, 0] = 10
        self.assertFalse(np.any(np.isnan(f)))
        testing.assert_array_equal(good, f)

        t = np.empty((3, 3))
        t[:] = np.nan
        f = fill(t)
        self.assertTrue(np.all(np.isnan(f)))

        t = np.empty((3, 3))
        t[:] = np.nan
        t[1, 1] = 1
        f = fill(t)
        testing.assert_array_equal(f, 1)

    def test_atleastnd(self):
        from idi.util import atleastnd

        self.assertTupleEqual(atleastnd(0, 2).shape, (1, 1))
        self.assertTupleEqual(atleastnd(np.zeros((2, 2)), 1).shape, (2, 2))
        self.assertTupleEqual(atleastnd(np.zeros((3, 2)), 4).shape, (1, 1, 3, 2))

    def test_rebin(self):
        from idi.util import rebin

        t = np.arange(20).reshape(4, 5).astype(float)
        testing.assert_array_equal(rebin(t, (1, 1), 'sum'), t)
        testing.assert_array_equal(rebin(t, (2, 2), 'sum'), np.array([[12.0, 20.0], [52.0, 60.0]]))
        self.assertEqual(rebin(t, 4, 'sum'), t[:, :4].sum())
        self.assertEqual(rebin(t, 3, 'max'), t[:3, :3].max())
        self.assertEqual(rebin(t, 5, 'max'), t.max())

        t = np.ones((4, 4), bool)
        self.assertEqual(rebin(t, 2, 'min').dtype, bool)
        testing.assert_array_equal(rebin(t, 2,), np.ones((2, 2)))

        t = np.ones((20, 2, 2))
        testing.assert_array_equal(rebin(t, (10, 20, 1), 'mean'), np.ones((2, 1, 2)))

        self.assertRaises(ValueError, rebin, t, 1, 'blub')
        self.assertRaises(ValueError, rebin, t, (1, 1), 'sum')

    '''
    TODO:
    
    create_mask
    diffdist
    
    fftfilter_mean
    fftfilter_std

    filter_std
    arrayfromiter
    split
    '''


if __name__ == '__main__':
    unittest.main()
