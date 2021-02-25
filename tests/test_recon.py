import unittest
import numpy as np
import numpy.testing as testing
import numba.cuda


class basic(unittest.TestCase):
    def test_import(self):
        import idi.reconstruction


class ft_test(unittest.TestCase):
    def test_correlator(self):
        from idi.reconstruction import ft

        with ft.correlator(np.zeros((32, 32), bool), 1000) as correlator:
            t = np.ones((32, 32))
            c = correlator.unwrap(correlator.corr(t))
            self.assertTupleEqual(c.shape, (1, 64, 64))
            self.assertTupleEqual(np.unravel_index(np.argmax(c), c.shape), (0, 32, 32))
            self.assertAlmostEqual(c.max(), 32 * 32)

        with ft.correlator(np.zeros((32, 16), bool), 100) as correlator:
            t = np.ones((32, 16))
            c = correlator.unwrap(correlator.corr(t))
            self.assertTupleEqual(c.shape[1:], (32, 64))
            self.assertTupleEqual(np.unravel_index(np.argmax(c), c.shape), (0, 16, 32))

        mask = np.zeros((17, 27), bool)
        mask[5:10] = True
        with ft.correlator(mask, 64) as correlator:
            t = np.ones((17, 27))
            c1 = correlator.unwrap(correlator.corr(t))
            t[mask] = 100
            c2 = correlator.unwrap(correlator.corr(t))
            testing.assert_allclose(c1, c2)

    def test_tiles_correlator(self):
        from idi.reconstruction import ft
        import itertools

        def _qs(shape, z, offset):
            y, x = np.meshgrid(np.arange(shape[1], dtype=np.float64), np.arange(shape[0], dtype=np.float64))
            x -= offset[0]
            y -= offset[1]
            d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            return np.array([(k / d * z) for k in (z, y, x)])

        z = 16
        N0, N1 = 8, 8
        off = 1
        qs = np.stack([_qs((N0, N1), z, (i, j)).T for i, j in itertools.product([-off, N0 + off], [-off, N1 + off])]).reshape(4, -1, 3)
        mean = np.ones((4, N0 * N1))
        maskout = np.zeros((4, N0 * N1), bool)

        with ft.correlator_tiles(qs, maskout, mean) as correlator:
            correlator.add(mean)
            correlator.add(mean)
            c = correlator.result(True)
            self.assertTupleEqual(c.shape, (4, 32, 32))
            self.assertAlmostEqual(np.nanmax(c), 1)
            self.assertAlmostEqual(np.nanmin(c), 1)
            self.assertGreater(np.nansum(c), N0 * N1)

    def test_corr(self):
        from idi.reconstruction import ft

        c = ft.corr(np.ones((17, 8)), 16)
        self.assertTupleEqual(c.shape[1:], (16, 32))

    def test_unwrap(self):
        from idi.reconstruction import ft

        pass


@unittest.skipIf(not numba.cuda.is_available(), 'no cuda available')
class gpu_test(unittest.TestCase):
    def test_cucor(self):
        from idi.reconstruction import cucor

        t = np.ones((32, 32))
        with cucor.corrfunction(t.shape, 1000, 32) as f:
            c = f(t)
            self.assertTupleEqual(c.shape[1:], (64, 64))
            self.assertGreater(len(c), 1)
            self.assertEqual(c.max(), t.sum())
            maxid = np.unravel_index(np.argmax(c), c.shape)
            self.assertTupleEqual(maxid, tuple((i / 2 for i in c.shape)))

    def test_cucorrrad(self):
        from idi.reconstruction import cucorrad

        t = np.ones((32, 32))
        with cucorrad.corrfunction(t.shape, 100, 16) as f:
            c = f(t)
            self.assertEqual(len(c), 16)
            self.assertEqual(c.min(), t.sum())

    def test_cusimple(self):
        from idi.reconstruction import cusimple
        import scipy.signal as ss

        t = np.arange(32 * 16).reshape(32, 16).astype(float)
        c = cusimple.corr(t)
        g = ss.correlate(t, t, 'full', 'fft')
        testing.assert_allclose(c[1:, 1:], g, atol=1e-7)

        t = np.arange(15 * 33).reshape(15, 33).astype(float)
        c = cusimple.corr(t, norm=True)
        gc = ss.correlate(t / t.mean(), t / t.mean(), 'full', 'fft')
        gn = ss.correlate(np.ones_like(t), np.ones_like(t), 'full', 'fft')
        g = gc / gn
        g[gn < 0.9] = np.nan
        testing.assert_allclose(c[1:, 1:], g, atol=1e-7)


class cpu_test(unittest.TestCase):
    def test_cpucor(self):
        from idi.reconstruction import cpucor

        t = np.ones((32, 32))
        with cpucor.corrfunction(t.shape, 100, 16) as f:
            c = f(t)
            self.assertTupleEqual(c.shape[1:], (33, 33))
            self.assertGreater(len(c), 1)
            self.assertEqual(c.max(), t.sum())
            maxid = np.unravel_index(np.argmax(c), c.shape)
            self.assertTupleEqual(maxid, tuple((i // 2 for i in c.shape)))

    def test_cpucorrrad(self):
        from idi.reconstruction import cpucorrad

        t = np.ones((32, 32))
        with cpucorrad.corrfunction(t.shape, 100, 16) as f:
            c = f(t)
            self.assertEqual(len(c), 16)
            self.assertEqual(c.min(), t.sum())

    def test_cpusimple(self):
        from idi.reconstruction import cpusimple
        import scipy.signal as ss
        import scipy.fft

        t = np.arange(32 * 16).reshape(32, 16).astype(float)
        c = cpusimple.corr(t)
        g = ss.correlate(t, t, 'full', 'fft')
        testing.assert_allclose(c[1:, 1:], g, atol=1e-7)

        t = np.arange(15 * 33).reshape(15, 33).astype(float)
        c = cpusimple.corr(t, fftfunctions=[scipy.fft.rfftn, scipy.fft.irfftn], norm=True)
        gc = ss.correlate(t / t.mean(), t / t.mean(), 'full', 'fft')
        gn = ss.correlate(np.ones_like(t), np.ones_like(t), 'full', 'fft')
        g = gc / gn
        g[gn < 0.9] = np.nan
        testing.assert_allclose(c[1:, 1:], g, atol=1e-7)


class singleshotnorm_test(unittest.TestCase):
    def setUp(self):
        from idi.reconstruction import singleshotnorm

        mask = np.ones((64, 64), bool)
        mask[:10, :10] = 0
        self.correlator = singleshotnorm.correlator(mask)

    def test_corr(self):
        t = np.ones((64, 64))
        c = self.correlator.corr(t)
        excepted = np.ones((127, 127))
        excepted[:10, :10] = 0
        excepted[-10:, -10:] = 0
        testing.assert_allclose(c, excepted, atol=1e-12)

    def test_properties(self):
        mask = np.ones((64, 64), bool)
        mask[:10, :10] = 0
        testing.assert_array_equal(self.correlator.mask, mask)
        self.assertTupleEqual(self.correlator.shape_input, (64, 64))
        self.assertTupleEqual(self.correlator.shape_result, (127, 127))


if __name__ == '__main__':
    unittest.main()
