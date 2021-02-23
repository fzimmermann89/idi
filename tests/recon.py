import unittest
import numpy as np
import numpy.testing as testing


class basic(unittest.TestCase):
    def test_import(self):
        import idi.reconstruction


class ft_test(unittest.TestCase):
    def test_correlator(self):
        from idi.reconstruction import ft

        pass

    def test_tiles_correlator(self):
        from idi.reconstruction import ft

        pass

    def test_corr(self):
        from idi.reconstruction import ft

        pass

    def test_unwrap(self):
        from idi.reconstruction import ft

        pass


class cpu_test(unittest.TestCase):
    def test_cpucor(self):
        from idi.reconstruction import cpucor

    def test_cpucorrrad(self):
        from idi.reconstruction import cpucorrad

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
        g[gn < 1] = np.nan
        testing.assert_allclose(c[1:, 1:], g, atol=1e-7)


class singleshotnorm_test(unittest.TestCase):
    from idi.reconstruction import singleshotnorm

    def test_correlator(self):
        pass


if __name__ == '__main__':
    unittest.main()
