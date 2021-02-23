import unittest
import numpy as np
import numpy.testing as testing


class basic(unittest.TestCase):
    def test_import(self):
        import idi.simulation


class simobject(unittest.TestCase):
    def test_sphere(self):
        from idi.simulation import simobj

        o = simobj.sphere(E=1, N=1e3, r=10)
        o.rndPos = False
        o.rndPhase = False

        self.assertAlmostEqual(o.k, 2 * np.pi / 1.24)
        self.assertEqual(o.N, 1000)

        t = o.get()
        self.assertTupleEqual(t.shape, (1000, 4))
        pos0 = t[:, :3]
        phase0 = t[:, 3:]
        testing.assert_array_less(np.linalg.norm(pos0, axis=1), 10)
        testing.assert_allclose(phase0, 0)

        o.r = 5
        t = o.get()
        self.assertTupleEqual(t.shape, (1000, 4))
        pos1 = t[:, :3]
        r = np.linalg.norm(pos1, axis=1)
        self.assertAlmostEqual(r.max(), 5, delta=0.5)
        testing.assert_array_less(r, 5)

        o.rndPhase = True
        pos2, phase2 = o.get2()
        self.assertFalse(np.allclose(phase2, 0))
        self.assertAlmostEqual(phase2.mean(), np.pi, delta=0.2)
        testing.assert_allclose(pos1, pos2)

        o.rndPos = True
        pos3, phase3 = o.get2()
        self.assertFalse(np.allclose(pos2, pos3))
        self.assertFalse(np.allclose(phase2, phase3))

        with self.assertWarns(Warning):
            o.rotangles = [1, 1, 1]

    def test_sc(self):
        from idi.simulation import simobj

        o=simobj.sc(1,100,1)
        self.assertEqual(len(o.get()),100)

    def test_gauss(self):
        from idi.simulation import simobj

        o = simobj.gauss(1, 100, 1)
        self.assertEqual(len(o.get()), 100)

    def test_multisphere(self):
        from idi.simulation import simobj

        o = simobj.multisphere(1, 100, 1,Nspheres=10)
        self.assertEqual(len(o.get()), 100)

        o = simobj.multisphere(1, 100, 1, fwhm=10)
        self.assertEqual(len(o.get()), 100)

if __name__ == '__main__':
    unittest.main()
