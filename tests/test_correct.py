import numpy as np

from xscat import correct as co
from xscat import scatter as sc


def test_ratio_gain_and_corrections():
    p = np.array([[1.0, 0.5, 0.01]], dtype=np.float32)
    s = np.array([[0.02, 0.02, 0.02]], dtype=np.float32)
    g = sc.ratio_gain(p, s)
    assert np.allclose(g, p / (p + s))
    assert sc.ratio_gain(p, s, min_gain=0.5).min() >= 0.5
    m = p + s
    t, g2 = co.ratio_fullp(m, p, s)
    assert np.allclose(t, p, atol=1e-6) and np.allclose(g2, g)
    t_leap, _ = co.ratio_leap(m, g)
    assert np.allclose(t_leap, t)


def test_bad_rows_and_replacement():
    bright = np.full(10, 1000.0)
    bright[[3, 7]] = 100.0
    bad = co.bad_rows_from_bright(bright)
    assert list(bad) == [3, 7]
    stack = np.arange(2 * 10 * 4, dtype=np.float32).reshape(2, 10, 4)
    mapping = co.replace_rows(stack, bad)
    assert mapping == {3: 2, 7: 6} or mapping == {3: 4, 7: 8} or set(mapping) == {3, 7}
    assert np.array_equal(stack[:, 3], stack[:, mapping[3]])


def test_open_beam_mask_and_downsample_factors():
    m = np.ones((40, 40), dtype=np.float32)
    m[10:30, 10:30] = 0.2
    ob = co.open_beam_mask(m, 0.9, 2)
    assert ob[0, 0] and not ob[20, 20] and not ob[9, 20]
    assert sc.detector_downsample_factor(1500, 1900, 204) == 10
    assert sc.volume_downsample_factor((476, 424, 424), 199) == 3
