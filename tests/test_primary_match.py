"""Unit tests of the primary matching on synthetic edges (numpy / scipy only)."""
import numpy as np
from scipy import ndimage

from xscat import primary as pm


def _disc(n=160, radius=45.0, inside=0.08):
    yy, xx = np.mgrid[:n, :n]
    obj = (yy - n / 2.0) ** 2 + (xx - n / 2.0) ** 2 < radius ** 2
    return np.where(obj, inside, 1.0).astype(np.float64), obj


def test_silhouette_band_hugs_the_edge():
    p, obj = _disc()
    band = pm.silhouette_band(np.where(obj, 10.0, 0.0), width_px=4)
    edge = obj & ~ndimage.binary_erosion(obj)
    assert band[edge].all()
    assert not band[ndimage.distance_transform_edt(~obj) > 6].any()
    assert 0 < band.sum() < 0.3 * band.size


def test_estimate_shift_recovers_a_subpixel_shift():
    rng = np.random.default_rng(0)
    p, obj = _disc()
    ref = ndimage.gaussian_filter(p, 1.2) + rng.normal(0, 0.004, p.shape)
    true = (0.7, -1.3)
    img = ndimage.shift(ndimage.gaussian_filter(p, 1.2), (-true[0], -true[1]), order=3, mode='nearest') + rng.normal(0, 0.004, p.shape)
    dr, dc, peak = pm.estimate_shift(ref, img, pm.silhouette_band(np.where(obj, 10.0, 0.0), 6), max_shift_px=5)
    assert abs(dr - true[0]) < 0.15 and abs(dc - true[1]) < 0.15 and peak > 0.9


def test_estimate_shift_ignores_level_differences():
    p, obj = _disc()
    ref = ndimage.gaussian_filter(p, 1.0)
    img = ndimage.gaussian_filter(np.where(obj, 0.2, 0.95), 1.0)
    dr, dc, _ = pm.estimate_shift(ref, img, pm.silhouette_band(np.where(obj, 10.0, 0.0), 6), max_shift_px=4)
    assert abs(dr) < 0.1 and abs(dc) < 0.1


def test_fit_psf_sigma_recovers_the_blur():
    rng = np.random.default_rng(1)
    p, obj = _disc()
    s = 0.02 * ndimage.gaussian_filter(1.0 - p, 12.0) + 0.01
    band = pm.silhouette_band(np.where(obj, 10.0, 0.0), 8)
    for true in (0.8, 1.8, 3.0):
        meas = ndimage.gaussian_filter(p, true) + s + rng.normal(0, 0.003, p.shape)
        sigma, costs, rms0, rms1 = pm.fit_psf_sigma(meas, p, s, band)
        assert abs(sigma - true) < 0.2 and rms1 < 0.5 * rms0


def test_solve_rigid_shift_recovers_the_offset():
    rng = np.random.default_rng(3)
    delta_true = np.array([0.31, -0.22, 0.12])
    thetas = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    jac = np.zeros((12, 2, 3))
    for v, th in enumerate(thetas):
        jac[v, 0, 2] = 2.5
        jac[v, 1, 0] = -2.5 * np.sin(th)
        jac[v, 1, 1] = 2.5 * np.cos(th)
    shifts = np.einsum('vij,j->vi', jac, delta_true) + rng.normal(0, 0.03, (12, 2))
    shifts[4] = np.nan
    delta, res, used = pm.solve_rigid_shift(shifts, jac)
    assert used == 11 and np.allclose(delta, delta_true, atol=0.03)


def test_open_beam_noise_estimate():
    rng = np.random.default_rng(5)
    m = np.ones((4, 200, 200), dtype=np.float32)
    m[:, 60:140, 60:140] = 0.3
    m += rng.normal(0, 1, m.shape).astype(np.float32) * 0.008 * np.sqrt(m)
    assert abs(pm.open_beam_noise(m) - 0.008) < 0.0012
