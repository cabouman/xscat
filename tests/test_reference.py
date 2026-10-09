import numpy as np

from xscat import reference as rf


def _volume(n=48, mu=0.2, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[:n, :n, :n]
    solid = ((y - n / 2) ** 2 + (x - n / 2) ** 2 < (0.3 * n) ** 2) & (z > 6) & (z < n - 6)
    solid &= ~(((y - n / 2) ** 2 + (x - n / 2) ** 2) < (0.1 * n) ** 2)      # a hole that must stay open
    vol = mu * solid + rng.normal(0, 0.02 * mu, solid.shape)
    return vol.astype(np.float32), solid


def test_segment_recovers_the_support_and_keeps_the_hole():
    vol, solid = _volume()
    support, report = rf.segment(vol, opening_voxels=1, verbose=0)
    inter = (support & solid).sum()
    dice = 2 * inter / (support.sum() + solid.sum())
    assert dice > 0.9, dice
    n = vol.shape[0]
    assert not support[n // 2, n // 2, n // 2]
    assert report['components_kept'] == 1 and 0 < report['support_fraction'] < 0.5


def test_reference_from_reconstruction_has_the_density_and_the_binned_grid():
    vol, solid = _volume()
    ref, report = rf.reference_from_reconstruction(vol, 8.9e-3, binning=2, opening_voxels=1, verbose=0)
    assert ref.shape == (24, 24, 24) and ref.dtype == np.float32
    assert np.isclose(ref.max(), 8.9e-3) and 0 < (ref > 0).mean() < 0.5


def test_orientations_are_a_group_of_16_with_identity():
    v = np.random.default_rng(1).random((5, 6, 7))
    assert len(rf.ORIENTATION_NAMES) == 16
    assert np.array_equal(rf.apply_orientation(v, 'z+0'), v)
    assert rf.apply_orientation(v, 'z-T90').shape == (5, 6, 7) or rf.apply_orientation(v, 'z-T90').shape == (5, 7, 6)
    name, scores = rf.best_orientation(v, lambda w: float(w[0, 0, 0]), verbose=0)
    assert name in rf.ORIENTATION_NAMES and len(scores) == 16


def test_bounding_box_and_fov():
    sup = np.zeros((10, 10, 10), bool)
    sup[2:5, 3:4, 6:9] = True
    box = rf.bounding_box(sup, margin=1)
    assert box == (slice(1, 6), slice(2, 5), slice(5, 10))
    fov = rf.cylindrical_fov((3, 20, 20))
    assert fov.shape == (3, 20, 20) and fov[0, 10, 10] and not fov[0, 0, 0]
