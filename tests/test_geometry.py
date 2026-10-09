import numpy as np

from xscat import geometry as geo


def test_binned_grid_offsets_centre_the_binned_grid():
    off = geo.binned_grid_offsets((10, 9, 8), 2, 0.5)        # x, y, z
    assert off == (0.0, -0.25, 0.0)
    assert geo.binned_grid_offsets((952, 848, 848), 2, 0.1) == (0.0, 0.0, 0.0)


def test_block_means_preserve_mass():
    rng = np.random.default_rng(0)
    stack = rng.random((3, 9, 11)).astype(np.float32)
    b = geo.block_mean_views(stack, 2)
    assert b.shape == (3, 4, 5)
    assert np.isclose(b[0, 0, 0], stack[0, :2, :2].mean())
    vol = rng.random((8, 6, 6)).astype(np.float32)
    v = geo.block_mean_volume(vol, 2)
    assert v.shape == (4, 3, 3) and np.isclose(v.sum() * 8, vol.sum(), rtol=1e-5)


def test_geometry_subset_and_binning():
    g = geo.ConeBeamGeometry(10, 100, 120, 0.1, 0.1, 49.5, 60.0, np.arange(10, dtype=np.float32), 300.0, 900.0)
    assert np.isclose(g.magnification, 3.0)
    s = g.subset([2, 5])
    assert s.num_views == 2 and list(s.angles_deg) == [2.0, 5.0]
    b = g.binned(2)
    assert (b.num_rows, b.num_cols) == (50, 60) and np.isclose(b.pixel_width, 0.2) and np.isclose(b.center_row, 24.5)
    grid = geo.VolumeGrid((20, 20, 20), 0.05)
    assert grid.binned(2).voxel_mm == 0.1 and grid.shifted((1.0, 0.0, 0.0)).offsets_xyz == (1.0, 0.0, 0.0)
