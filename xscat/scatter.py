"""LEAP's first-order scatter model on a down-sampled geometry, and the up-sampling of its fields.

LEAP limits ``scatter_model`` to about 200^3 voxels and 256^2 detector pixels, so the reference is cropped to its
bounding box and block-averaged, the detector is block-binned, and the resulting scatter transmission ``S`` (scatter
divided by the open-beam signal) and LEAP's own gain ``P / (P + S)`` are up-sampled to the full detector afterwards.
"""
from dataclasses import dataclass

import numpy as np

from . import geometry as geo
from .reference import bounding_box


def detector_downsample_factor(num_rows, num_cols, target_side=204, max_side=255, max_pixels=256 ** 2):
    """Smallest integer factor with ``int(n / factor) <= target_side`` on both sides and LEAP's pixel limit met."""
    f = max(1, int(np.ceil(max(num_rows, num_cols) / float(target_side))) - 1)
    while True:
        r, c = int(num_rows / f), int(num_cols / f)
        if max(r, c) <= min(int(target_side), int(max_side)) and r * c <= int(max_pixels):
            return f
        f += 1


def volume_downsample_factor(shape_zyx, target_side=199, max_voxels=200 ** 3):
    """Smallest integer factor with every ``int(n / factor) <= target_side`` and at most ``max_voxels`` voxels."""
    f = max(1, int(np.ceil(max(shape_zyx) / float(target_side))) - 1)
    while True:
        dims = [int(s / f) for s in shape_zyx]
        if max(dims) <= int(target_side) and int(np.prod(dims)) <= int(max_voxels):
            return f
        f += 1


@dataclass
class ScatterFields:
    """Output of :func:`simulate_scatter` on LEAP's coarse grid.

    Attributes:
        s_lr: Scatter transmission ``S`` ``(view, row_lr, col_lr)``.
        gain_lr: LEAP's gain ``P / (P + S)`` on the same grid (``None`` if not requested).
        p_lr: Primary transmission of the phantom on the coarse grid (our evaluation of LEAP's tables).
        path_lr: Path length through the reference (mm) on the coarse grid.
        phantom_lr: The block-averaged phantom handed to LEAP (density), ``(z, y, x)``.
        voxel_lr_mm: Voxel size of ``phantom_lr``.
        detector_factor: Detector binning factor ``fd``.
        volume_factor: Volume binning factor ``fv``.
        seconds: Time spent in ``scatter_model``.
    """
    s_lr: np.ndarray
    gain_lr: np.ndarray
    p_lr: np.ndarray
    path_lr: np.ndarray
    phantom_lr: np.ndarray
    voxel_lr_mm: float
    detector_factor: int
    volume_factor: int
    seconds: float


def simulate_scatter(reference_density, ref_grid, geometry, physics, *, gpus=None, want_gain=True, detector_side=204, volume_side=199, verbose=1):
    """LEAP first-order scatter of a mass-density reference.

    Args:
        reference_density: Reference volume (g/mm^3), ``(z, y, x)`` on ``ref_grid``.
        ref_grid: :class:`~xscat.geometry.VolumeGrid` of the reference.
        geometry: Full-detector :class:`~xscat.geometry.ConeBeamGeometry` (all views to be corrected).
        physics: :class:`~xscat.physics.Physics` (scatter tables, density).
        gpus: LEAP device indices.
        want_gain: Also compute LEAP's own ratio gain (``jobType -1``) for the original ratio correction.
        detector_side, volume_side: LEAP size targets for the coarse detector and phantom.

    Returns:
        :class:`ScatterFields` and the LEAP model (needed to up-sample with :func:`upsample`).
    """
    import time
    box = bounding_box(reference_density > 0, margin=2)
    phantom = np.ascontiguousarray(reference_density[box], dtype=np.float32)
    centres = [(sl.start + sl.stop - 1) / 2.0 - (n - 1) / 2.0 for sl, n in zip(box, reference_density.shape)]
    vox = ref_grid.voxel_mm
    box_grid = geo.VolumeGrid(phantom.shape, vox, tuple(o + c * vox for o, c in zip(ref_grid.offsets_xyz, (centres[2], centres[1], centres[0]))))
    fv = volume_downsample_factor(phantom.shape, volume_side)
    fd = detector_downsample_factor(geometry.num_rows, geometry.num_cols, detector_side)
    model = geo.leap_model(geometry, box_grid, gpus)
    model.down_sample_projections([1, fd, fd])
    f_lr = model.down_sample_volume([fv, fv, fv], phantom) if fv > 1 else phantom
    f_lr = np.ascontiguousarray(np.clip(np.asarray(f_lr, dtype=np.float32), 0.0, None))
    det_lr = (int(model.get_numRows()), int(model.get_numCols()))
    if verbose:
        print(f'[scatter] phantom box {phantom.shape} -> x{fv} {f_lr.shape} ({vox * fv:.3f} mm voxels); detector x{fd} -> {det_lr} '
              f'({geometry.pixel_width * fd:.3f} mm pixels)')
    g_lr = model.allocate_projections()
    model.project(g_lr, f_lr)
    p_lr = physics.primary(g_lr)
    path_lr = np.ascontiguousarray(np.asarray(g_lr, dtype=np.float32) / np.float32(physics.density))
    del g_lr
    if not model.convert_to_modularbeam():
        raise RuntimeError('LEAP could not convert the geometry to modular-beam')
    s_dn, e_dn, det, sigma, dsigma = physics.scatter_tables
    t0 = time.time()
    s_lr = model.scatter_model(f_lr, s_dn, e_dn, det, sigma, dsigma, 0)
    if s_lr is None:
        raise RuntimeError('LEAP scatter_model (jobType 0) failed')
    s_lr = np.ascontiguousarray(np.clip(np.asarray(s_lr, dtype=np.float32), 0.0, None))
    seconds = time.time() - t0
    if verbose:
        print(f'[scatter] scatter_model -> S / air {s_lr.shape} in {seconds:.0f} s: mean {s_lr.mean():.5f} max {s_lr.max():.5f}')
    gain_lr = None
    if want_gain:
        t0 = time.time()
        gain_lr = model.scatter_model(f_lr, s_dn, e_dn, det, sigma, dsigma, -1)
        if gain_lr is None:
            raise RuntimeError('LEAP scatter_model (jobType -1) failed')
        gain_lr = np.ascontiguousarray(np.asarray(gain_lr, dtype=np.float32))
        diff = gain_lr - ratio_gain(p_lr, s_lr)
        if verbose:
            print(f'[scatter] scatter_model -> LEAP gain in {time.time() - t0:.0f} s: min {gain_lr.min():.4f} mean {gain_lr.mean():.4f}; '
                  f'against P_lr / (P_lr + S) from our primary: mean diff {diff.mean():+.5f}, max |diff| {np.abs(diff).max():.4f}')
    fields = ScatterFields(s_lr, gain_lr, p_lr, path_lr, f_lr, vox * fv, fd, fv, seconds)
    return fields, model


def upsample(model, lr_stack, geometry):
    """Up-sample a coarse ``(view, row_lr, col_lr)`` stack to the full detector with LEAP's interpolation."""
    fd = int(round(geometry.num_rows / lr_stack.shape[1]))
    n = lr_stack.shape[0]
    out = model.up_sample(np.array([1, fd, fd], dtype=np.float32), np.ascontiguousarray(lr_stack, dtype=np.float32),
                          dims=(n, geometry.num_rows, geometry.num_cols))
    if out is None or tuple(out.shape) != (n, geometry.num_rows, geometry.num_cols):
        raise RuntimeError('LEAP up_sample failed')
    return np.asarray(out, dtype=np.float32)


def ratio_gain(primary, scatter, min_gain=0.0):
    """The multiplicative scatter gain ``P / (P + S)``, optionally floored at ``min_gain``."""
    p = np.asarray(primary, dtype=np.float32)
    s = np.clip(np.asarray(scatter, dtype=np.float32), 0.0, None)
    g = p / np.maximum(p + s, np.finfo(np.float32).tiny)
    return np.maximum(g, np.float32(min_gain)) if min_gain and min_gain > 0 else g
