"""Reader for Zeiss Metrotom scans (``UncorrectedSingle.uint16`` + ``0_reco_dll.log``).

The raw file holds dark frames, open-beam frames and the projections one after the other.  The reader normalises the
projections with the mean dark and open-beam frames, applies the per-view detector offsets of the log, takes the
negative logarithm and returns the attenuation stack together with the cone-beam geometry and the reconstruction grid
that the scanner's own log prescribes.
"""
import glob
import os

import numpy as np
from scipy import ndimage

from .geometry import ConeBeamGeometry, VolumeGrid

NUM_DARK_FRAMES = 10
NUM_BRIGHT_FRAMES = 90


def _lines_from(lines, phrase):
    """Index of the first line whose first word is ``phrase``."""
    for k, line in enumerate(lines):
        words = line.split()
        if words and words[0] == phrase:
            return k
    raise ValueError(f'phrase {phrase!r} not found in the log')


def _divisible_by_4(n, padding=50):
    """The scanner log's volume size, padded the way the original reconstruction code does."""
    return (n // 4) * 4 + 2 * padding if n % 4 != 0 else n + 100


def read_log(scan_dir):
    """Parse ``0_reco_dll.log``.

    Returns:
        dict with ``det_rows``, ``det_cols``, ``pixel_mm`` (x, y), ``sod``, ``sdd``, ``num_views``, ``grid_zyx`` (the padded
        reconstruction grid), ``center_offset`` (row, col) in pixels, ``angles_deg`` and ``proj_offsets_px`` (view, 2).
    """
    with open(os.path.join(scan_dir, '0_reco_dll.log'), 'r', encoding='utf8', errors='ignore') as f:
        lines = f.readlines()
    k = _lines_from(lines, 'detector')
    size = lines[k + 1].split('size: ')[1].split(' pixels')[0]
    det_cols, det_rows = int(size.split('x')[0]), int(size.split('x')[1])
    pitch = lines[k + 2].split('pitch: (')[1].split(')')[0]
    pix_x = float(pitch.split(' mm')[0])
    pix_y = float(pitch.split(' mm')[1].split(', ')[1].split(' mm')[0])
    k = _lines_from(lines, 'distance')
    sod = float(lines[k].split(':')[1].split('mm')[0])
    sdd = float(lines[k + 1].split(':')[1].split('mm')[0])
    k = _lines_from(lines, 'reconstruction')
    num_views = None
    for j in range(15):
        if 'number of projections' in lines[k + j]:
            num_views = int(lines[k + j].split(':')[1])
            break
    if num_views is None:
        raise ValueError('number of projections not found in the log')
    k = _lines_from(lines, 'volume')
    dims = [int(v) for v in lines[k + 1].split(':')[1].split('voxels')[0].split('x')]
    nxy = max(_divisible_by_4(dims[0]), _divisible_by_4(dims[1]))
    grid_zyx = (_divisible_by_4(dims[2]), nxy, nxy)
    k = _lines_from(lines, 'geometry')
    centre = lines[k + 2].split('(')[1].split(')')[0].split(',')
    pix_cent_col, pix_cent_row = float(centre[0]), float(centre[1])
    center_offset = (-(pix_cent_row - det_rows // 2), -(pix_cent_col - det_cols // 2))
    k = _lines_from(lines, 'index')
    rows = [ln.split() for ln in lines[k + 1:k + 1 + num_views]]
    angles_deg = np.array([float(r[1]) for r in rows], dtype=np.float64)
    offsets = np.array([[float(r[6]), float(r[7])] for r in rows], dtype=np.float64) * (sdd / sod) / pix_x
    return dict(det_rows=det_rows, det_cols=det_cols, pixel_mm=(pix_x, pix_y), sod=sod, sdd=sdd, num_views=num_views, grid_zyx=grid_zyx,
                center_offset=center_offset, angles_deg=angles_deg, proj_offsets_px=offsets)


def find_scan_dir(root):
    """The first directory below ``root`` that holds an ``UncorrectedSingle.uint16`` file."""
    hits = sorted(glob.glob(os.path.join(root, '**', 'UncorrectedSingle.uint16'), recursive=True))
    if not hits:
        raise FileNotFoundError(f'no UncorrectedSingle.uint16 under {root}')
    return os.path.dirname(hits[0])


def bright_row_medians(scan_dir, num_rows, num_cols):
    """Per-row median of the mean open-beam frame, read straight from the raw file (dead-row detection)."""
    path = os.path.join(scan_dir, 'UncorrectedSingle.uint16')
    frames = os.path.getsize(path) // (2 * num_rows * num_cols)
    mm = np.memmap(path, dtype=np.uint16, mode='r', shape=(frames, num_rows, num_cols))
    bright = np.asarray(mm[NUM_DARK_FRAMES:NUM_DARK_FRAMES + NUM_BRIGHT_FRAMES], dtype=np.float32).mean(axis=0)
    del mm
    return np.median(bright, axis=1)


def load_scan(scan_dir, view_subsample=1, verbose=1):
    """Attenuation stack, geometry and reconstruction grid of a Metrotom scan.

    Args:
        scan_dir: Directory with ``UncorrectedSingle.uint16`` and ``0_reco_dll.log``.
        view_subsample: Keep every n-th projection.
        verbose: Print a summary.

    Returns:
        ``(attenuation (view, row, col) float32, geometry, grid)`` with :class:`~xscat.geometry.ConeBeamGeometry`
        and :class:`~xscat.geometry.VolumeGrid`.  Non-finite attenuation values are set to zero.
    """
    info = read_log(scan_dir)
    rows, cols, n = info['det_rows'], info['det_cols'], info['num_views']
    path = os.path.join(scan_dir, 'UncorrectedSingle.uint16')
    mm = np.memmap(path, dtype=np.uint16, mode='r', shape=(NUM_DARK_FRAMES + NUM_BRIGHT_FRAMES + n, rows, cols))
    dark = np.asarray(mm[:NUM_DARK_FRAMES], dtype=np.float32).mean(axis=0)
    bright = np.asarray(mm[NUM_DARK_FRAMES:NUM_DARK_FRAMES + NUM_BRIGHT_FRAMES], dtype=np.float32).mean(axis=0) - dark
    views = np.arange(0, n, max(1, int(view_subsample)))
    att = np.empty((len(views), rows, cols), dtype=np.float32)
    for k, v in enumerate(views):
        counts = np.asarray(mm[NUM_DARK_FRAMES + NUM_BRIGHT_FRAMES + v], dtype=np.float32) - dark
        trans = counts / bright
        trans = ndimage.shift(trans, info['proj_offsets_px'][v], order=1, mode='nearest')
        with np.errstate(divide='ignore', invalid='ignore'):
            a = -np.log(trans)
        a[~np.isfinite(a)] = 0.0
        att[k] = a
    del mm
    pix_x, pix_y = info['pixel_mm']
    center_row = rows / 2 - info['center_offset'][0] - 0.5
    center_col = cols / 2 - info['center_offset'][1] - 0.5
    geometry = ConeBeamGeometry(len(views), rows, cols, pix_y, pix_x, center_row, center_col,
                                np.ascontiguousarray(info['angles_deg'][views], dtype=np.float32), info['sod'], info['sdd'])
    grid = VolumeGrid(tuple(int(s) for s in info['grid_zyx']), pix_x / (info['sdd'] / info['sod']))
    if verbose:
        print(f'[metrotom] {scan_dir}: {len(views)} of {n} views, detector {rows} x {cols} of {pix_x:g} mm, sod {info["sod"]:.2f} sdd {info["sdd"]:.2f} mm, '
              f'centre (row, col) ({center_row:.2f}, {center_col:.2f}); grid (z, y, x) {grid.shape_zyx} of {grid.voxel_mm:.5f} mm')
    return att, geometry, grid
