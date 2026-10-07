"""Reference volume of the scanned part: a segmentation of a first reconstruction (default) or an aligned CAD model.

The reference only has to say where the material is.  The scatter model projects it on a coarse grid, and the
high-resolution primary projects it at full resolution; both use the volume as a mass density (fraction of solid times
the material density, g/mm^3) in LEAP's ``(z, y, x)`` order.
"""
import numpy as np
from scipy import ndimage

from .geometry import block_mean_volume


def cylindrical_fov(shape_zyx, margin_fraction=0.02):
    """True inside the cylinder inscribed in the ``(y, x)`` square, shrunk by ``margin_fraction`` of the width."""
    nz, ny, nx = shape_zyx
    yy, xx = np.mgrid[:ny, :nx]
    r = 0.5 * min(ny, nx) * (1.0 - margin_fraction)
    disc = (yy - (ny - 1) / 2.0) ** 2 + (xx - (nx - 1) / 2.0) ** 2 <= r ** 2
    return np.broadcast_to(disc[None], (nz, ny, nx))


def bounding_box(support, margin=4):
    """Slices of the bounding box of a boolean volume, widened by ``margin`` voxels and clipped to the array."""
    idx = np.where(support)
    if idx[0].size == 0:
        raise ValueError('empty support has no bounding box')
    out = []
    for axis in range(support.ndim):
        lo = max(0, int(idx[axis].min()) - margin)
        hi = min(support.shape[axis], int(idx[axis].max()) + 1 + margin)
        out.append(slice(lo, hi))
    return tuple(out)


def _ball(radius):
    r = int(radius)
    z, y, x = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (x * x + y * y + z * z) <= r * r


def _chunked_binary(op, mask, structure, radius, chunk=48):
    """A binary morphology operation applied along z in overlapping chunks (bounded temporaries for large volumes)."""
    out = np.zeros_like(mask, dtype=bool)
    nz = mask.shape[0]
    r = int(radius)
    for a in range(0, nz, chunk):
        b = min(nz, a + chunk)
        lo, hi = max(0, a - r), min(nz, b + r)
        block = op(mask[lo:hi], structure=structure)
        out[a:b] = block[a - lo:a - lo + (b - a)]
    return out


def multiotsu_thresholds(values, classes=3, nbins=256):
    """Multi-Otsu thresholds of a sample (histogram between the 0.1 and 99.9 percentiles); ``classes - 1`` ascending values."""
    from skimage.filters import threshold_multiotsu
    v = np.asarray(values, dtype=np.float32).ravel()
    v = v[np.isfinite(v)]
    if v.size < 1000:
        raise ValueError(f'multi-Otsu needs more than 1000 finite samples, got {v.size}')
    lo, hi = np.percentile(v, [0.1, 99.9])
    return [float(t) for t in threshold_multiotsu(np.clip(v, lo, hi), classes=int(classes), nbins=int(nbins))]


def _components_with_seeds(low_mask, seed_mask):
    labels, n = ndimage.label(low_mask)
    if n == 0:
        return np.zeros_like(low_mask, dtype=bool), 0, 0
    ids = np.unique(labels[seed_mask & low_mask])
    ids = ids[ids > 0]
    return np.isin(labels, ids), int(n), int(ids.size)


def _largest_components(mask, keep, min_fraction):
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask, int(n), int(n)
    sizes = ndimage.sum(np.ones(mask.shape, dtype=np.float32), labels, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1]
    largest = float(sizes[order[0]])
    chosen = [int(i) + 1 for k, i in enumerate(order) if k < int(keep) or sizes[i] >= float(min_fraction) * largest]
    return np.isin(labels, chosen), int(n), len(chosen)


def segment(vol_zyx, *, fov=None, low_fraction=0.5, smooth_sigma=1.0, edge_margin_voxels=2, opening_voxels=2, keep_components=1,
            min_component_fraction=0.01, sample_stride=2, thresholds=None, verbose=1):
    """Single-material support of a reconstruction from a three-class multi-Otsu split.

    1. The volume is Gaussian-smoothed (``smooth_sigma`` voxels).
    2. The voxels inside ``fov`` (default: the inscribed cylinder) are split into air, grey (partial volume, streaks,
       haze) and solid with multi-Otsu.
    3. Seeds are the voxels above the upper threshold.  The object grows into the grey class down to
       ``low = t_low + low_fraction (t_high - t_low)``, only in regions connected to seeds and within
       ``edge_margin_voxels`` of a seed.
    4. A binary opening with a ball of ``opening_voxels`` removes speckle and thin bridges; the ``keep_components``
       largest components are kept.  Internal channels are not filled.

    Returns:
        ``(support bool (z, y, x), report dict)``.
    """
    v = np.nan_to_num(np.asarray(vol_zyx, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if smooth_sigma and smooth_sigma > 0:
        v = ndimage.gaussian_filter(v, float(smooth_sigma), mode='nearest')
    fov = cylindrical_fov(v.shape) if fov is None else np.asarray(fov, dtype=bool)
    stride = (slice(None, None, max(1, int(sample_stride))),) * 3
    if thresholds is None:
        thresholds = multiotsu_thresholds(v[stride][fov[stride]], classes=3)
    thresholds = [float(t) for t in thresholds]
    t_low, t_high = thresholds[0], thresholds[-1]
    low = t_low + float(low_fraction) * (t_high - t_low)
    seeds = (v > t_high) & fov
    grown, n_low, n_seeded = _components_with_seeds((v > low) & fov, seeds)
    m = int(edge_margin_voxels)
    if m > 0:
        grown &= _chunked_binary(ndimage.binary_dilation, seeds, _ball(m), m)
    support = grown
    r = int(opening_voxels)
    if r > 0 and support.any():
        ball = _ball(r)
        eroded = _chunked_binary(ndimage.binary_erosion, support, ball, r)
        support = _chunked_binary(ndimage.binary_dilation, eroded, ball, r) & support
    n_comp, n_kept = 0, 0
    if int(keep_components) > 0 and support.any():
        support, n_comp, n_kept = _largest_components(support, keep_components, min_component_fraction)
    report = dict(thresholds=thresholds, low_threshold=low, low_fraction=float(low_fraction), smooth_sigma=float(smooth_sigma or 0.0),
                  edge_margin_voxels=m, opening_voxels=r, seed_fraction=float(seeds.mean()), grown_fraction=float(grown.mean()),
                  support_fraction=float(support.mean()), components_above_low=n_low, components_with_seeds=n_seeded,
                  components_after_opening=n_comp, components_kept=n_kept)
    if verbose:
        print(f'[reference] multi-Otsu thresholds {", ".join(f"{t:.5g}" for t in thresholds)}; seeds (> {t_high:.5g}) grown down to {low:.5g}, '
              f'edge margin {m} vox, opening {r} vox, {keep_components} component(s): seeds {100 * report["seed_fraction"]:.2f} % -> '
              f'support {100 * report["support_fraction"]:.2f} % of the volume')
    return np.ascontiguousarray(support), report


def reference_from_reconstruction(volume_zyx, density_g_mm3, binning=1, **segment_kwargs):
    """Mass-density reference from a reconstruction: block-bin, segment, multiply the support by the density.

    Args:
        volume_zyx: Reconstruction (attenuation, ``(z, y, x)``).
        density_g_mm3: Density of the solid material.
        binning: Block binning applied before the segmentation (the reference grid).
        **segment_kwargs: Passed to :func:`segment`.

    Returns:
        ``(density volume float32, report)`` where the report holds the segmentation statistics.
    """
    src = block_mean_volume(volume_zyx, binning)
    support, report = segment(src, **segment_kwargs)
    return np.ascontiguousarray(support.astype(np.float32) * np.float32(density_g_mm3)), report


# ----------------------------------------------------------------------------------------------------------------------
# aligned CAD (optional)
# ----------------------------------------------------------------------------------------------------------------------
ORIENTATION_NAMES = tuple(f'{z}{t}{k}' for z in ('z+', 'z-') for t in ('', 'T') for k in (0, 90, 180, 270))


def load_cad_tiff(path, verbose=1):
    """A binary CAD stack (tiff pages along z) as a uint8 ``(z, y, x)`` array with values 0 / 1."""
    import tifffile
    with tifffile.TiffFile(path) as t:
        cad = t.series[0].asarray()
    if cad.ndim != 3:
        raise ValueError(f'{path}: expected a 3-D stack, got shape {cad.shape}')
    cad = np.ascontiguousarray((cad > 0).astype(np.uint8))
    if verbose:
        print(f'[reference] CAD {path}: shape (z, y, x) {cad.shape}, filled {100 * cad.mean():.2f} %')
    return cad


def apply_orientation(vol_zyx, name):
    """One of the 16 axis conventions that keep z the rotation axis (``'z+0'`` is the identity)."""
    if name not in ORIENTATION_NAMES:
        raise ValueError(f'unknown orientation {name!r}; use one of {ORIENTATION_NAMES}')
    v = vol_zyx[::-1] if name.startswith('z-') else vol_zyx
    rest = name[2:]
    if rest.startswith('T'):
        v = np.swapaxes(v, 1, 2)
        rest = rest[1:]
    return np.ascontiguousarray(np.rot90(v, int(rest) // 90, axes=(1, 2)))


def best_orientation(vol_zyx, score, verbose=1):
    """The orientation maximising ``score(oriented volume)`` over the 16 conventions; returns ``(name, scores)``."""
    scores = {name: float(score(apply_orientation(vol_zyx, name))) for name in ORIENTATION_NAMES}
    best = max(scores, key=scores.get)
    if verbose:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        print('[reference] CAD orientation scores: ' + ', '.join(f'{n} {s:+.4f}' for n, s in ranked[:4]) + f' (best {best})')
    return best, scores


def reference_from_cad(cad_zyx, density_g_mm3, binning=1, orientation='z+0'):
    """Mass-density reference from a binary CAD on the reconstruction grid (fractional occupancy after binning)."""
    cad = apply_orientation(np.asarray(cad_zyx), orientation)
    return np.ascontiguousarray(block_mean_volume(cad, binning) * np.float32(density_g_mm3), dtype=np.float32)
