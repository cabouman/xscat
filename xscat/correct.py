"""The two ratio corrections, applied chunk by chunk to a projection stack.

Both take the measured transmission ``M`` and return the corrected transmission ``M G`` with a gain ``G = P / (P + S)``:

* :func:`ratio_leap` uses LEAP's gain from the coarse detector, up-sampled -- the original LEAP correction;
* :func:`ratio_fullp` uses the matched model primary at full detector resolution and LEAP's up-sampled ``S``.
"""
import time

import numpy as np

from . import scatter as sc


def bad_rows_from_bright(bright_row_median, fraction=0.5):
    """Detector rows whose open-beam median is below ``fraction`` of the panel median (dead or masked rows)."""
    b = np.asarray(bright_row_median, dtype=np.float64)
    return np.where(b < float(fraction) * np.median(b))[0]


def replace_rows(attenuation_vrc, bad_rows):
    """Replace every bad row of a ``(view, row, col)`` stack by the nearest good row, in place; returns the mapping."""
    if len(bad_rows) == 0:
        return {}
    bad = set(int(r) for r in bad_rows)
    good = np.array([r for r in range(attenuation_vrc.shape[1]) if r not in bad])
    if good.size == 0:
        raise ValueError('every detector row is bad')
    mapping = {}
    for r in sorted(bad):
        src = int(good[np.argmin(np.abs(good - r))])
        attenuation_vrc[:, r, :] = attenuation_vrc[:, src, :]
        mapping[r] = src
    return mapping


def open_beam_mask(meas, threshold=0.9, margin_px=8):
    """Pixels that see the open beam in a transmission image: above ``threshold`` and eroded by ``margin_px``."""
    from scipy import ndimage
    m = np.asarray(meas) > float(threshold)
    if margin_px > 0:
        m = ndimage.binary_erosion(m, iterations=int(margin_px), border_value=1)
    return m


def ratio_leap(meas, gain_full, min_gain=0.0):
    """Original LEAP ratio correction: ``M`` times LEAP's up-sampled gain.  Returns ``(T, G)``."""
    g = np.asarray(gain_full, dtype=np.float32)
    if min_gain and min_gain > 0:
        g = np.maximum(g, np.float32(min_gain))
    return (np.asarray(meas, dtype=np.float32) * g).astype(np.float32), g


def ratio_fullp(meas, primary_full, scatter_full, min_gain=0.0):
    """High-resolution ratio correction: ``M P / (P + S)`` with the full-resolution primary.  Returns ``(T, G)``."""
    g = sc.ratio_gain(primary_full, scatter_full, min_gain)
    return (np.asarray(meas, dtype=np.float32) * g).astype(np.float32), g


def correct_stack(attenuation, mode, *, fields, leap_model, geometry, projector=None, chunk=16, min_gain=0.0, keep_views=(), log=print):
    """Correct a full attenuation stack with one mode.

    Args:
        attenuation: Measured attenuation ``(view, row, col)``.
        mode: ``'ratio-leap'``, ``'ratio-fullp'`` (matched primary) or ``'ratio-fullp-bare'`` (sharp, as-registered primary).
        fields: :class:`~xscat.scatter.ScatterFields`.
        leap_model: The LEAP model returned by :func:`~xscat.scatter.simulate_scatter` (for the up-sampling).
        geometry: Full :class:`~xscat.geometry.ConeBeamGeometry`.
        projector: :class:`~xscat.primary.PrimaryProjector` (needed for the fullp modes).
        chunk: Views per chunk.
        min_gain: Floor of the gain (0 = none, as LEAP).
        keep_views: Views whose corrected attenuation and gain are returned for figures.
        log: Print function.

    Returns:
        ``(corrected attenuation (view, row, col), stats dict, picked dict {view: (attenuation, gain)})``.
    """
    raw = np.asarray(attenuation, dtype=np.float32)
    n_views = raw.shape[0]
    out = np.empty(raw.shape, dtype=np.float32)
    picked = {}
    gain_min, gain_sum, change_sum = np.inf, 0.0, 0.0
    t0 = time.time()
    for v0 in range(0, n_views, int(chunk)):
        v1 = min(n_views, v0 + int(chunk))
        meas = np.exp(-raw[v0:v1])
        if mode == 'ratio-leap':
            if fields.gain_lr is None:
                raise ValueError("ratio-leap needs LEAP's gain (simulate_scatter(..., want_gain=True))")
            corrected, gain = ratio_leap(meas, sc.upsample(leap_model, fields.gain_lr[v0:v1], geometry), min_gain)
        elif mode in ('ratio-fullp', 'ratio-fullp-bare'):
            if projector is None:
                raise ValueError('the fullp modes need a PrimaryProjector')
            s_full = np.clip(sc.upsample(leap_model, fields.s_lr[v0:v1], geometry), 0.0, None)
            views = np.arange(v0, v1)
            p_full, _ = projector.bare(views) if mode == 'ratio-fullp-bare' else projector.matched(views)
            corrected, gain = ratio_fullp(meas, p_full, s_full, min_gain)
        else:
            raise ValueError(f'unknown mode {mode!r}')
        out[v0:v1] = -np.log(np.maximum(corrected, 1e-12))
        gain_min = min(gain_min, float(gain.min()))
        gain_sum += float(gain.mean()) * (v1 - v0)
        change_sum += float((out[v0:v1] - raw[v0:v1]).mean()) * (v1 - v0)
        for v in keep_views:
            if v0 <= int(v) < v1:
                picked[int(v)] = (out[int(v)].copy(), gain[int(v) - v0].copy())
    stats = dict(mode=mode, gain_min=gain_min, gain_mean=gain_sum / n_views, mean_attenuation_change=change_sum / n_views, seconds=time.time() - t0)
    log(f'[{mode}] corrected {n_views} views in {stats["seconds"]:.0f} s: gain min {gain_min:.4f} mean {stats["gain_mean"]:.4f}, '
        f'mean attenuation change {stats["mean_attenuation_change"]:+.4f}')
    return out, stats, picked
