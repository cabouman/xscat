"""The model primary at full detector resolution, matched to the measurement.

A geometric projection of a binary reference has edges one detector pixel wide; the measured transmission does not:
the focal spot, the detector point-spread function, the rotation during an exposure and partial volume spread every
edge over a few pixels, and the registration of the reference leaves a fraction of a pixel.  Used pixel by pixel in
the gain ``P / (P + S)`` such a primary over-corrects the air side of an edge and under-corrects the object side,
which the reconstruction turns into streaks tangent to the surface.

:func:`match_primary` measures and removes both mismatches at a few calibration views without trusting the model
anywhere but at the silhouette: a sub-pixel shift between the measurement and ``P + S`` on a band around the
silhouette (normalised cross-correlation of high-pass filtered images), one rigid offset of the reference explaining
the per-view shifts (from numerically measured Jacobians), and the Gaussian width that makes the blurred primary plus
scatter match the measured edge profiles.  :class:`PrimaryProjector` then projects the re-aligned reference and blurs
the result.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from . import geometry as geo


# ----------------------------------------------------------------------------------------------------------------------
# edge matching (numpy only)
# ----------------------------------------------------------------------------------------------------------------------
def silhouette_band(path_length, width_px=8, min_path_mm=0.05, border_px=0):
    """Pixels within ``width_px`` of the silhouette boundary of the reference (``path_length > min_path_mm``)."""
    obj = np.asarray(path_length) > float(min_path_mm)
    band = np.zeros(obj.shape, dtype=bool)
    if not obj.any() or obj.all():
        return band
    edge = obj & ~ndimage.binary_erosion(obj)
    band = ndimage.binary_dilation(edge, iterations=max(1, int(width_px)))
    b = int(border_px)
    if b > 0:
        band[:b] = False; band[-b:] = False; band[:, :b] = False; band[:, -b:] = False
    return band


def _highpass(img, sigma_px):
    img = np.asarray(img, dtype=np.float64)
    return img - ndimage.gaussian_filter(img, float(sigma_px), mode='nearest')


def _parabolic_offset(c_minus, c_0, c_plus):
    denom = c_minus - 2.0 * c_0 + c_plus
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (c_minus - c_plus) / denom, -1.0, 1.0))


def estimate_shift(ref, img, band, max_shift_px=6, highpass_px=4.0, min_pixels=100):
    """Sub-pixel shift ``(dr, dc)`` that moves ``img`` onto ``ref`` on the pixels of ``band``.

    Normalised cross-correlation of high-pass filtered images over the integer shifts within ``max_shift_px``, refined
    by a parabola through the peak (``ndimage.shift`` convention: ``out[r] = in[r - dr]``).

    Returns:
        ``(dr, dc, peak)``; ``nan`` when the band holds fewer than ``min_pixels`` usable pixels.
    """
    ref = np.asarray(ref, dtype=np.float64)
    img = np.asarray(img, dtype=np.float64)
    band = np.asarray(band, dtype=bool)
    s = int(max_shift_px)
    margin = s + int(np.ceil(3.0 * float(highpass_px))) + 1
    usable = band.copy()
    usable[:margin] = False; usable[-margin:] = False; usable[:, :margin] = False; usable[:, -margin:] = False
    if usable.sum() < int(min_pixels):
        return float('nan'), float('nan'), float('nan')
    rows = np.where(usable.any(axis=1))[0]
    cols = np.where(usable.any(axis=0))[0]
    r0, r1 = rows[0] - margin, rows[-1] + margin + 1
    c0, c1 = cols[0] - margin, cols[-1] + margin + 1
    a = _highpass(ref[r0:r1, c0:c1], highpass_px)
    b = _highpass(img[r0:r1, c0:c1], highpass_px)
    w = usable[r0:r1, c0:c1]
    aw = a[w]
    aw = aw - aw.mean()
    an = float(np.sqrt(np.sum(aw ** 2))) + 1e-30
    ncc = np.full((2 * s + 1, 2 * s + 1), -np.inf)
    for i in range(-s, s + 1):
        for j in range(-s, s + 1):
            v = np.roll(b, (i, j), axis=(0, 1))[w]
            v = v - v.mean()
            ncc[i + s, j + s] = float(np.dot(aw, v)) / (an * (float(np.sqrt(np.sum(v ** 2))) + 1e-30))
    pi, pj = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    dr, dc = float(pi - s), float(pj - s)
    if 0 < pi < 2 * s:
        dr += _parabolic_offset(ncc[pi - 1, pj], ncc[pi, pj], ncc[pi + 1, pj])
    if 0 < pj < 2 * s:
        dc += _parabolic_offset(ncc[pi, pj - 1], ncc[pi, pj], ncc[pi, pj + 1])
    return dr, dc, float(ncc[pi, pj])


def solve_rigid_shift(shifts_px, jacobians):
    """Least-squares rigid offset (mm) from per-view detector shifts ``(V, 2)`` and Jacobians ``(V, 2, 3)``.

    Returns:
        ``(delta (3,), residuals (V, 2) with nan for ignored views, number of views used)``.
    """
    shifts = np.asarray(shifts_px, dtype=np.float64).reshape(-1, 2)
    jac = np.asarray(jacobians, dtype=np.float64).reshape(-1, 2, 3)
    ok = np.all(np.isfinite(shifts), axis=1) & np.all(np.isfinite(jac.reshape(-1, 6)), axis=1)
    if ok.sum() < 2:
        return np.zeros(3), np.full(shifts.shape, np.nan), int(ok.sum())
    delta, *_ = np.linalg.lstsq(jac[ok].reshape(-1, 3), shifts[ok].reshape(-1), rcond=None)
    res = np.full(shifts.shape, np.nan)
    res[ok] = shifts[ok] - np.einsum('vij,j->vi', jac[ok], delta)
    return delta, res, int(ok.sum())


def blur_views(stack, sigma_px):
    """Gaussian blur over the last two axes (a copy when ``sigma_px > 0``, the input otherwise)."""
    a = np.asarray(stack, dtype=np.float32)
    if not sigma_px or float(sigma_px) <= 0:
        return a
    sig = (0.0,) * (a.ndim - 2) + (float(sigma_px), float(sigma_px))
    return ndimage.gaussian_filter(a, sig, mode='nearest')


def fit_psf_sigma(meas, prim, scat, band, sigmas=None, min_pixels=100):
    """Gaussian width making ``blur(prim) + scat`` match ``meas`` on ``band`` (least squares over ``sigmas``, refined).

    Returns:
        ``(sigma_px, costs over the grid, rms before, rms at the best sigma)``.
    """
    meas = np.asarray(meas, dtype=np.float64)
    prim = np.asarray(prim, dtype=np.float64)
    scat = np.asarray(scat, dtype=np.float64)
    band = np.asarray(band, dtype=bool)
    sigmas = np.arange(0.0, 4.01, 0.1) if sigmas is None else np.asarray(sigmas, dtype=np.float64)
    if band.sum() < int(min_pixels):
        return float('nan'), np.full(sigmas.shape, np.nan), float('nan'), float('nan')
    margin = int(np.ceil(3.0 * float(sigmas.max()))) + 1
    rows = np.where(band.any(axis=1))[0]
    cols = np.where(band.any(axis=0))[0]
    r0, r1 = max(rows[0] - margin, 0), min(rows[-1] + margin + 1, band.shape[0])
    c0, c1 = max(cols[0] - margin, 0), min(cols[-1] + margin + 1, band.shape[1])
    m, p, s_, w = meas[r0:r1, c0:c1], prim[r0:r1, c0:c1], scat[r0:r1, c0:c1], band[r0:r1, c0:c1]
    costs = np.empty(sigmas.shape)
    for k, sig in enumerate(sigmas):
        pb = ndimage.gaussian_filter(p, float(sig), mode='nearest') if sig > 0 else p
        costs[k] = float(np.mean((m[w] - pb[w] - s_[w]) ** 2))
    k = int(np.argmin(costs))
    sigma = float(sigmas[k])
    if 0 < k < sigmas.size - 1:
        sigma += _parabolic_offset(costs[k - 1], costs[k], costs[k + 1]) * float(sigmas[k + 1] - sigmas[k])
    return max(sigma, 0.0), costs, float(np.sqrt(costs[0])), float(np.sqrt(costs[k]))


# ----------------------------------------------------------------------------------------------------------------------
# the projector
# ----------------------------------------------------------------------------------------------------------------------
class PrimaryProjector:
    """Projects the reference at full detector resolution and turns the path lengths into a primary transmission.

    Args:
        reference_density: Reference volume (g/mm^3), ``(z, y, x)``.
        ref_grid: Its :class:`~xscat.geometry.VolumeGrid`.
        geometry: Full :class:`~xscat.geometry.ConeBeamGeometry`.
        physics: :class:`~xscat.physics.Physics`.
        gpus: LEAP device indices.

    Attributes:
        offset_mm: Rigid re-alignment of the reference ``(x, y, z)`` in mm (set by :func:`match_primary`).
        blur_px: Gaussian system blur applied to the primary (detector pixels; set by :func:`match_primary`).
    """

    def __init__(self, reference_density, ref_grid, geometry, physics, gpus=None):
        self.reference = np.ascontiguousarray(reference_density, dtype=np.float32)
        self.grid = ref_grid
        self.geometry = geometry
        self.physics = physics
        self.gpus = gpus
        self.offset_mm = np.zeros(3)
        self.blur_px = 0.0
        self._model = None

    def project(self, views, offset_mm=None, blur_px=0.0):
        """``(P, L)`` at the given views: primary transmission (blurred by ``blur_px``) and path length (mm)."""
        views = np.asarray(views, dtype=int)
        grid = self.grid.shifted(np.zeros(3) if offset_mm is None else offset_mm)
        model = geo.leap_model(self.geometry.subset(views), grid, self.gpus)
        g = model.allocate_projections()
        model.project(g, self.reference)
        p = self.physics.primary(g)
        return blur_views(p, blur_px), np.asarray(g, dtype=np.float32) / np.float32(self.physics.density)

    def matched(self, views):
        """The primary at ``views`` with the fitted re-alignment and blur applied."""
        return self.project(views, self.offset_mm, self.blur_px)

    def bare(self, views):
        """The primary at ``views`` as registered, sharp."""
        return self.project(views)


@dataclass
class MatchResult:
    """Diagnostics of :func:`match_primary`."""
    calib_views: np.ndarray
    shifts_px_before: np.ndarray
    shifts_px_after: np.ndarray
    offset_mm: np.ndarray
    offset_px: np.ndarray
    shift_applied: bool
    views_used: int
    psf_px: float
    psf_px_per_view: np.ndarray
    psf_grid: np.ndarray
    psf_costs: list
    edge_rms_before: np.ndarray
    edge_rms_after: np.ndarray
    noise_air: float
    band_px: int
    seconds: float
    extras: dict = field(default_factory=dict)

    def as_dict(self):
        """JSON-friendly copy."""
        out = {}
        for k, v in self.__dict__.items():
            if k == 'extras':
                continue
            out[k] = v.tolist() if isinstance(v, np.ndarray) else ([c.tolist() for c in v] if k == 'psf_costs' else v)
        return out


def open_beam_noise(meas, threshold=0.9, margin_px=8, hp_sigma_px=2.0, min_pixels=500):
    """Transmission noise of an open-beam pixel: robust std of the high-pass filtered open-beam pixels of a stack."""
    m = np.asarray(meas, dtype=np.float32)
    if m.ndim == 2:
        m = m[None]
    vals = []
    for img in m:
        ob = img > float(threshold)
        if margin_px > 0:
            ob = ndimage.binary_erosion(ob, iterations=int(margin_px), border_value=1)
        if ob.sum() < min_pixels:
            continue
        hp = img - ndimage.gaussian_filter(img, float(hp_sigma_px), mode='nearest')
        v = hp[ob]
        vals.append(v - np.median(v))
    if not vals:
        return float('nan')
    v = np.concatenate(vals)
    return float(1.4826 * np.median(np.abs(v)) / np.sqrt(0.94))


def match_primary(projector, measured_transmission, scatter_full, calib_views, *, band_px=8, shift='rigid', max_shift_px=3.0,
                  psf='auto', psf_max_px=4.0, log=print):
    """Fit the rigid re-alignment and the system blur of the model primary; sets them on ``projector``.

    Args:
        projector: :class:`PrimaryProjector`.
        measured_transmission: Measured transmission at ``calib_views`` ``(K, rows, cols)``.
        scatter_full: Scatter transmission ``S`` at the same views, full detector.
        calib_views: Indices of the calibration views.
        band_px: Half width of the silhouette band used for the matching.
        shift: ``'rigid'`` to fit and apply one offset of the reference, ``'none'`` to skip.
        max_shift_px: Mismatches or offsets larger than this (detector pixels) are not trusted.
        psf: ``'auto'`` to fit the blur, ``'none'`` for no blur, or a number of pixels.
        psf_max_px: Largest blur tried by the fit.
        log: Print function for the progress lines.

    Returns:
        :class:`MatchResult`.
    """
    import time
    t0 = time.time()
    geometry = projector.geometry
    calib = np.asarray(calib_views, dtype=int)
    meas = np.asarray(measured_transmission, dtype=np.float32)
    s_c = np.clip(np.asarray(scatter_full, dtype=np.float32), 0.0, None)
    mag = geometry.magnification
    px = geometry.pixel_width
    search = int(np.ceil(max_shift_px)) + 2
    p_bare, L_c = projector.bare(calib)
    bands = np.stack([silhouette_band(L_c[k], band_px) for k in range(len(calib))])

    def mismatch(p_stack):
        return np.array([estimate_shift(meas[k], p_stack[k] + s_c[k], bands[k], max_shift_px=search)[:2] for k in range(len(calib))])

    def rms(sh):
        ok = np.all(np.isfinite(sh), axis=1)
        return float(np.sqrt(np.mean(sh[ok] ** 2))) if ok.any() else float('nan')

    shifts0 = mismatch(p_bare)
    log(f'[primary] silhouette mismatch, measurement vs P + S at {len(calib)} calibration views (px, row/col): '
        + ' '.join(f'{int(v)}:({a:+.2f},{b:+.2f})' for v, (a, b) in zip(calib, shifts0)) + f'; rms {rms(shifts0):.2f} px')
    offset, offset_px, applied, used, shifts1 = np.zeros(3), np.zeros(3), False, 0, shifts0
    p_al = p_bare
    if shift == 'rigid':
        probe_mm = 2.0 * projector.grid.voxel_mm
        jac = np.zeros((len(calib), 2, 3))
        for axis in range(3):
            probe = np.zeros(3)
            probe[axis] = probe_mm
            p_probe, _ = projector.project(calib, probe)
            for k in range(len(calib)):
                dr, dc, _ = estimate_shift(p_probe[k], p_bare[k], bands[k], max_shift_px=int(np.ceil(probe_mm * mag / px)) + 3)
                jac[k, :, axis] = (dr / probe_mm, dc / probe_mm)
        usable = np.all(np.isfinite(shifts0), axis=1) & (np.nanmax(np.abs(shifts0), axis=1, initial=0.0) <= max_shift_px)
        delta, _, used = solve_rigid_shift(np.where(usable[:, None], shifts0, np.nan), jac)
        delta_px = delta * mag / px
        if used >= 3 and np.all(np.isfinite(delta)) and float(np.max(np.abs(delta_px))) <= max_shift_px:
            projector.offset_mm = delta
            offset, offset_px, applied = delta, delta_px, True
            p_al, _ = projector.project(calib, delta)
            shifts1 = mismatch(p_al)
            log(f'[primary] reference re-aligned by (x, y, z) ({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}) mm = '
                f'({delta_px[0]:+.2f}, {delta_px[1]:+.2f}, {delta_px[2]:+.2f}) px from {used} views; residual mismatch rms {rms(shifts0):.2f} -> {rms(shifts1):.2f} px')
        else:
            log(f'[primary] WARNING: rigid re-alignment not applied ({used} usable views, fitted offset ({delta_px[0]:+.2f}, {delta_px[1]:+.2f}, '
                f'{delta_px[2]:+.2f}) px, limit {max_shift_px:g} px); the reference is used as registered')
    psf_arg = str(psf).strip().lower()
    grid_sig = np.arange(0.0, float(psf_max_px) + 1e-9, 0.1)
    sig_v = np.full(len(calib), np.nan)
    rms_b = np.full(len(calib), np.nan)
    rms_a = np.full(len(calib), np.nan)
    costs = []
    if psf_arg == 'auto':
        for k in range(len(calib)):
            sig_v[k], c, rms_b[k], rms_a[k] = fit_psf_sigma(meas[k], p_al[k], s_c[k], bands[k], grid_sig)
            costs.append(c)
        blur = float(np.nanmedian(sig_v)) if np.isfinite(sig_v).any() else 0.0
        log(f'[primary] system blur from the edge profiles: sigma {blur:.2f} px (per view ' + ' '.join(f'{x:.1f}' for x in sig_v)
            + f'); edge residual rms {np.nanmean(rms_b):.4f} -> {np.nanmean(rms_a):.4f} (transmission)')
        if blur >= float(psf_max_px) - 0.05:
            log('[primary] WARNING: the fitted blur sits at the upper limit; raise psf_max_px or check the registration')
    else:
        blur = 0.0 if psf_arg in ('none', '', '0') else float(psf_arg)
        log(f'[primary] system blur fixed at sigma {blur:.2f} px')
    projector.blur_px = blur
    noise = open_beam_noise(meas)
    log(f'[primary] open-beam transmission noise per pixel: {noise:.5f}')
    result = MatchResult(calib, shifts0, shifts1, offset, offset_px, applied, used, blur, sig_v, grid_sig, costs, rms_b, rms_a, noise,
                         int(band_px), time.time() - t0)
    result.extras = dict(p_bare=p_bare, p_aligned=p_al, bands=bands, meas=meas, scatter=s_c)
    return result
