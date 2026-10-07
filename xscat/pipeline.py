"""The complete correction of one scan: load, reference, LEAP scatter, matched primary, corrections, FDK, figures.

:func:`run` is what ``demo/demo_blade.py`` calls; every stage is a public function of the other modules, so a user
can also assemble a variant from the pieces.
"""
import json
import os
import time

import numpy as np

from . import correct as co
from . import geometry as geo
from . import metrotom
from . import physics as ph
from . import primary as pr
from . import reference as rf
from . import scatter as sc
from .figures import Figures

MODES = ('ratio-leap', 'ratio-fullp', 'ratio-fullp-bare')


def _log(*a):
    print(*a, flush=True)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def cad_orientation_check(cad, grid, geometry, attenuation, picks, physics, gpus, binning=4, orientation='auto', log=_log):
    """Pick the axis convention of a CAD stack by correlating its projections with the measured attenuation."""
    cb = int(binning)
    grid_chk = grid.binned(cb)
    geom_chk = geometry.subset(picks).binned(cb)
    meas_chk = geo.block_mean_views(attenuation[picks], cb)
    cad_chk = geo.block_mean_volume(cad, cb)
    model = geo.leap_model(geom_chk, grid_chk, gpus)

    def union_corr(a, b, frac=0.05):
        keep = (a > frac * a.max()) | (b > frac * b.max())
        if keep.sum() < 100:
            return 0.0
        return float(np.corrcoef(a[keep].ravel().astype(np.float64), b[keep].ravel().astype(np.float64))[0, 1])

    def score(vol):
        g = model.allocate_projections()
        model.project(g, np.ascontiguousarray(vol * np.float32(physics.density), dtype=np.float32))
        return float(np.mean([union_corr(g[j], meas_chk[j]) for j in range(len(picks))]))

    if orientation == 'auto':
        name, scores = rf.best_orientation(cad_chk, score, verbose=1)
    else:
        name, scores = orientation, {orientation: score(rf.apply_orientation(cad_chk, orientation))}
    log(f'[reference] CAD orientation {name}: projection correlation {scores[name]:+.4f}')
    return name, scores


def run(scan_dir, out_dir, *, material='Ni', density_g_mm3=8.902e-3, kvp=200.0, take_off_deg=11.0, filters='Sn:1.0', detector='CsI:0.6',
        modes=('ratio-leap', 'ratio-fullp'), reference='seg', cad_path=None, cad_orientation='auto', view_subsample=1, det_subsample=1,
        fdk_binning=1, ref_binning=2, seg_low_fraction=0.5, seg_smooth_sigma=1.0, seg_opening_mm=0.3, leap_detector_side=204, leap_volume_side=199,
        match_shift='rigid', match_psf='auto', match_band_px=8, match_max_shift_px=3.0, match_psf_max_px=4.0, calib_views=12, min_gain=0.0,
        chunk=16, num_picks=3, tag=None, save_npy=False, viewer=True, gpus=None):
    """Correct a Metrotom scan with the requested modes and compare the FDK reconstructions.

    Args:
        scan_dir: Directory with ``UncorrectedSingle.uint16`` and ``0_reco_dll.log``.
        out_dir: Where figures, the report and optional arrays go (created).
        material, density_g_mm3, kvp, take_off_deg, filters, detector: Physics of the scan (:class:`~xscat.physics.Physics`).
        modes: Any of ``'ratio-leap'`` (original LEAP), ``'ratio-fullp'`` (matched high-resolution primary), ``'ratio-fullp-bare'``.
        reference: ``'seg'`` (segmentation of the raw+BH FDK, default) or ``'cad'`` (binary tiff on the reconstruction grid).
        cad_path, cad_orientation: CAD stack and its axis convention (``'auto'`` picks the best of 16 by projection correlation).
        view_subsample, det_subsample: Use every n-th view / block-bin the detector by n (quick trials).
        fdk_binning: Reconstruction grid binning of the FDK volumes.
        ref_binning: Grid binning of the reference (multiple of ``fdk_binning``).
        seg_low_fraction, seg_smooth_sigma, seg_opening_mm: Segmentation settings (:func:`~xscat.reference.segment`).
        leap_detector_side, leap_volume_side: Size targets for LEAP's scatter model.
        match_shift, match_psf, match_band_px, match_max_shift_px, match_psf_max_px, calib_views: Primary matching (:func:`~xscat.primary.match_primary`).
        min_gain: Floor of the ratio gains (0 = none).
        chunk: Views per correction chunk.
        num_picks: Views used for the projection figures.
        tag: File name prefix (default from the settings).
        save_npy: Also save the corrected attenuation stacks and FDK volumes.
        viewer: Open the slice viewers at the end (mbirjax).
        gpus: LEAP device indices (default: detected).

    Returns:
        dict with ``volumes``, ``labels``, ``report`` and ``picked`` (per-view arrays of the figures).
    """
    import importlib
    try:
        importlib.import_module('leapctype')
    except ImportError as exc:
        raise ImportError('xscat needs LEAP (leapct) and xrayphysics; see the installation page of the documentation') from exc
    modes = [m for m in modes]
    for m in modes:
        if m not in MODES:
            raise ValueError(f'unknown mode {m}; choose from {MODES}')
    os.makedirs(out_dir, exist_ok=True)
    tag = tag or f'xscat_{reference}_viewSub{view_subsample}_detSub{det_subsample}_fdkBin{fdk_binning}'
    report = dict(scan_dir=scan_dir, modes=modes, reference=reference, settings=dict(material=material, density_g_mm3=density_g_mm3, kvp=kvp, filters=filters,
                  detector=detector, view_subsample=view_subsample, det_subsample=det_subsample, fdk_binning=fdk_binning, ref_binning=ref_binning))
    figs = Figures(out_dir, tag, log=_log)
    gpus = geo.configure_gpus() if gpus is None else gpus

    # ---- 1. data ----
    t0 = time.time()
    raw, geometry, grid = metrotom.load_scan(scan_dir, view_subsample=view_subsample)
    d = int(det_subsample)
    if d > 1:
        raw = np.ascontiguousarray(geo.block_mean_views(raw, d))
        geometry = geometry.binned(d)
        grid = geo.VolumeGrid(tuple(n // d for n in grid.shape_zyx), grid.voxel_mm * d)
    bad = co.bad_rows_from_bright(metrotom.bright_row_medians(scan_dir, geometry.num_rows * d, geometry.num_cols * d))
    bad = sorted(set(int(r) // d for r in bad))
    row_map = co.replace_rows(raw, bad)
    picks = np.linspace(0, geometry.num_views - 1, num_picks + 2)[1:-1].round().astype(int)
    _log(f'[data] attenuation {raw.shape}, bad rows {bad} -> {row_map}; loaded in {time.time() - t0:.0f} s')
    report['geometry'] = dict(num_views=geometry.num_views, rows=geometry.num_rows, cols=geometry.num_cols, pixel_mm=geometry.pixel_width, sod=geometry.sod,
                              sdd=geometry.sdd, grid_zyx=list(grid.shape_zyx), voxel_mm=grid.voxel_mm, bad_rows=bad, picks=picks.tolist())

    # ---- 2. physics, FDK baselines ----
    physics = ph.Physics(material, density_g_mm3, kvp, take_off_deg, filters, detector)
    report['physics'] = dict(effective_energy_keV=physics.effective_energy_keV, mu_expected_per_mm=physics.mu_expected_per_mm)
    fb = int(fdk_binning)
    grid_fdk = grid.binned(fb)
    leapct = geo.leap_model(geometry, grid_fdk, gpus)
    volumes, labels = [], []
    t = time.time()
    volumes.append(leapct.FBP(raw)); labels.append('raw FDK')
    raw_bh = np.array(raw, dtype=np.float32, order='C', copy=True)
    physics.apply_beam_hardening(leapct, raw_bh)
    fdk_bh = leapct.FBP(raw_bh)
    del raw_bh
    volumes.append(fdk_bh); labels.append('raw+BH FDK')
    _log(f'[fdk] raw and raw+BH FDK {fdk_bh.shape} in {time.time() - t:.0f} s')

    # ---- 3. reference ----
    rb = int(ref_binning)
    if rb % fb != 0:
        raise ValueError('ref_binning must be a multiple of fdk_binning')
    grid_ref = grid.binned(rb)
    if reference == 'cad':
        if not cad_path:
            raise ValueError("reference='cad' needs cad_path")
        cad = rf.load_cad_tiff(cad_path)
        if d > 1:
            cad = geo.block_mean_volume(cad, d)
        if tuple(cad.shape) != tuple(grid.shape_zyx):
            raise ValueError(f'CAD grid {cad.shape} does not match the reconstruction grid {grid.shape_zyx}')
        name, scores = cad_orientation_check(cad, grid, geometry, raw, picks, physics, gpus, orientation=cad_orientation)
        ref_density = rf.reference_from_cad(cad, physics.density, rb, name)
        report['reference_info'] = dict(kind='cad', orientation=name, scores=scores)
    else:
        ref_density, seg_report = rf.reference_from_reconstruction(fdk_bh, physics.density, rb // fb, low_fraction=seg_low_fraction, smooth_sigma=seg_smooth_sigma,
                                                                   opening_voxels=int(round(seg_opening_mm / grid_ref.voxel_mm)))
        report['reference_info'] = dict(kind='seg', **seg_report)
    _log(f'[reference] {reference} on a {ref_density.shape} grid of {grid_ref.voxel_mm:.4f} mm, mass {float(ref_density.sum()) * grid_ref.voxel_mm ** 3:.1f} g')

    # ---- 4. LEAP scatter on the coarse grid ----
    fields, leap_sc = sc.simulate_scatter(ref_density, grid_ref, geometry, physics, gpus=gpus, want_gain='ratio-leap' in modes,
                                          detector_side=leap_detector_side, volume_side=leap_volume_side)
    report['scatter'] = dict(detector_factor=fields.detector_factor, volume_factor=fields.volume_factor, voxel_lr_mm=fields.voxel_lr_mm,
                             s_mean=float(fields.s_lr.mean()), s_max=float(fields.s_lr.max()), seconds=fields.seconds)
    figs.reference('01_reference_and_phantom', ref_density / np.float32(physics.density), fields.phantom_lr / np.float32(physics.density), grid_ref.voxel_mm,
                   fields.voxel_lr_mm, f'reference ({reference}) and the phantom handed to LEAP, fraction of solid {material}')
    figs.lowres('02_leap_coarse_fields', raw[picks], fields.s_lr, fields.p_lr, fields.gain_lr, picks)

    # ---- 5. the full-resolution primary, matched ----
    projector, match = None, None
    if any(m.startswith('ratio-fullp') for m in modes):
        projector = pr.PrimaryProjector(ref_density, grid_ref, geometry, physics, gpus)
        calib = np.unique(np.concatenate([np.linspace(0, geometry.num_views - 1, max(3, int(calib_views)) + 2)[1:-1].round().astype(int), picks]))
        s_c = sc.upsample(leap_sc, fields.s_lr[calib], geometry)
        match = pr.match_primary(projector, np.exp(-raw[calib]), s_c, calib, band_px=match_band_px, shift=match_shift, max_shift_px=match_max_shift_px,
                                 psf=match_psf, psf_max_px=match_psf_max_px, log=_log)
        report['primary_matching'] = match.as_dict()
        k = int(np.argmin(np.abs(calib - picks[len(picks) // 2])))
        gain_row = None
        if fields.gain_lr is not None:
            g_full = sc.upsample(leap_sc, fields.gain_lr[calib[k]:calib[k] + 1], geometry)[0]
            band = match.extras['bands'][k]
            rows_b = np.where(band.any(axis=1))[0]
            row = int(rows_b[np.argmax(band[rows_b].sum(axis=1))]) if rows_b.size else geometry.num_rows // 2
            gain_row = g_full[row]
        figs.edge_match('03_primary_edge_match', match, k, int(calib[k]), gain_row)

    # ---- 6. corrections, beam hardening, FDK ----
    picked = {}
    report['corrections'] = {}
    for mode in modes:
        corrected, stats, pk = co.correct_stack(raw, mode, fields=fields, leap_model=leap_sc, geometry=geometry, projector=projector, chunk=chunk,
                                                min_gain=min_gain, keep_views=picks, log=_log)
        for v, (att, gain) in pk.items():
            picked[(mode, v)] = dict(attenuation=att, gain=gain)
        physics.apply_beam_hardening(leapct, corrected)
        for v in picks:
            picked[(mode, int(v))]['attenuation_bh'] = corrected[int(v)].copy()
        stats['open_beam_residual'] = {int(v): float(corrected[int(v)][co.open_beam_mask(np.exp(-raw[int(v)]), 0.9, max(4, geometry.num_cols // 200))].mean()) for v in picks}
        if save_npy:
            np.save(os.path.join(out_dir, f'{tag}_{mode}_attenuation_bh_vrc.npy'), corrected)
        t = time.time()
        vol = leapct.FBP(corrected)
        stats['seconds_fdk'] = time.time() - t
        del corrected
        volumes.append(vol); labels.append(f'{mode} +BH FDK')
        report['corrections'][mode] = stats

    # ---- 7. figures ----
    v = int(picks[len(picks) // 2])
    ratio_modes = [m for m in modes]
    figs.gains('04_gains_full_resolution', raw[v], v, {m: picked[(m, v)]['gain'] for m in ratio_modes})
    figs.corrected('05_corrected_projections', raw[v], v, {m: picked[(m, v)]['attenuation'] for m in modes})
    edges = np.array([0.5, 1, 2, 3, 5, 7, 10, 13, 16, 20, 25, 30, 40, 60], dtype=np.float32)
    centres = 0.5 * (edges[:-1] + edges[1:])
    curves = {}
    report['linearity'] = {}
    fd = fields.detector_factor
    for label, mode in [('raw+BH', None)] + [(m, m) for m in modes]:
        med = []
        for vv in picks:
            if mode is None:
                img = np.array(raw[int(vv)][None], dtype=np.float32, order='C', copy=True)
                physics.apply_beam_hardening(leapct, img)
                img = img[0]
            else:
                img = picked[(mode, int(vv))]['attenuation_bh']
            att_lr = geo.block_mean_views(img[None], fd)[0]
            L = fields.path_lr[int(vv)][:att_lr.shape[0], :att_lr.shape[1]]
            med.append([float(np.median(att_lr[(L >= a) & (L < b)])) if ((L >= a) & (L < b)).sum() >= 20 else np.nan for a, b in zip(edges[:-1], edges[1:])])
        med = np.nanmedian(np.array(med), axis=0)
        curves[label] = med
        ok = np.isfinite(med) & (centres <= 12)
        slope = float(np.sum(med[ok] * centres[ok]) / np.sum(centres[ok] ** 2)) if ok.sum() >= 2 else float('nan')
        report['linearity'][label] = dict(path_mm=centres.tolist(), median_attenuation=[None if not np.isfinite(x) else float(x) for x in med], slope_thin_per_mm=slope)
        _log(f'[linearity] {label:12s}: attenuation after BH / reference path, thin part {slope:.4f} /mm (expected {physics.mu_expected_per_mm:.4f})')
    figs.linearity('06_attenuation_vs_pathlength', centres, curves, physics.mu_expected_per_mm)
    vmax = float(np.percentile(volumes[-1][volumes[-1].shape[0] // 2], 99.8))
    figs.volumes('07_fdk_comparison', volumes, labels, vmax)
    if save_npy:
        for vol, lab in zip(volumes, labels):
            np.save(os.path.join(out_dir, f'{tag}_fdk_{lab.replace(" ", "_").replace("+", "")}.npy'), vol)
    with open(os.path.join(out_dir, f'{tag}_report.json'), 'w') as fh:
        json.dump(_json_safe(report), fh, indent=2)
    _log(f'[done] report {os.path.join(out_dir, f"{tag}_report.json")}')

    # ---- 8. viewers ----
    if viewer:
        from .viewer import slice_viewer
        slice_viewer(np.ascontiguousarray(ref_density / np.float32(physics.density)), title=f'1/3  volume projected for P_sim: {reference} at {grid_ref.voxel_mm:.3f} mm '
                     f'(fraction of solid {material}; slider = z)', slice_label=[f'reference ({reference})'], vmin=0.0, vmax=1.0)
        slice_viewer(np.ascontiguousarray(fields.phantom_lr / np.float32(physics.density)), title=f'2/3  phantom handed to LEAP for S: {fields.voxel_lr_mm:.3f} mm '
                     f'(fraction of solid {material}; slider = z)', slice_label=['LEAP scatter phantom'], vmin=0.0, vmax=1.0)
        slice_viewer(*volumes, title='3/3  FDK: ' + ' | '.join(labels), slice_label=labels)
    return dict(volumes=volumes, labels=labels, report=report, picked=picked, match=match, fields=fields)
