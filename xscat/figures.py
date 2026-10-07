"""Diagnostic figures of the pipeline (matplotlib, Agg backend; written as PNG files)."""
import os

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


class Figures:
    """Writes numbered PNG figures ``<prefix>_<name>.png`` into ``out_dir``."""

    def __init__(self, out_dir, prefix, dpi=70, log=print):
        self.out_dir, self.prefix, self.dpi, self.log = out_dir, prefix, dpi, log
        os.makedirs(out_dir, exist_ok=True)
        self.paths = []

    def save(self, fig, name):
        plt = _plt()
        fig.tight_layout()
        path = os.path.join(self.out_dir, f'{self.prefix}_{name}.png')
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        self.paths.append(path)
        self.log(f'   figure {path}')
        return path

    def reference(self, name, ref_fraction, phantom_fraction, voxel_mm, voxel_lr_mm, title):
        """Centre slices of the reference (fraction of solid) and of LEAP's block-averaged scatter phantom."""
        plt = _plt()
        fig, axs = plt.subplots(2, 3, figsize=(15, 9))
        for row, (vol, vox, lab) in enumerate([(ref_fraction, voxel_mm, 'reference -> P_sim'), (phantom_fraction, voxel_lr_mm, 'LEAP scatter phantom -> S')]):
            z, y, x = (s // 2 for s in vol.shape)
            for col, (img, ax_lab) in enumerate([(vol[z], 'axial (y, x)'), (vol[:, y], 'coronal (z, x)'), (vol[:, :, x], 'sagittal (z, y)')]):
                axs[row, col].imshow(img, cmap='gray', vmin=0, vmax=1)
                axs[row, col].set_title(f'{lab} {vol.shape} @ {vox:.3f} mm: {ax_lab}')
                axs[row, col].axis('off')
        fig.suptitle(title)
        return self.save(fig, name)

    def lowres(self, name, attenuation_views, s_lr, p_lr, gain_lr, views):
        """Measured attenuation, LEAP's S, our P/(P+S) and LEAP's own gain on the coarse grid for a few views."""
        plt = _plt()
        from .scatter import ratio_gain
        ncol = 4 if gain_lr is not None else 3
        fig, axs = plt.subplots(len(views), ncol, figsize=(5 * ncol, 4.5 * len(views)))
        axs = np.atleast_2d(axs)
        for i, v in enumerate(views):
            axs[i, 0].imshow(attenuation_views[i], cmap='gray', vmin=0, vmax=np.percentile(attenuation_views[i], 99.5))
            axs[i, 0].set_title(f'view {int(v)}: measured attenuation'); axs[i, 0].axis('off')
            im = axs[i, 1].imshow(s_lr[v], cmap='viridis', vmin=0, vmax=float(s_lr[list(views)].max())); axs[i, 1].set_title('LEAP S / air (coarse grid)'); axs[i, 1].axis('off')
            plt.colorbar(im, ax=axs[i, 1], fraction=0.046)
            g = ratio_gain(p_lr[v], s_lr[v])
            im = axs[i, 2].imshow(g, cmap='magma', vmin=float(g.min()), vmax=1); axs[i, 2].set_title('P / (P + S), coarse grid'); axs[i, 2].axis('off')
            plt.colorbar(im, ax=axs[i, 2], fraction=0.046)
            if gain_lr is not None:
                im = axs[i, 3].imshow(gain_lr[v], cmap='magma', vmin=float(g.min()), vmax=1); axs[i, 3].set_title("LEAP's own gain (coarse grid)"); axs[i, 3].axis('off')
                plt.colorbar(im, ax=axs[i, 3], fraction=0.046)
        return self.save(fig, name)

    def edge_match(self, name, match, k, view, gain_leap_row=None):
        """Residual M - (P + S) before / after the matching, an edge profile, the gains along it and the blur fit."""
        plt = _plt()
        from .primary import blur_views
        from .scatter import ratio_gain
        ex = match.extras
        meas, s_c, p_bare, p_al, band = ex['meas'][k], ex['scatter'][k], ex['p_bare'][k], ex['p_aligned'][k], ex['bands'][k]
        p_fin = blur_views(p_al, match.psf_px)
        if not band.any():
            return None
        rows_b, cols_b = np.where(band.any(axis=1))[0], np.where(band.any(axis=0))[0]
        r0, r1 = max(rows_b[0] - 20, 0), min(rows_b[-1] + 21, meas.shape[0])
        c0, c1 = max(cols_b[0] - 20, 0), min(cols_b[-1] + 21, meas.shape[1])
        row = int(rows_b[np.argmax(band[rows_b].sum(axis=1))])
        res_b, res_a = meas - (p_bare + s_c), meas - (p_fin + s_c)
        lim = float(np.percentile(np.abs(res_b[band]), 99))
        fig, axs = plt.subplots(2, 3, figsize=(18, 11))
        axs[0, 0].imshow(meas[r0:r1, c0:c1], cmap='gray', vmin=0, vmax=1); axs[0, 0].set_title(f'view {view}: measured transmission'); axs[0, 0].axis('off')
        axs[0, 0].axhline(row - r0, color='r', lw=0.8)
        for ax, res, lab in [(axs[0, 1], res_b, 'as registered, sharp'), (axs[0, 2], res_a, f're-aligned + blur {match.psf_px:.2f} px')]:
            im = ax.imshow(res[r0:r1, c0:c1], cmap='seismic', vmin=-lim, vmax=lim)
            ax.set_title(f'M - (P + S), {lab}: rms on the band {np.sqrt(np.mean(res[band] ** 2)):.4f}'); ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        x = np.arange(c0, c1)
        axs[1, 0].plot(x, meas[row, c0:c1], 'k', label='measured M')
        axs[1, 0].plot(x, (p_bare + s_c)[row, c0:c1], label='P + S, as registered (sharp)')
        axs[1, 0].plot(x, (p_fin + s_c)[row, c0:c1], label=f'P + S, re-aligned + blur {match.psf_px:.2f} px')
        axs[1, 0].set_title(f'row {row}: transmission across the part'); axs[1, 0].grid(alpha=.3); axs[1, 0].legend(fontsize=8)
        axs[1, 1].plot(x, ratio_gain(p_bare, s_c)[row, c0:c1], label='gain, sharp primary')
        axs[1, 1].plot(x, ratio_gain(p_fin, s_c)[row, c0:c1], label='gain, matched primary (ratio-fullp)')
        if gain_leap_row is not None:
            axs[1, 1].plot(x, gain_leap_row[c0:c1], '--', label="LEAP's coarse gain (ratio-leap)")
        axs[1, 1].set_ylim(0, 1.05); axs[1, 1].set_title('gain P / (P + S) along the same row'); axs[1, 1].grid(alpha=.3); axs[1, 1].legend(fontsize=8)
        for kk, c in enumerate(match.psf_costs):
            if np.all(np.isfinite(c)):
                axs[1, 2].plot(match.psf_grid, c / max(c.max(), 1e-30), alpha=.6, label=f'view {int(match.calib_views[kk])}' if kk < 8 else None)
        axs[1, 2].axvline(match.psf_px, color='k', lw=1, label=f'median sigma {match.psf_px:.2f} px')
        axs[1, 2].set_xlabel('blur sigma (detector px)'); axs[1, 2].set_ylabel('edge residual (normalised)'); axs[1, 2].legend(fontsize=7); axs[1, 2].grid(alpha=.3)
        off = match.offset_mm
        axs[1, 2].set_title('blur fit per calibration view' + (f'; rigid offset ({off[0]:+.3f}, {off[1]:+.3f}, {off[2]:+.3f}) mm' if match.shift_applied else '; no rigid offset applied'))
        return self.save(fig, name)

    def gains(self, name, attenuation_view, view, gains_by_mode):
        """Full-resolution gains of the modes at one view, with centre-row profiles."""
        plt = _plt()
        modes = list(gains_by_mode)
        fig, axs = plt.subplots(2, len(modes) + 1, figsize=(6 * (len(modes) + 1), 11))
        axs = np.atleast_2d(axs)
        row = attenuation_view.shape[0] // 2
        axs[0, 0].imshow(attenuation_view, cmap='gray', vmin=0, vmax=np.percentile(attenuation_view, 99.5)); axs[0, 0].set_title(f'view {view}: measured attenuation'); axs[0, 0].axis('off')
        vmin = float(min(g.min() for g in gains_by_mode.values()))
        for j, m in enumerate(modes):
            g = gains_by_mode[m]
            axs[0, j + 1].imshow(g, cmap='magma', vmin=vmin, vmax=1); axs[0, j + 1].set_title(f'{m}: gain at full resolution'); axs[0, j + 1].axis('off')
            axs[1, 0].plot(g[row], label=m)
            axs[1, j + 1].plot(-np.log(np.maximum(g, 1e-12))[row], label='-ln gain (attenuation removed)')
            axs[1, j + 1].plot(attenuation_view[row], 'k', alpha=.4, label='raw attenuation'); axs[1, j + 1].grid(alpha=.3); axs[1, j + 1].legend(fontsize=8); axs[1, j + 1].set_title(m)
        axs[1, 0].set_ylim(0, 1.05); axs[1, 0].grid(alpha=.3); axs[1, 0].legend(fontsize=8); axs[1, 0].set_title('gain profiles, centre row')
        return self.save(fig, name)

    def corrected(self, name, attenuation_view, view, corrected_by_mode):
        """Corrected attenuation (before beam hardening) of the modes at one view."""
        plt = _plt()
        modes = list(corrected_by_mode)
        fig, axs = plt.subplots(2, len(modes) + 1, figsize=(5 * (len(modes) + 1), 10))
        vmax = float(np.percentile(attenuation_view, 99.5))
        row = attenuation_view.shape[0] // 2
        axs[0, 0].imshow(attenuation_view, cmap='gray', vmin=0, vmax=vmax); axs[0, 0].set_title(f'view {view}: raw attenuation'); axs[0, 0].axis('off')
        axs[1, 0].plot(attenuation_view[row], 'k', label='raw, centre row')
        for j, m in enumerate(modes):
            a = corrected_by_mode[m]
            axs[0, j + 1].imshow(a, cmap='gray', vmin=0, vmax=vmax); axs[0, j + 1].set_title(f'{m} (before BH)'); axs[0, j + 1].axis('off')
            axs[1, j + 1].plot(attenuation_view[row], 'k', alpha=.4, label='raw'); axs[1, j + 1].plot(a[row], label=f'{m}, centre row')
            axs[1, j + 1].grid(alpha=.3); axs[1, j + 1].legend(fontsize=8)
            axs[1, 0].plot(a[row], alpha=.7, label=m)
        axs[1, 0].grid(alpha=.3); axs[1, 0].legend(fontsize=7)
        return self.save(fig, name)

    def linearity(self, name, centres, curves, mu_expected):
        """Median attenuation after beam hardening against the reference path length, per mode."""
        plt = _plt()
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for label, med in curves.items():
            ax[0].plot(centres, med, 'o-', ms=3, label=label); ax[1].plot(centres, med / np.maximum(centres, 1e-3), 'o-', ms=3, label=label)
        ax[1].axhline(mu_expected, color='k', lw=0.8, label=f'expected {mu_expected:.3f} /mm')
        ax[0].set_xlabel('reference path length (mm)'); ax[0].set_ylabel('median attenuation after BH'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
        ax[1].set_xlabel('reference path length (mm)'); ax[1].set_ylabel('attenuation / path (/mm)'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
        return self.save(fig, name)

    def volumes(self, name, volumes, labels, vmax):
        """Centre slices (axial, coronal, sagittal) of the FDK volumes side by side."""
        plt = _plt()
        n = len(volumes)
        fig, axs = plt.subplots(3, n, figsize=(5 * n, 14))
        axs = np.atleast_2d(axs)
        if axs.shape[0] != 3:
            axs = axs.T
        z, y, x = (s // 2 for s in volumes[0].shape)
        for j, (vol, lab) in enumerate(zip(volumes, labels)):
            for i, (img, ax_lab) in enumerate([(vol[z], 'axial'), (vol[:, y], 'coronal'), (vol[:, :, x], 'sagittal')]):
                axs[i, j].imshow(img, cmap='gray', vmin=-0.05 * vmax, vmax=vmax); axs[i, j].set_title(f'{lab}: {ax_lab}'); axs[i, j].axis('off')
        return self.save(fig, name)
