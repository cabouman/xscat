"""Cone-beam geometry, reconstruction grid and the LEAP model objects built from them.

All volumes are in LEAP's ``(z, y, x)`` order and all projection stacks in ``(view, row, col)`` order.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConeBeamGeometry:
    """Circular cone-beam geometry of a scan.

    Attributes:
        num_views: Number of projections.
        num_rows: Detector rows.
        num_cols: Detector columns.
        pixel_height: Detector pixel height (mm).
        pixel_width: Detector pixel width (mm).
        center_row: Row of the central ray (LEAP convention, pixel units, may be fractional).
        center_col: Column of the central ray.
        angles_deg: Projection angles (degrees), shape ``(num_views,)``.
        sod: Source to rotation-axis distance (mm).
        sdd: Source to detector distance (mm).
    """
    num_views: int
    num_rows: int
    num_cols: int
    pixel_height: float
    pixel_width: float
    center_row: float
    center_col: float
    angles_deg: np.ndarray
    sod: float
    sdd: float

    @property
    def magnification(self):
        """Geometric magnification ``sdd / sod``."""
        return self.sdd / self.sod

    def subset(self, views):
        """The same geometry restricted to the given view indices."""
        views = np.asarray(views, dtype=int)
        return ConeBeamGeometry(len(views), self.num_rows, self.num_cols, self.pixel_height, self.pixel_width, self.center_row,
                                self.center_col, np.ascontiguousarray(self.angles_deg[views], dtype=np.float32), self.sod, self.sdd)

    def binned(self, factor):
        """The geometry of the detector block-binned by ``factor`` (LEAP's ``down_sample_projections`` convention)."""
        f = int(factor)
        if f <= 1:
            return self
        return ConeBeamGeometry(self.num_views, self.num_rows // f, self.num_cols // f, self.pixel_height * f, self.pixel_width * f,
                                (self.center_row + 0.5) / f - 0.5, (self.center_col + 0.5) / f - 0.5, self.angles_deg, self.sod, self.sdd)


@dataclass
class VolumeGrid:
    """A reconstruction grid: shape ``(z, y, x)``, cubic voxel size (mm) and LEAP volume offsets ``(x, y, z)`` (mm)."""
    shape_zyx: tuple
    voxel_mm: float
    offsets_xyz: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0))

    def binned(self, factor):
        """The grid block-binned by ``factor`` (trailing remainder dropped), centred like the original grid."""
        f = int(factor)
        if f <= 1:
            return self
        off = binned_grid_offsets(self.shape_zyx, f, self.voxel_mm)
        return VolumeGrid(tuple(n // f for n in self.shape_zyx), self.voxel_mm * f, tuple(o + b for o, b in zip(self.offsets_xyz, off)))

    def shifted(self, delta_xyz_mm):
        """The same grid moved by ``delta_xyz_mm`` (mm)."""
        return VolumeGrid(self.shape_zyx, self.voxel_mm, tuple(o + float(d) for o, d in zip(self.offsets_xyz, delta_xyz_mm)))


def binned_grid_offsets(shape_zyx, factor, voxel_mm):
    """LEAP ``(offsetX, offsetY, offsetZ)`` of the block-mean grid of a centred volume (trailing remainder dropped).

    Args:
        shape_zyx: Shape of the full grid.
        factor: Binning factor.
        voxel_mm: Voxel size of the full grid (mm).

    Returns:
        Tuple ``(x, y, z)`` of offsets in mm that keep the binned grid centred on the full one.
    """
    f = int(factor)
    off = [((f * (n // f)) - n) / 2.0 * float(voxel_mm) for n in shape_zyx]          # z, y, x
    return off[2], off[1], off[0]


def block_mean_views(stack_vrc, factor):
    """Block mean of a ``(view, row, col)`` stack over rows and columns (trailing remainder dropped)."""
    f = int(factor)
    if f <= 1:
        return np.asarray(stack_vrc, dtype=np.float32)
    v, r, c = stack_vrc.shape
    a = np.asarray(stack_vrc[:, :r // f * f, :c // f * f], dtype=np.float32)
    return a.reshape(v, r // f, f, c // f, f).mean(axis=(2, 4), dtype=np.float32)


def block_mean_volume(vol_zyx, factor):
    """Block mean of a 3-D array by an integer factor along every axis (trailing remainder dropped)."""
    f = int(factor)
    if f <= 1:
        return np.asarray(vol_zyx, dtype=np.float32)
    nz, ny, nx = (s // f for s in vol_zyx.shape)
    v = np.asarray(vol_zyx[:nz * f, :ny * f, :nx * f], dtype=np.float32)
    return v.reshape(nz, f, ny, f, nx, f).mean(axis=(1, 3, 5), dtype=np.float32)


def configure_gpus(verbose=True):
    """Logical CUDA device indices for LEAP, from ``CUDA_VISIBLE_DEVICES`` or ``nvidia-smi``.

    Returns:
        List of logical device indices (``[0]`` when nothing can be detected).
    """
    import os
    import subprocess
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    source = 'pre-set by the environment'
    if visible is None or not str(visible).strip():
        try:
            result = subprocess.run(['nvidia-smi', '-L'], check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
            count = len([ln for ln in result.stdout.splitlines() if ln.strip().startswith('GPU ')]) if result.returncode == 0 else 0
        except Exception:
            count = 0
        if count > 0:
            visible = ','.join(str(i) for i in range(count))
            os.environ['CUDA_VISIBLE_DEVICES'] = visible
            source = f'auto-detected {count} NVIDIA GPU(s)'
        else:
            visible = None
            source = 'not set; no GPU detected'
    tokens = [t.strip() for t in str(visible).split(',') if t.strip() and t.strip() != '-1'] if visible else []
    gpus = list(range(len(tokens))) if tokens else [0]
    if verbose:
        print(f'[gpu] CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")} ({source}); LEAP logical indices {gpus}')
    return gpus


def leap_model(geometry, grid, gpus=None):
    """A LEAP ``tomographicModels`` object with the given cone-beam geometry and volume grid, in ``(z, y, x)`` order.

    Args:
        geometry: :class:`ConeBeamGeometry`.
        grid: :class:`VolumeGrid`.
        gpus: Logical CUDA device indices (default: :func:`configure_gpus` without output).

    Returns:
        The configured ``leapct.tomographicModels`` instance.
    """
    from leapctype import tomographicModels
    model = tomographicModels()
    set_order = getattr(model, 'set_volumeDimensionOrder', None)
    if set_order is not None and set_order(1) is False:
        raise RuntimeError('LEAP refused the (z, y, x) volume order')
    model.set_gpus(configure_gpus(verbose=False) if gpus is None else gpus)
    g = geometry
    model.set_conebeam(g.num_views, g.num_rows, g.num_cols, g.pixel_height, g.pixel_width, g.center_row, g.center_col,
                       np.ascontiguousarray(g.angles_deg, dtype=np.float32), g.sod, g.sdd)
    nz, ny, nx = grid.shape_zyx
    ox, oy, oz = grid.offsets_xyz
    if model.set_volume(numX=int(nx), numY=int(ny), numZ=int(nz), voxelWidth=float(grid.voxel_mm), voxelHeight=float(grid.voxel_mm),
                        offsetX=float(ox), offsetY=float(oy), offsetZ=float(oz)) is False:
        raise RuntimeError('LEAP rejected the volume grid')
    return model
