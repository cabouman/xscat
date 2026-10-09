"""Scatter correction of the Metrotom blade scan: original LEAP ratio against the high-resolution matched primary.

The script reads the scan, reconstructs it once (FDK), segments that reconstruction into the reference volume (default),
runs LEAP's first-order scatter model on it, and corrects the projections in two ways that share the same scatter
estimate:

    ratio-leap    LEAP's correction as shipped: the gain P / (P + S) on LEAP's down-sampled detector, up-sampled
    ratio-fullp   the same S, but P projected at full detector resolution from the reference, re-aligned to the measured
                  silhouette and blurred by the fitted system blur (xscat.primary.match_primary)

The FDK volumes of raw, raw+BH and the two corrections are written as figures and opened in the slice viewer.

    python demo/demo_blade.py --data-root /path/to/blade_data_with_cad --part bottom
    python demo/demo_blade.py --data-root ... --part bottom --view-subsample 5 --det-subsample 2 --fdk-binning 2    # quick trial
    python demo/demo_blade.py --data-root ... --part bottom --reference cad                                        # the registered CAD instead

Run in an environment with LEAP (leapct) and xrayphysics; mbirjax is optional (viewer).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.dirname(HERE) not in sys.path:
    sys.path.insert(0, os.path.dirname(HERE))

from xscat import metrotom, pipeline  # noqa: E402

PART_DIRS = {'bottom': ('Blade_bottom', 'CAD_Reg_bot.tiff'), 'top': ('Blade_top', 'CAD_Reg_top.tiff')}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group('data')
    g.add_argument('--data-root', required=True, help='folder holding Blade_bottom/ and Blade_top/')
    g.add_argument('--part', choices=sorted(PART_DIRS), default='bottom')
    g.add_argument('--scan-dir', default=None, help='scan folder (default: found under the part folder)')
    g.add_argument('--out-dir', default=None, help='output folder (default: demo/output/<part>)')
    g.add_argument('--view-subsample', type=int, default=1)
    g.add_argument('--det-subsample', type=int, default=1)
    g.add_argument('--fdk-binning', type=int, default=1)
    g = p.add_argument_group('reference')
    g.add_argument('--reference', choices=('seg', 'cad'), default='seg', help='segmentation of the reconstruction (default) or the registered CAD')
    g.add_argument('--cad', default=None, help='CAD tiff (default: the part folder\'s file)')
    g.add_argument('--cad-orientation', default='auto')
    g.add_argument('--ref-binning', type=int, default=2, help='grid binning of the reference')
    g.add_argument('--seg-low-fraction', type=float, default=0.5)
    g.add_argument('--seg-smooth-sigma', type=float, default=1.0)
    g.add_argument('--seg-opening-mm', type=float, default=0.3)
    g = p.add_argument_group('physics (Metrotom blade scan)')
    g.add_argument('--material', default='Ni')
    g.add_argument('--density', type=float, default=8.902e-3, help='g/mm^3')
    g.add_argument('--kvp', type=float, default=200.0)
    g.add_argument('--take-off-deg', type=float, default=11.0)
    g.add_argument('--filters', default='Sn:1.0')
    g.add_argument('--detector', default='CsI:0.6')
    g = p.add_argument_group('correction')
    g.add_argument('--modes', default='ratio-leap,ratio-fullp', help="comma list of ratio-leap, ratio-fullp, ratio-fullp-bare")
    g.add_argument('--leap-detector-side', type=int, default=204)
    g.add_argument('--leap-volume-side', type=int, default=199)
    g.add_argument('--match-shift', choices=('rigid', 'none'), default='rigid')
    g.add_argument('--match-psf', default='auto', help="'auto', 'none' or a blur in pixels")
    g.add_argument('--match-band-px', type=int, default=8)
    g.add_argument('--match-max-shift-px', type=float, default=3.0)
    g.add_argument('--calib-views', type=int, default=12)
    g.add_argument('--min-gain', type=float, default=0.0)
    g.add_argument('--chunk-views', type=int, default=16)
    g = p.add_argument_group('output')
    g.add_argument('--num-picks', type=int, default=3)
    g.add_argument('--save-npy', action='store_true', help='also save the corrected projections and FDK volumes')
    g.add_argument('--no-viewer', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    part_dir, cad_name = PART_DIRS[args.part]
    scan_dir = args.scan_dir or metrotom.find_scan_dir(os.path.join(args.data_root, part_dir))
    cad_path = args.cad or os.path.join(args.data_root, part_dir, cad_name)
    out_dir = args.out_dir or os.path.join(HERE, 'output', args.part)
    return pipeline.run(scan_dir, out_dir, material=args.material, density_g_mm3=args.density, kvp=args.kvp, take_off_deg=args.take_off_deg,
                        filters=args.filters, detector=args.detector, modes=[m.strip() for m in args.modes.split(',') if m.strip()],
                        reference=args.reference, cad_path=cad_path, cad_orientation=args.cad_orientation, view_subsample=args.view_subsample,
                        det_subsample=args.det_subsample, fdk_binning=args.fdk_binning, ref_binning=args.ref_binning, seg_low_fraction=args.seg_low_fraction,
                        seg_smooth_sigma=args.seg_smooth_sigma, seg_opening_mm=args.seg_opening_mm, leap_detector_side=args.leap_detector_side,
                        leap_volume_side=args.leap_volume_side, match_shift=args.match_shift, match_psf=args.match_psf, match_band_px=args.match_band_px,
                        match_max_shift_px=args.match_max_shift_px, calib_views=args.calib_views, min_gain=args.min_gain, chunk=args.chunk_views,
                        num_picks=args.num_picks, tag=f'blade_{args.part}_{args.reference}', save_npy=args.save_npy, viewer=not args.no_viewer)


if __name__ == '__main__':
    main()
