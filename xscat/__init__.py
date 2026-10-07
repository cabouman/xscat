"""xscat: scatter correction of cone-beam CT projections with a high-resolution model primary.

The package corrects measured transmission images for scattered radiation with the first-order scatter model of
LEAP (``leapct``) and a reference volume of the scanned part, normally a segmentation of a first reconstruction.
Two corrections share the same scatter estimate:

* :func:`xscat.correct.ratio_leap` -- LEAP's own ratio correction, the gain ``P / (P + S)`` evaluated on LEAP's
  down-sampled detector and up-sampled;
* :func:`xscat.correct.ratio_fullp` -- the same gain with the model primary ``P`` projected at full detector
  resolution and matched to the measurement (rigid re-alignment of the reference, system blur), so that the gain
  follows the edges of the part instead of smearing them.

See :mod:`xscat.pipeline` for the complete workflow and ``demo/demo_blade.py`` for a worked example.
"""

__version__ = '0.1.0'

from . import geometry, physics, reference, scatter, primary, correct, pipeline  # noqa: E402,F401

__all__ = ['geometry', 'physics', 'reference', 'scatter', 'primary', 'correct', 'pipeline']
