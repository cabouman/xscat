"""xscat: X-ray CT scatter correction.

xscat estimates and removes the scattered radiation in X-ray CT
measurements.  Scans enter as (sinogram, model) pairs produced by
mbirtorch preprocessing, and the corrected sinogram goes back to
mbirtorch for reconstruction.
"""

__version__ = '0.0.1'

__all__ = []
