xscat: X-ray CT Scatter Correction
==================================

xscat estimates and removes the scattered radiation in X-ray CT
measurements.

Built on mbirtorch
------------------

xscat is built on `mbirtorch <https://github.com/cabouman/mbirtorch>`_.
Scans enter as a sinogram plus a tomography model, the pair produced
by mbirtorch preprocessing, and the corrected sinogram goes back to
mbirtorch for reconstruction.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   overview
   install
   quick_start
   usr_api
   credits
