.. _QuickStart:

===========
Quick Start
===========

The script below corrects one scan for scatter and reconstructs it.

.. code-block:: python

    import mbirtorch.preprocess as mtp
    import xscat

    # Preprocess the scan: scanner-specific, returns (sinogram, model).
    sino, model = mtp.zeiss.get_sino_and_model('scan.txrm')

    # Scatter correction.  (API to be designed.)

    # Reconstruct the corrected sinogram.
    recon, _ = model.recon(sino)
