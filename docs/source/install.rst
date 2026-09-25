============
Installation
============

xscat is installed from source:

.. code-block:: bash

   git clone git@github.com:cabouman/xscat.git
   cd xscat
   pip install .

This installs the Python dependencies (numpy, scipy, torch, mbirtorch,
h5py, pyyaml, matplotlib) automatically.  mbirtorch reads the scanner
data and provides the tomography models.

To verify the installation, run the test suite:

.. code-block:: bash

   pip install pytest
   pytest tests/
