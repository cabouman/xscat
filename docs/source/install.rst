============
Installation
============

xscat estimates scatter with the first-order scatter model of
`LEAP <https://github.com/LLNL/LEAP>`_ and the x-ray tables of
`XrayPhysics <https://github.com/kylechampley/XrayPhysics>`_, C++ libraries
(LEAP with CUDA) that are built from source, so the recommended installation is
the one-step script, which builds them into a new conda environment.

One-step installation
---------------------

Requirements:

* conda (Miniconda, Miniforge or Anaconda);
* git, make and a C/C++ compiler (Ubuntu: ``sudo apt install build-essential git``;
  macOS: ``xcode-select --install``);
* to run the scatter correction, Linux with an NVIDIA GPU whose driver supports
  CUDA 12.8 (``nvidia-smi`` shows the highest CUDA version the driver supports).

.. code-block:: bash

   git clone git@github.com:cabouman/xscat.git
   cd xscat
   bash dev_scripts/clean_install_all.sh
   conda activate xscat

The script deletes and recreates the ``xscat`` conda environment, then

1. installs CMake, PyTorch (on Linux a CUDA build that matches the GPU) and,
   on Linux, the CUDA toolkit from conda;
2. clones LEAP and XrayPhysics into ``build/third_party`` and builds them
   (LEAP takes several minutes; it compiles for every major GPU architecture);
3. installs xscat in editable mode with its Python dependencies
   (mbirtorch, scikit-image, ...) and builds this documentation.

The build itself needs no GPU.  On the Purdue clusters Gilbreth, Gautschi and
Negishi, run it in an interactive job, since LEAP's compilation is heavy for a
front-end.  The script loads the conda module there; a new shell needs the
same ``module load`` before ``conda activate xscat``.  On macOS, LEAP cannot
be built (it needs CUDA): the environment then has XrayPhysics and everything
else, which is enough to develop and test xscat but not to run the scatter
model.

The CUDA toolkit version, the PyTorch wheel index and the LEAP and XrayPhysics
versions are set in ``dev_scripts/config.sh``.  ``bash dev_scripts/install_leap.sh``
rebuilds only the compiled dependencies in the existing environment.

Python package only
-------------------

``pip install .`` installs xscat and its Python dependencies (numpy, scipy,
scikit-image, tifffile, torch, mbirtorch, h5py, pyyaml, matplotlib), but not
LEAP or XrayPhysics, which the scatter correction needs.  ``pip install ".[viewer]"``
adds mbirjax for the slice viewers at the end of a run.
