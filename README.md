# xscat

xscat estimates and removes the scattered radiation in X-ray CT
measurements.

xscat is built on [mbirtorch](https://github.com/cabouman/mbirtorch).
Scans enter as a sinogram plus a tomography model, the pair produced
by mbirtorch preprocessing, and the corrected sinogram goes back to
mbirtorch for reconstruction.

Full documentation: [xscat.readthedocs.io](https://xscat.readthedocs.io).

## Install

```bash
git clone git@github.com:cabouman/xscat.git
cd xscat
pip install .
```

This installs mbirtorch and the other dependencies automatically.

## Citation

Please cite the software when referencing this package.

```bibtex
@misc{xscat,
  title = {{X}-ray {CT} {S}catter {C}orrection},
  author = {Jingsong Lin and Obaidullah Rahman and Amirkoushyar Ziabari and Gregery T. Buzzard and Charles A. Bouman},
  howpublished = {Software library available from \url{https://github.com/cabouman/xscat}},
  note = {Version 0.0.1},
  year = 2026
}
```

GitHub's "Cite this repository" button on the repository page generates the
citation from `CITATION.cff`.
