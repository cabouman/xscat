# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'xscat'
copyright = '2026, XScat development team'
author = 'XScat development team'

import xscat
release = xscat.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'sphinxext.opengraph',
]

templates_path = ['_templates']
exclude_patterns = []

# Google-style docstrings only.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_book_theme'
html_theme_options = {
    'repository_url': 'https://github.com/cabouman/xscat',
    'use_repository_button': True,
}
html_title = 'xscat'
html_static_path = ['_static']

# Open Graph / social link preview.
ogp_site_url = 'https://xscat.readthedocs.io/en/latest/'
ogp_type = 'website'
ogp_enable_meta_description = False
ogp_description_length = 0
ogp_social_cards = {'enable': False}
