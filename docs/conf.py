# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Add project root directory to path
import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Points to the project root directory

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Ramsey Number Search'
copyright = '2026, Ryan Wheeler + ChatGPT'
author = 'Ryan Wheeler + ChatGPT'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Automatically core extracts docstrings
    'sphinx.ext.napoleon',     # Parses Google and NumPy style docstrings
    'sphinx.ext.viewcode',     # Adds links to the original source code lines
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
