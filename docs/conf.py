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

# Render "Attributes:" sections as an :ivar:/:vartype: field list rather than
# individual ".. attribute::" directives. Without this, Napoleon's generated
# attribute directives collide with autodoc's own introspection of dataclass
# fields (enabled via :undoc-members: in the .rst pages), producing
# "duplicate object description" warnings for every documented attribute.
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_sidebars = {
    # The '**' applies this sidebar configuration to all pages
    '**': [
        'globaltoc.html',  # Shows the full, static tree of your entire site
        'relations.html',  # Shows "next" and "previous" links
        'searchbox.html'   # Keeps the search bar at the bottom
    ]
}