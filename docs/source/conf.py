# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

def setup(app):
    app.add_css_file("css/customWidth.css")


project = 'PyFADe'
copyright = '2026, Heitor Lopes'
author = 'Heitor Lopes'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0,os.path.abspath("../../src"))

project = "PyFADe"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_code_tabs",
    "breathe",
]

autodoc_member_order = 'bysource'

templates_path = ['_templates']
exclude_patterns = []

breathe_projects = {
    "finelpy": "../build/doxygen/xml"
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'navigation_depth': 4,
    'titles_only': False,

}

html_context = {
  'display_github': True,
  'github_user': 'RL2-SP3',
  'github_repo': 'pyfade',
  'github_version': 'main',
  'conf_py_path': '/docs/source/',
}

html_static_path = ['_static']

