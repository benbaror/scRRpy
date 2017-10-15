========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - |version| |downloads| |wheel| |supported-versions| |supported-implementations|

.. |docs| image:: https://readthedocs.org/projects/scrrpy/badge/?style=flat
    :target: https://readthedocs.org/projects/scrrpy
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/benbaror/scrrpy.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/benbaror/scrrpy

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/benbaror/scrrpy?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/benbaror/scrrpy

.. |requires| image:: https://requires.io/github/benbaror/scrrpy/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/benbaror/scrrpy/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/benbaror/scrrpy/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/benbaror/scrrpy

.. |version| image:: https://img.shields.io/pypi/v/scrrpy.svg?style=flat
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/scrrpy

.. |downloads| image:: https://img.shields.io/pypi/dm/scrrpy.svg?style=flat
    :alt: PyPI Package monthly downloads
    :target: https://pypi.python.org/pypi/scrrpy

.. |wheel| image:: https://img.shields.io/pypi/wheel/scrrpy.svg?style=flat
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/scrrpy

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/scrrpy.svg?style=flat
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/scrrpy

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/scrrpy.svg?style=flat
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/scrrpy


.. end-badges

Calculation of Scalar Resonant Relaxation

* Free software: BSD license

Installation
============

::

    pip install scrrpy

Documentation
=============

https://scrrpy.readthedocs.io/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
