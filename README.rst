========
Overview
========

.. start-badges

.. end-badges

Calculation of Scalar Resonant Relaxation

* Free software: BSD license

Installation
============

::

    pip install scrrpy

Documentation
=============

.. https://scrrpy.readthedocs.io/

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
