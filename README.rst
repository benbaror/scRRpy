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
         | |codecov| |coveralls|

.. |docs| image:: https://readthedocs.org/projects/scrrpy/badge/?style=flat
     :target: https://readthedocs.org/projects/scrrpy
     :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/benbaror/scRRpy.svg?branch=master
     :alt: Travis-CI Build Status
     :target: https://travis-ci.org/benbaror/scRRpy

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/benbaror/scrrpy?branch=master&svg=true
     :alt: AppVeyor Build Status
     :target: https://ci.appveyor.com/project/benbaror/scrrpy

.. |requires| image:: https://requires.io/github/benbaror/scRRpy/requirements.svg?branch=master
     :alt: Requirements Status
     :target: https://requires.io/github/benbaror/scRRpy/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/benbaror/scrrpy/coverage.svg?branch=master
     :alt: Coverage Status
     :target: https://codecov.io/github/benbaror/scrrpy
.. |coveralls| image:: https://coveralls.io/repos/github/benbaror/scRRpy/badge.svg?branch=master
     :target: https://coveralls.io/github/benbaror/scRRpy?branch=master



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
