"""Preprocessing, synchronization and analysis of single-unit recordings with behavior.

Conventions used throughout the package:

* All times are in **seconds**, and every time-valued name ends in ``_s``.
* Neural data are arrays shaped ``(n_channels, n_samples)``.
* Functions take explicit arguments and never read file paths or globals.
"""

__version__ = "0.0.1"
