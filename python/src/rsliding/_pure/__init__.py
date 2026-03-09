"""
Directory contains code to perform n-dimensional sliding mean, median, standard deviation,
and sigma clipping on numpy arrays (with/without NaNs) given a kernel (with/without weights).
"""

from .convolution import BorderType, Convolution, Padding
from .mode import SlidingMean, SlidingMedian
from .standard_deviation import SlidingStandardDeviation
from .sigma_clipping import SlidingSigmaClipping
