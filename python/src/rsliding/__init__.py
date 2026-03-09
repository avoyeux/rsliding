"""
Contains the public python API for the rsliding library.
"""

try:
    # IMPORTs rust local
    from .padding import Padding
    from .convolution import Convolution
    from .sliding_mean import SlidingMean
    from .sliding_median import SlidingMedian
    from .sliding_standard_deviation import SlidingStandardDeviation
    from .sliding_sigma_clipping import SlidingSigmaClipping
except ImportError:
    print(
        "\033[1;31mrsliding correct rust binary not found."
        "Importing the corresponding full Python fallback\033[0m",
        flush=True,
    )

    # IMPORTs python local
    from ._pure import (
        Padding, Convolution, SlidingMean, SlidingMedian, SlidingStandardDeviation,
        SlidingSigmaClipping,
    )

# EXPORT
__all__ = [
    "Padding",
    "Convolution",
    "SlidingMean",
    "SlidingMedian",
    "SlidingStandardDeviation",
    "SlidingSigmaClipping",
]
