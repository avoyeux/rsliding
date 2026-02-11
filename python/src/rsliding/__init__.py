"""
Contains the public python API for the rsliding library.
"""

# IMPORTs
import numpy as np

# IMPORTs local
from . import _bindings as _rust

# TYPE ANNOTATIONs
from typing import Literal
import numpy.typing as npt

# API public
__all__ = [
    "padding", "convolution", "sliding_mean", "sliding_median", "sliding_standard_deviation",
    "sliding_sigma_clipping",
]

# todo update this when all padding options are added inside the rust code.
# todo update docstring and/or decide for the contiguous


def padding(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
    ) -> npt.NDArray[np.float64]:
    """
    N-dimensional padding of a numpy float64 array given a kernel.
    NaN values are ignored.
    If no valid values inside a window, output value is NaN.

    Args:
        data (npt.NDArray[np.float64]): the input array to pad.
        kernel (npt.NDArray[np.float64]): the kernel (used to get the padding width).
        pad_mode (Literal["constant", "reflect", "replicate"]): the padding mode to use.
        pad_value (float): the padding value to use.

    Returns:
        npt.NDArray[np.float64]: the padded array.
    """
    # CONTIGUOUS
    data = np.ascontiguousarray(data)
    kernel = np.ascontiguousarray(kernel)
    return _rust.padding(data, kernel, pad_mode, pad_value)

def convolution(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]:
    """
    N-dimensional convolution of a numpy float64 array given a kernel.
    NaN values are ignored.
    If no valid values inside a window, output value is NaN.
    The kernel can contain weights.

    Args:
        data (npt.NDArray[np.float64]): the input array to convolve.
        kernel (npt.NDArray[np.float64]): the kernel used (can contain weights).
        pad_mode (Literal["constant", "reflect", "replicate"]): the padding mode to use.
        pad_value (float): the padding value to use.
        threads (int | None, optional): the number of threads to use. If None, uses the number of
            available CPU cores. Defaults to None.
    Returns:
        npt.NDArray[np.float64]: the convolution.
    """
    return _rust.convolution(data, kernel, pad_mode, pad_value, threads)

def sliding_mean(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]:
    """
    N-dimensional sliding mean of a numpy float64 array given a kernel.
    NaN values are ignored.
    If no valid values inside a window, output value is NaN.
    The kernel can contain weights.

    Args:
        data (npt.NDArray[np.float64]): the input array to get the sliding mean for.
        kernel (npt.NDArray[np.float64]): the kernel used (can contain weights).
        pad_mode (Literal["constant", "reflect", "replicate"]): the padding mode to use.
        pad_value (float): the padding value to use.
        threads (int | None, optional): the number of threads to use. If None, uses the number of
            available CPU cores. Defaults to None.

    Returns:
        npt.NDArray[np.float64]: the sliding mean.
    """
    return _rust.sliding_mean(data, kernel, pad_mode, pad_value, threads)

def sliding_median(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]:
    """
    N-dimensional sliding median of a numpy float64 array given a kernel.
    NaN values are ignored.
    If no valid values inside a window, output value is NaN.
    The kernel can contain weights.

    Args:
        data (npt.NDArray[np.float64]): the input array to get the sliding median for.
        kernel (npt.NDArray[np.float64]): the kernel used (can contain weights).
        pad_mode (Literal["constant", "reflect", "replicate"]): the padding mode to use.
        pad_value (float): the padding value to use.
        threads (int | None, optional): the number of threads to use. If None, uses the number of
            available CPU cores. Defaults to None.

    Returns:
        npt.NDArray[np.float64]: the sliding median.
    """
    return _rust.sliding_median(data, kernel, pad_mode, pad_value, threads)

def sliding_standard_deviation(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
        threads: int | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    N-dimensional sliding standard deviation of a numpy float64 array given a kernel.
    NaN values are ignored.
    If no valid values inside a window, output value is NaN.
    The kernel can contain weights.
    Also computes the sliding mean.

    Args:
        data (npt.NDArray[np.float64]): the input array to get the sliding standard deviation for.
        kernel (npt.NDArray[np.float64]): the kernel used (can contain weights).
        pad_mode (Literal["constant", "reflect", "replicate"]): the padding mode to use.
        pad_value (float): the padding value to use.
        threads (int | None, optional): the number of threads to use. If None, uses the number of
            available CPU cores. Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: the sliding standard deviation and
            the sliding mean.
    """
    return _rust.sliding_standard_deviation(data, kernel, pad_mode, pad_value, threads)

def sliding_sigma_clipping(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        center_mode: str,
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
        sigma_upper: float | None,
        sigma_lower: float | None,
        max_iterations: int | None,
        threads: int | None = None,
    ) -> npt.NDArray[np.float64]:
    """
    N-dimensional sliding sigma clipping of a numpy float64 array given a kernel
    todo update and change docstring

    Args:
        data (npt.NDArray[np.float64]): _description_
        kernel (npt.NDArray[np.float64]): _description_
        sigma_upper (float): _description_
        sigma_lower (float): _description_
        center_mode (str): _description_
        max_iterations (int): _description_
        pad_mode (Literal["constant", "reflect", "replicate"]): the padding mode to use.
        pad_value (float): _description_
        threads (int | None, optional): the number of threads to use. If None, uses the number of
            available CPU cores. Defaults to None.

    Returns:
        npt.NDArray[np.float64]: _description_
    """

    result = _rust.sliding_sigma_clipping(
        data,
        kernel,
        center_mode,
        pad_mode,
        pad_value,
        sigma_upper,
        sigma_lower,
        max_iterations,
        threads,
    )
    return result
