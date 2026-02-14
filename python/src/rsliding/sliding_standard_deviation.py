"""
Code to compute the sliding standard deviation of a given ndarray data with NaN values and a
weighted kernel. Also computes the sliding mean.
"""
from __future__ import annotations

# IMPORTs
import numpy as np

# IMPORTs local
from . import _bindings as _rust
from .utils import KernelType, BorderType, BaseCheck

# TYPE ANNOTATIONs
import numpy.typing as npt

# TYPE ANNOTATIONs
__all__ = ["SlidingStandardDeviation"]



class SlidingStandardDeviation(BaseCheck):
    """
    Does a sliding standard deviation of data with NaN values and a kernel with weights.
    Also computes the sliding mean.
    NaN values are ignored. 
    If no valid data in the given window, the output is set to NaN.
    Use the 'standard_deviation' property to access the sliding standard deviation result.
    Use the 'mean' property to access the sliding mean result.
    """

    def __init__(
            self,
            data: npt.NDArray[np.float64],
            kernel: KernelType,
            borders: BorderType,
            pad_value: float = 0.,
            force_contiguous: bool = True,
            threads: int | None = 1,
            neumaier: bool = False,
        ) -> None:
        """
        Does a sliding standard deviation of data with NaN values and a kernel with weights.
        Also computes the sliding mean.
        NaN values are ignored.
        If no valid data in the given window, the output is set to NaN.
        Use the 'standard_deviation' property to access the sliding standard deviation result.
        Use the 'mean' property to access the sliding mean result.

        Args:
            data (npt.NDArray[np.float64]): the data to compute the sliding standard deviation on.
            kernel (KernelType): the kernel to use for the sliding standard deviation.
            borders (BorderType, optional): the border type to use for padding. If None, does the
                same operation than setting the borders to 'constant' and pad_value to np.nan (i.e.
                adaptative kernel sizes). Defaults to 'reflect'.
            pad_value (float, optional): the value to use for padding when borders is 'constant'.
                Defaults to 0.
            force_contiguous (bool, optional): whether to force the data and kernel to be
                contiguous in memory. Defaults to True.
            threads (int | None, optional): the number of threads to use in the sliding operation.
                If set to None, uses all the available logical cores. Defaults to 1.
            neumaier (bool, optional): whether to use the Neumaier algorithm for floating point
                summation. Never tested a case where it would make a difference, but it is more
                stable numerically. More expensive computationally. Defaults to False.
        """

        self._data = data
        self._kernel = self._check_kernel(kernel)
        self._borders = borders
        self._pad_value = pad_value
        self._threads = threads
        self._neumaier = neumaier

        if force_contiguous: self._make_contiguous()

        # RUN
        self._standard_deviation, self._mean = self._sliding_standard_deviation()

    @property
    def standard_deviation(self) -> npt.NDArray[np.float64]:
        """
        The sliding standard deviation result.

        Returns:
            npt.NDArray[np.float64]: the sliding standard deviation result.
        """
        return self._standard_deviation

    @property
    def mean(self) -> npt.NDArray[np.float64]:
        """
        The sliding mean result.

        Returns:
            npt.NDArray[np.float64]: the sliding mean result.
        """
        return self._mean

    def _sliding_standard_deviation(
            self,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Computes the sliding standard deviation using rust.
        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: the sliding standard deviation
                and mean results.
        """

        pad_mode = self._borders if self._borders is not None else 'constant'
        pad_value = self._pad_value if self._borders is not None else np.nan
        standard_deviation, mean  = _rust.sliding_standard_deviation(
            self._data,
            self._kernel,
            pad_mode,#type:ignore
            pad_value,
            self._neumaier,
            self._threads,
        )
        return standard_deviation, mean
