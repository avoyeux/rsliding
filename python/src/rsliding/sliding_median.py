"""
Code to compute the sliding median of a given ndarray data with NaN values and a weighted kernel.
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
__all__ = ["SlidingMedian"]



class SlidingMedian(BaseCheck):
    """
    Does a sliding median of data with NaN values and a kernel with weights.
    NaN values are ignored.
    If no valid data in the given window, the output is set to NaN.
    Use the 'median' property to access the sliding median result.
    """

    def __init__(
            self,
            data: npt.NDArray[np.float64],
            kernel: KernelType,
            borders: BorderType,
            pad_value: float = 0.,
            force_contiguous: bool = True,
            threads: int | None = 1,
        ) -> None:
        """
        Does a sliding median of data with NaN values and a kernel with weights.
        NaN values are ignored.
        If no valid data in the given window, the output is set to NaN.
        Use the 'median' property to access the sliding median result.

        Args:
            data (npt.NDArray[np.float64]): the data to compute the sliding median on.
            kernel (KernelType): the kernel to use for the sliding median.
            borders (BorderType, optional): the border type to use for padding. If None, does the
                same operation than setting the borders to 'constant' and pad_value to np.nan (i.e.
                adaptative kernel sizes). Defaults to 'reflect'.
            pad_value (float, optional): the value to use for padding when borders is 'constant'.
                Defaults to 0.
            force_contiguous (bool, optional): whether to force the data and kernel to be
                contiguous in memory. Defaults to True.
            threads (int | None, optional): the number of threads to use in the sliding operation.
                If set to None, uses all the available logical cores. Defaults to 1.
        """

        self._data = data
        self._kernel = self._check_kernel(kernel)
        self._borders = borders
        self._pad_value = pad_value
        self._threads = threads

        if force_contiguous: self._make_contiguous()

        # RUN
        self._median = self._sliding_median()

    @property
    def median(self) -> npt.NDArray[np.float64]:
        """
        The sliding median result.

        Returns:
            npt.NDArray[np.float64]: the sliding median result.
        """
        return self._median

    def _sliding_median(self) -> npt.NDArray[np.float64]:
        """
        Computes the sliding median using rust.
        Returns:
            npt.NDArray[np.float64]: the sliding median result.
        """

        pad_mode = self._borders if self._borders is not None else 'constant'
        pad_value = self._pad_value if self._borders is not None else np.nan
        median  = _rust.sliding_median(
            self._data,
            self._kernel,
            pad_mode,#type:ignore
            pad_value,
            self._threads,
        )
        return median
