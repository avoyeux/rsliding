"""
Code to compute the convolution of a given ndarray data with NaN values and a weighted kernel.
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
__all__ = ["Convolution"]



class Convolution(BaseCheck):
    """
    Does a convolution of data with NaN values and a kernel with weights.
    NaN values are ignored.
    If no valid data in the given window, the output is set to NaN.
    Use the 'convolution' property to access the convolution result.
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
        Does a convolution of data with NaN values and a kernel with weights.
        NaN values are ignored.
        If no valid data in the given window, the output is set to NaN.
        Use the 'convolution' property to access the convolution result.

        Args:
            data (npt.NDArray[np.float64]): the data to convolve.
            kernel (KernelType): the kernel to use for the convolution.
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
        self._convolution = self._convolve()

    @property
    def convolution(self) -> npt.NDArray[np.float64]:
        """
        The convolution result.

        Returns:
            npt.NDArray[np.float64]: the convolution result.
        """
        return self._convolution

    def _convolve(self) -> npt.NDArray[np.float64]:
        """
        Computes the convolution using rust.

        Returns:
            npt.NDArray[np.float64]: the convolution result.
        """

        pad_mode = self._borders if self._borders is not None else 'constant'
        pad_value = self._pad_value if self._borders is not None else np.nan
        convolution  = _rust.convolution(
            self._data,
            self._kernel,
            pad_mode,#type:ignore
            pad_value,
            self._threads,
        )
        return convolution
