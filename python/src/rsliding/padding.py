"""
Code to pad N-dimensional data given a kernel.
The padding naming follows the cv2.filter2D convention, except when set to None.
"""
from __future__ import annotations

# IMPORTs
import numpy as np

# IMPORTs local
from . import _bindings as _rust
from .utils import KernelType, BorderType, BaseCheck

# TYPE ANNOTATIONs
import numpy.typing as npt

# API public
__all__ = ["Padding"]



class Padding(BaseCheck):
    """
    To add padding to data according to the border type.
    The border type follows the cv2.filter2D border naming convention.
    """

    def __init__(
            self,
            data: npt.NDArray[np.float64],
            kernel: KernelType,
            borders: BorderType = 'reflect',
            pad_value: float = 0.,
            force_contiguous: bool = True,
        ) -> None:
        """
        Adds padding to the given data according to the border type.
        The border type follows the cv2.filter2D border naming convention. The padding is added
        using np.pad.
        To get the padded data, use the 'padded' property.

        Args:
            data (npt.NDArray[np.float64]): the data to pad.
            kernel (KernelType): the kernel to use for the convolution.
            borders (BorderType, optional): the border type to use for padding. If None, does the
                same operation than setting the borders to 'constant' and pad_value to np.nan (i.e.
                adaptative kernel sizes). Defaults to 'reflect'.
            pad_value (float, optional): the value to use for padding when borders is 'constant'.
                Defaults to 0.
            force_contiguous (bool, optional): whether to force the data and kernel to be
                contiguous in memory. Defaults to True.
        """

        self._data = data
        self._kernel = self._check_kernel(kernel)
        self._borders = borders
        self._pad_value = pad_value

        if force_contiguous: self._make_contiguous()

        # RUN
        self._padded_data = self._padding()

    @property
    def padded(self) -> npt.NDArray[np.float64]:
        """
        Gives the padded data.

        Returns:
            npt.NDArray[np.float64]: the padded data.
        """
        return self._padded_data

    def _padding(self) -> npt.NDArray[np.float64]:
        """
        N-dimensional padding of a numpy float64 array given a kernel.

        Returns:
            npt.NDArray[np.float64]: the padded array.
        """

        pad_mode = self._borders if self._borders is not None else 'constant'
        pad_value = self._pad_value if self._borders is not None else np.nan
        return _rust.padding(self._data, self._kernel, pad_mode, pad_value)#type:ignore
