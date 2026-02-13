"""
Code to compute the sliding sigma clipping of a given ndarray data with NaN values and a weighted
kernel.
"""
from __future__ import annotations

# IMPORTs
import numpy as np

# IMPORTs local
from . import _bindings as _rust
from .utils import KernelType, BorderType, BaseCheck

# TYPE ANNOTATIONs
from typing import Literal
import numpy.typing as npt

# SENTINEL for default sigma lower/upper
class _UseSigma: pass
_USE_SIGMA = _UseSigma()

# API public
__all__ = ["SlidingSigmaClipping"]



class SlidingSigmaClipping(BaseCheck):
    """
    Does a sliding sigma clipping of data with NaN values and a kernel with weights.
    The clipped data is swapped with the final mode of the corresponding window.
    NaN values are ignored.
    If the final mode value is NaN, then the corresponding output is NaN.
    Use the 'clipped' property to access the sliding sigma clipping result.
    The result can be a masked array or the a numpy array depending on the 'masked_array' argument.
    """

    def __init__(
            self,
            data: npt.NDArray[np.float64],
            kernel: KernelType = 3,
            center_choice: Literal['median', 'mean'] = 'median',
            sigma: float = 3.,
            sigma_lower: float | None | _UseSigma = _USE_SIGMA,
            sigma_upper: float | None | _UseSigma = _USE_SIGMA,
            borders: BorderType = 'reflect',
            pad_value: float = 0.,
            max_iters: int | None = 5,
            masked_array: bool = False,
            force_contiguous: bool = True,
            threads: int | None = 1,
        ) -> None:
        """
        Does a sliding sigma clipping of data with NaN values and a kernel with weights.
        The clipped data is swapped with the final mode of the corresponding window.
        NaN values are ignored.
        If the final mode value is NaN, then the corresponding output is NaN.
        Use the 'clipped' property to access the sliding sigma clipping result.
        The result can be a masked array or the a numpy array depending on the 'masked_array'
        argument.

        Args:
            data (np.ndarray): the data to sigma clip.
            kernel (KernelType, optional): the kernel information used for computing the modes
                and standard deviations. Can be an int (square kernel), a tuple of ints (defining
                the shape of the kernel) or a numpy ndarray (defining the full kernel with
                weights). Defaults to 3.
            center_choice (Literal['median', 'mean'], optional): the function to use for computing
                the mode for each pixel. Defaults to 'median'.
            sigma (float): the number of standard deviations to use for both the lower and upper
                clipping limit. Overridden by 'sigma_lower' and/or 'sigma_upper'. 
            sigma_lower (float | None | _UseSigma, optional): the number of standard deviations to
                use for the lower clipping limit. It will be set to 'sigma' if _USE_SIGMA. When set
                to None, no lower clipping is done. Defaults to _USE_SIGMA.
            sigma_upper (float | None | _UseSigma, optional): the number of standard deviations to
                use for the upper clipping limit. It will be set to 'sigma' if _USE_SIGMA. When set
                to None, no upper clipping is done. Defaults to _USE_SIGMA.
            borders (BorderType, optional): the border type to use for padding. If None, does the
                same operation than setting the borders to 'constant' and pad_value to np.nan (i.e.
                adaptative kernel sizes). Defaults to 'reflect'.
            pad_value (float, optional): the value to use for padding when borders is 'constant'.
                Defaults to 0.
            max_iters (int | None, optional): the maximum number of iterations to perform.
                If None, iterate until convergence. Defaults to 5.
            masked_array (bool, optional): whether to return a MaskedArray (True) or a normal
                ndarray (False). Defaults to False.
            force_contiguous (bool, optional): whether to force the data and kernel to be
                contiguous in memory. Also casts the data to float64. Defaults to True.
            threads (int | None, optional): the number of threads to use in the sliding operation.
                If set to None, uses all the available logical cores. Defaults to 1.

        Raises:
            ValueError: if both sigma_upper and sigma_lower are set to None.
        """

        self._data = data
        self._kernel = self._check_kernel(kernel)
        self._borders = borders
        self._pad_value = pad_value
        self._masked_array = masked_array
        self._center_choice = center_choice
        self._sigma_lower = sigma if sigma_lower is _USE_SIGMA else sigma_lower
        self._sigma_upper = sigma if sigma_upper is _USE_SIGMA else sigma_upper
        self._threads = threads
        self._max_iters = max_iters

        # CHECK
        if (self._sigma_upper is None) and (self._sigma_lower is None):
            raise ValueError("At least one 'sigma_upper' or 'sigma_lower' must be not None.")

        if force_contiguous: self._make_contiguous()

        # RUN
        self._clipped = self._sigma_clipping()

    @property
    def clipped(self) -> npt.NDArray[np.float64] | np.ma.MaskedArray:
        """
        The sliding sigma clipping result.
        The clipped values are set to the corresponding final mode value.
        Can have NaN values if no valid data are left in a given window.
        Is a MaskedArray if 'masked_array' was set to True.
        The mask represents what values where swapped with the corresponding mode value.

        Returns:
            npt.NDArray[np.float64] | np.ma.MaskedArray: the sliding sigma clipping result.
        """
        return self._clipped

    def _sigma_clipping(self) -> npt.NDArray[np.float64] | np.ma.MaskedArray:
        """
        Computes the sliding sigma clipping using rust.

        Returns:
            npt.NDArray[np.float64] | np.ma.MaskedArray: the sliding sigma clipping result.
        """

        pad_mode = self._borders if self._borders is not None else 'constant'
        pad_value = self._pad_value if self._borders is not None else np.nan
        sigma_clipped, mask = _rust.sliding_sigma_clipping(
            self._data,
            self._kernel,
            self._center_choice,
            pad_mode,#type:ignore
            pad_value,
            self._sigma_upper,#type:ignore
            self._sigma_lower,#type:ignore
            self._max_iters,
            self._threads,
        )
        if self._masked_array: return np.ma.MaskedArray(sigma_clipped, mask=mask)
        return sigma_clipped
