"""
Base class (mixin) used to defined the same shared kernel and contiguous checks.
"""
from __future__ import annotations

# IMPORTs
import numpy as np

# TYPE ANNOTATIONs
from typing import Protocol, TypeAlias, Literal
import numpy.typing as npt
KernelType: TypeAlias = int | tuple[int, ...] | npt.NDArray[np.float64]
BorderType: TypeAlias = Literal['reflect', 'constant', 'replicate'] | None

# API public
__all__ = ["KernelType", "BorderType", "BaseCheck"]



class _HasData(Protocol):
    _data: npt.NDArray[np.float64]
class _HasDataAndKernel(Protocol):
    _data: npt.NDArray[np.float64]
    _kernel: npt.NDArray[np.float64]

class BaseCheck:
    """
    Base class that contains a method used to check and convert the kernel used in the sliding
    operations. Also contains a method to convert the data and kernel to C contiguous arrays.
    """

    def _check_kernel(self: _HasData,kernel: KernelType) -> npt.NDArray[np.float64]:
        """
        To check the input kernel shape, type and convert it to an ndarray if needed.

        Args:
            kernel (int | tuple[int, ...] | npt.NDArray[np.float64]): the kernel to check.

        Raises:
            TypeError: if the kernel is not an int, a tuple of ints or an ndarray.
            ValueError: if the kernel shape is not composed of positive odd integers or if the
                kernel dimensions do not match the data dimensions.

        Returns:
            npt.NDArray[np.float64]: the kernel as an ndarray.
        """

        if isinstance(kernel, int):
            if kernel <=0 or kernel % 2 == 0:
                raise ValueError("The kernel size must be a positive odd integer.")
            return np.ones((kernel,) * self._data.ndim, dtype=self._data.dtype)#type:ignore
        elif isinstance(kernel, tuple):
            if any(k <=0 or k % 2 == 0 for k in kernel):
                raise ValueError("All kernel dimensions must be positive odd integers.")
            elif len(kernel) != self._data.ndim:
                raise ValueError(
                    "If 'kernel' is given as a tuple, it must have the same number of "
                    "dimensions as 'data'."
                )
            return np.ones(kernel, dtype=self._data.dtype)#type:ignore
        elif isinstance(kernel, np.ndarray):
            if any(s <=0 or s % 2 == 0 for s in kernel.shape):
                raise ValueError("All kernel dimensions must be positive odd integers.")
            elif kernel.ndim != self._data.ndim:
                raise ValueError(
                    "If 'kernel' is given as a numpy ndarray, it must have the same number of "
                    "dimensions as 'data'."
                )
            return kernel
        else:
            raise TypeError("The kernel must be an integer, a tuple of integers or an ndarray.")

    def _make_contiguous(self: _HasDataAndKernel) -> None:
        """
        To convert the data and kernel to C contiguous arrays (needed for the rust code to run as
        intended).
        """

        self._data = np.ascontiguousarray(self._data)
        self._kernel = np.ascontiguousarray(self._kernel)
