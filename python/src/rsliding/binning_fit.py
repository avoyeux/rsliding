"""
Code to fit a given ndarray while binning the data at the same time.
Hence the resulting array is of smaller shape than the input array (shape decrease depends on the
bins).
"""
from __future__ import annotations

# IMPORTs third-party
import numpy as np

# IMPORTs local
from . import _bindings as _rust

# TYPE ANNOTATIONs
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import numpy.typing as npt


# todo add proper checks for each parameter

class BinningFit:
    # todo add docstring

    def __init__(
            self,
            data: npt.NDArray[np.float64],
            error: npt.NDArray[np.float64],
            waves: npt.NDArray[np.float64],
            fit_axis: int,
            bins: tuple[int, ...],
            init_guess: npt.NDArray[np.float64],
            lower_bounds: npt.NDArray[np.float64],
            upper_bounds: npt.NDArray[np.float64],
            x_scales: npt.NDArray[np.float64],
            force_contiguous: bool = True,
            threads: int | None = 1,
        ) -> None:
        # todo add docstring

        if force_contiguous:
            data = np.ascontiguousarray(data, dtype=np.float64)
            error = np.ascontiguousarray(error, dtype=np.float64)
            waves = np.ascontiguousarray(waves, dtype=np.float64)
            init_guess = np.ascontiguousarray(init_guess, dtype=np.float64)
            lower_bounds = np.ascontiguousarray(lower_bounds, dtype=np.float64)
            upper_bounds = np.ascontiguousarray(upper_bounds, dtype=np.float64)
            x_scales = np.ascontiguousarray(x_scales, dtype=np.float64)

        self._data = data
        self._error = error
        self._waves = waves
        self._fit_axis = fit_axis
        self._bins = bins
        self._init_guess = init_guess
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        self._x_scales = x_scales
        self._threads = threads

        # RUN
        self._params, self._errors, self._cost = self._run_binning_fit()

    @property
    def params(self) -> npt.NDArray[np.float64]:
        """The binned fit parameters."""
        return self._params

    @property
    def errors(self) -> npt.NDArray[np.float64]:
        """The binned fit parameter uncertainties."""
        return self._errors

    @property
    def cost(self) -> npt.NDArray[np.float64]:
        """The binned fit cost (chi-squared)."""
        return self._cost

    def _run_binning_fit(
            self,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:

        params, errors, cost = _rust.binning_fit(
            self._data,
            self._error,
            self._waves,
            self._fit_axis,
            self._bins,
            self._init_guess,
            self._lower_bounds,
            self._upper_bounds,
            self._x_scales,
            self._threads,
        )
        return params, errors, cost
