"""
Contains the function signatures of the rust functions that are exposed to python.
"""

# IMPORTs third-party
import numpy as np

# TYPE ANNOTATIONs
from typing import Literal
import numpy.typing as npt



def padding(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
    ) -> npt.NDArray[np.float64]: ...

def convolution(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
        neumaier: bool,
        num_threads: int | None,
    ) -> npt.NDArray[np.float64]: ...

def sliding_mean(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
        neumaier: bool,
        num_threads: int | None,
    ) -> npt.NDArray[np.float64]: ...

def sliding_median(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
        num_threads: int | None,
    ) -> npt.NDArray[np.float64]: ...

def sliding_standard_deviation(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
        neumaier: bool,
        num_threads: int | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

def sliding_sigma_clipping(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        center_mode: str,
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
        neumaier: bool,
        sigma_upper: float | None,
        sigma_lower: float | None,
        max_iterations: int | None,
        num_threads: int | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]: ...

def binning_fit(
        data: npt.NDArray[np.float64],
        error: npt.NDArray[np.float64],
        waves: npt.NDArray[np.float64],
        fit_axis: int,
        bins: tuple[int, ...],
        init_guess: npt.NDArray[np.float64],
        lower_bounds: npt.NDArray[np.float64],
        upper_bounds: npt.NDArray[np.float64],
        x_scales: npt.NDArray[np.float64],
        num_threads: int | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...
