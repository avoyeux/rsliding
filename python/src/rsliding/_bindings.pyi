"""
Contains the function signatures of the rust functions that are exposed to python.
"""

# IMPORTs
import numpy as np
import numpy.typing as npt
from typing import Literal



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
        num_threads: int | None,
    ) -> npt.NDArray[np.float64]: ...

def sliding_mean(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
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
        num_threads: int | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

def sliding_sigma_clipping(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        center_mode: str,
        pad_mode: Literal["constant", "reflect", "replicate"],
        pad_value: float,
        sigma_upper: float | None,
        sigma_lower: float | None,
        max_iterations: int | None,
        num_threads: int | None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]: ...
