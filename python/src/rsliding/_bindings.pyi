"""
Contains the function signatures of the rust functions that are exposed to python.
"""

# IMPORTs
import numpy as np
import numpy.typing as npt



def padding(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_value: float,
    ) -> npt.NDArray[np.float64]: ...

def convolution(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_value: float,
    ) -> npt.NDArray[np.float64]: ...

def sliding_mean(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_value: float,
    ) -> npt.NDArray[np.float64]: ...

def sliding_median(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_value: float,
    ) -> npt.NDArray[np.float64]: ...

def sliding_standard_deviation(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        pad_value: float,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

def sliding_sigma_clipping(
        data: npt.NDArray[np.float64],
        kernel: npt.NDArray[np.float64],
        sigma_upper: float,
        sigma_lower: float,
        center_mode: str,
        max_iterations: int,
        pad_value: float,
    ) -> npt.NDArray[np.float64]: ...
