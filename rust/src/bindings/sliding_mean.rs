//! Python bindings for the sliding mean operation.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// local
use crate::bindings::utils::{array_d_to_py_array, py_array_to_array_d};
use crate::core::padding::{PaddingMode, PaddingWorkspace};
use crate::core::sliding_mean::sliding_mean;

/// N-dimensional sliding mean of an input array with a kernel.
/// NaN values in the input are ignored in the mean calculation.
/// If no valid values in the kernel window, the output is set to NaN.
/// Kernel can contain weights (will be normalized).
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
///    Input N-dimensional array.
/// kernel : numpy.ndarray[float64]
///    Kernel (weights) array with the same number of dimensions as ``data``.
/// pad_value : float64
///    Constant value used to pad the borders of ``data``.
#[pyfunction(name = "sliding_mean")]
pub fn py_sliding_mean<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    pad_value: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let mut data_arr = py_array_to_array_d(&data)?;
    let kernel_arr = py_array_to_array_d(&kernel)?;

    // pad
    let pad_mode = PaddingMode::Constant(pad_value);
    let mut padded = PaddingWorkspace::new(data_arr.shape(), kernel_arr.shape(), pad_mode).unwrap();
    padded.pad_input(data_arr.view());

    // sliding mean
    sliding_mean(&padded, data_arr.view_mut(), kernel_arr.view());
    array_d_to_py_array(py, data_arr)
}
