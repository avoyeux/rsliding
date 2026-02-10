//! Python bindings for the sliding mean operation.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// local
use crate::bindings::utils::{array_d_to_py_array, py_array_to_array_d};
use crate::core::padding::{PaddingMode, SlidingWorkspace};
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
    pad_mode: &str,
    pad_value: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let mut data_arr = py_array_to_array_d(&data)?;
    let kernel_arr = py_array_to_array_d(&kernel)?;

    // pad
    let padding_mode = match pad_mode {
        "constant" => PaddingMode::Constant(pad_value),
        "reflect" => PaddingMode::Reflect,
        "replicate" => PaddingMode::Replicate,
        _ => {
            let args = format!(
                "Invalid padding mode: {}. Must be one of 'constant', 'reflect', 'replicate', or 'wrap'.",
                pad_mode,
            );
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(args));
        }
    };
    let mut padded = SlidingWorkspace::new(data_arr.shape(), kernel_arr, padding_mode).unwrap();
    padded.pad_input(data_arr.view());

    // sliding mean
    sliding_mean(&mut padded, data_arr.view_mut());
    array_d_to_py_array(py, data_arr)
}
