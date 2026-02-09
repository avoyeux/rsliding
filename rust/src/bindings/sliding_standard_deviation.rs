//! Python bindings for the sliding standard deviation operation.

use ndarray::ArrayD;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// local
use crate::bindings::utils::{array_d_to_py_array, py_array_to_array_d};
use crate::core::padding::{PaddingMode, PaddingWorkspace};
use crate::core::sliding_standard_deviation::sliding_standard_deviation;

/// N-dimensional sliding standard deviation of an input array with a kernel.
/// NaN values in the input are ignored in the standard deviation calculation.
/// If no valid values in the kernel window, the output is set to NaN.
/// Kernel can contain weights.
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
///  Input N-dimensional array.
/// kernel : numpy.ndarray[float64]
/// Kernel (weights) array with the same number of dimensions as ``data``.
/// pad_value : float64
/// Constant value used to pad the borders of ``data``.
///
/// Returns
/// -------
/// Tuple of two numpy.ndarray[float64]:
/// 1. Array with the same shape as ``data`` containing the sliding standard deviation result
/// 2. Array with the same shape as ``data`` containing the sliding mean result (used in standard deviation calculation)
#[pyfunction(name = "sliding_standard_deviation")]
pub fn py_sliding_standard_deviation<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    pad_value: f64,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    let mut data_arr = py_array_to_array_d(&data)?;
    let kernel_arr = py_array_to_array_d(&kernel)?;

    // pad
    let pad_mode = PaddingMode::Constant(pad_value);
    let mut padded = PaddingWorkspace::new(data_arr.shape(), kernel_arr.shape(), pad_mode).unwrap();
    padded.pad_input(data_arr.view());
    let mut mean_buffer = ArrayD::zeros(padded.valid_shape.clone());

    // sliding standard deviation
    sliding_standard_deviation(
        &padded,
        data_arr.view_mut(),
        mean_buffer.view_mut(),
        kernel_arr.view(),
    );
    let standard_deviation = array_d_to_py_array(py, data_arr)?;
    let mean = array_d_to_py_array(py, mean_buffer)?;
    Ok((standard_deviation, mean))
}
