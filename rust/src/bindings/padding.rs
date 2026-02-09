//! Python bindings for the padding operation.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// local
use crate::bindings::utils::{array_d_to_py_array, py_array_to_array_d};
use crate::core::padding::{PaddingMode, PaddingWorkspace};

/// Pad an N-dimensional array according to the specified kernel shape and padding value.
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
///    Input N-dimensional array.
/// kernel_shape : numpy.ndarray[uint64]
///   Shape of the kernel (number of dimensions must match ``data``).
/// pad_value : float64
///   Constant value used to pad the borders of ``data``.
#[pyfunction(name = "padding")]
pub fn py_padding<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    pad_value: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let data_arr = py_array_to_array_d(&data)?;
    let kernel_arr = py_array_to_array_d(&kernel)?;

    // pad
    let pad_mode = PaddingMode::Constant(pad_value);
    let mut padded = PaddingWorkspace::new(data_arr.shape(), kernel_arr.shape(), pad_mode).unwrap();
    padded.pad_input(data_arr.view());

    // return the padded buffer as a new array
    array_d_to_py_array(py, padded.padded_buffer)
}
