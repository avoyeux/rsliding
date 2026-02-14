//! Python bindings for the padding operation.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// local
use crate::bindings::utils::{array_d_to_py_array, py_array_to_array_d};
use crate::core::padding::{PaddingMode, SlidingWorkspace};

/// Adds padding to an N-dimensional array according to the specified kernel shape and padding
/// option.
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
///    Input N-dimensional array.
/// kernel : numpy.ndarray[float64]
///    the kernel to use when doing the sliding operations (needs to have the same dimensionality
///    as ``data``).
/// pad_mode: str
///    the padding mode to use. Can be 'constant', 'reflect' or 'replicate'.
/// pad_value : float64
///    Constant value used to pad the borders of ``data`` (only used when pad_mode is 'constant').
///
/// Returns
/// ----------
/// numpy.ndarray[float64]
///    Padded N-dimensional array.
#[pyfunction(name = "padding")]
pub fn py_padding<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    pad_mode: &str,
    pad_value: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let data_arr = py_array_to_array_d(&data)?;
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

    // return the padded buffer as a new array
    array_d_to_py_array(py, padded.padded)
}
