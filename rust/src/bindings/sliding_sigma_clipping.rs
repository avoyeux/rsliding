//! Python bindings for the sliding sigma clipping operation.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// local
use crate::bindings::utils::{array_d_to_py_array, py_array_to_array_d};
use crate::core::padding::{PaddingMode, PaddingWorkspace};
use crate::core::sliding_sigma_clipping::{CenterMode, sliding_sigma_clipping};

/// N-dimensional sliding sigma clipping of an input array with a kernel.
/// NaN values in the input are ignored in the calculation.
/// If no valid values in the kernel window, the output is set to NaN.
/// Kernel can contain weights.
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
/// Input N-dimensional array.
/// kernel : numpy.ndarray[float64]
/// Kernel (weights) array with the same number of dimensions as ``data``.
/// sigma_upper : float64
/// Upper sigma threshold for clipping. If None, no upper clipping is applied.
/// sigma_lower : float64
/// Lower sigma threshold for clipping. If None, no lower clipping is applied.
/// center_mode : str
/// Method to compute the center value for sigma clipping, either 'mean' or 'median'.
/// max_iterations : int
/// Maximum number of iterations for sigma clipping. If 0, iterations continue until convergence.
/// pad_value : float64
/// Constant value used to pad the borders of ``data``.
///
/// Returns
/// -------
/// numpy.ndarray[float64]
/// Array with the same shape as ``data`` containing the sigma clipped result.
#[pyfunction(name = "sliding_sigma_clipping")]
pub fn py_sliding_sigma_clipping<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    sigma_upper: f64,
    sigma_lower: f64,
    center_mode: &str,
    max_iterations: usize,
    pad_value: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let mut data_arr = py_array_to_array_d(&data)?;
    let kernel_arr = py_array_to_array_d(&kernel)?;

    // pad
    let pad_mode = PaddingMode::Constant(pad_value);
    let mut padded = PaddingWorkspace::new(data_arr.shape(), kernel_arr.shape(), pad_mode).unwrap();
    padded.pad_input(data_arr.view());

    let center_mode = match center_mode {
        "mean" => CenterMode::Mean,
        "median" => CenterMode::Median,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid center_mode, expected 'mean' or 'median'",
            ));
        }
    };

    // sliding sigma clipping
    sliding_sigma_clipping(
        &mut padded,
        data_arr.view_mut(),
        kernel_arr.view(),
        &Some(sigma_upper),
        &Some(sigma_lower),
        &center_mode,
        &Some(max_iterations),
    );
    array_d_to_py_array(py, data_arr)
}
