//! Python bindings for the sliding median operation.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;

// local
use crate::bindings::utils::{array_d_to_py_array, py_array_to_array_d};
use crate::core::padding::{PaddingMode, SlidingWorkspace};
use crate::core::sliding_median::sliding_median;

/// N-dimensional sliding median of an input array with a kernel.
/// NaN values in the input are ignored in the median calculation.
/// If no valid values in the kernel window, the output is set to NaN.
/// Kernel can contain weights (will be normalized).
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
///   Input N-dimensional array.
/// kernel : numpy.ndarray[float64]
///  Kernel (weights) array with the same number of dimensions as ``data``.
/// pad_value : float64
///  Constant value used to pad the borders of ``data``.
///
/// Returns
/// -------
/// numpy.ndarray[float64]
///  Array with the same shape as ``data`` containing the sliding median result.
#[pyfunction(name = "sliding_median")]
pub fn py_sliding_median<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    pad_mode: &str,
    pad_value: f64,
    num_threads: Option<usize>,
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

    // threads
    match num_threads {
        Some(n) => {
            let pool = ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            py.allow_threads(|| {
                pool.install(|| {
                    // padding
                    let mut padded =
                        SlidingWorkspace::new(data_arr.shape(), kernel_arr, padding_mode).unwrap();
                    padded.pad_input(data_arr.view());

                    // sliding median
                    sliding_median(&padded, data_arr.view_mut());
                })
            });
        }
        None => {
            py.allow_threads(|| {
                // padding
                let mut padded =
                    SlidingWorkspace::new(data_arr.shape(), kernel_arr, padding_mode).unwrap();
                padded.pad_input(data_arr.view());

                // sliding median
                sliding_median(&padded, data_arr.view_mut());
            });
        }
    }

    array_d_to_py_array(py, data_arr)
}
