//! Python bindings for the sliding mean operation.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;

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
/// pad_mode: str
///    the padding mode to use. Can be 'constant', 'reflect' or 'replicate'.
/// pad_value : float64
///    Constant value used to pad the borders of ``data``. Used when pad_mode is set to 'constant'.
/// neumaier: bool
///   Whether to use Neumaier summation for the sliding mean and standard deviation calculations.
///    This can improve the numerical stability of the calculations, especially for large kernels or
///   data with large values. However, it it will be slightly slower than the standard summation.
/// num_threads: int | None
///     the number of threads to use in the sliding operation. If set to None, all available logical
///     units are used.
///
/// Returns
/// ----------
/// numpy.ndarray[float64]
///    Array with the same shape as ``data`` containing the sliding mean result.
#[pyfunction(name = "sliding_mean")]
pub fn py_sliding_mean<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    pad_mode: &str,
    pad_value: f64,
    neumaier: bool,
    num_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let mut data_arr = py_array_to_array_d(&data)?;
    let kernel_arr = py_array_to_array_d(&kernel)?;

    // pad mode
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

                    // sliding mean
                    sliding_mean(&padded, data_arr.view_mut(), neumaier);
                })
            });
        }
        None => {
            py.allow_threads(|| {
                // padding
                let mut padded =
                    SlidingWorkspace::new(data_arr.shape(), kernel_arr, padding_mode).unwrap();
                padded.pad_input(data_arr.view());

                // sliding mean
                sliding_mean(&padded, data_arr.view_mut(), neumaier);
            });
        }
    }

    array_d_to_py_array(py, data_arr)
}
