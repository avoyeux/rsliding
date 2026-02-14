//! Python bindings for the convolution operation.

use ndarray::Axis;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;

// local
use crate::bindings::utils::{array_d_to_py_array, py_array_to_array_d};
use crate::core::convolution::convolution;
use crate::core::padding::{PaddingMode, SlidingWorkspace};

/// Compute the N-dimensional convolution of an input array with a weighted kernel.
/// NaN values in the input are ignored in the convolution operation.
/// If no valid values in the kernel window, the output is set to NaN.
/// Kernel can contain weights.
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
///    Input N-dimensional array.
/// kernel : numpy.ndarray[float64]
///    Kernel (weights) array with the same number of dimensions as ``data``.
/// pad_mode: str
///    Padding mode to use. One of 'constant', 'reflect' or 'replicate'.
/// pad_value : float64
///    Constant value used to pad the borders of ``data``. Used when pad_mode is set to 'constant'.
/// neumaier: bool
///    Whether to use Neumaier summation for the convolution calculation. This can improve the
///    numerical stability of the calculations, especially for large kernels or data with large values.
///    However, it it will be slightly slower than the standard summation.
/// num_threads: int | None
///     the number of threads to use in the convolution. If None, uses the number of available
///     logical units.
///
/// Returns
/// ----------
/// numpy.ndarray[float64]
///    Array with the same shape as ``data`` containing the convolution result.
#[pyfunction(name = "convolution")]
pub fn py_convolution<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    pad_mode: &str,
    pad_value: f64,
    neumaier: bool,
    num_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let mut data_arr = py_array_to_array_d(&data)?;
    let mut kernel_arr = py_array_to_array_d(&kernel)?;

    // invert as the actual convolution function does a correlation operation.
    for axis in 0..kernel_arr.ndim() {
        kernel_arr.invert_axis(Axis(axis));
    }

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

                    // convolution
                    convolution(&padded, data_arr.view_mut(), neumaier);
                })
            });
        }
        None => {
            py.allow_threads(|| {
                // padding
                let mut padded =
                    SlidingWorkspace::new(data_arr.shape(), kernel_arr, padding_mode).unwrap();
                padded.pad_input(data_arr.view());

                // convolution
                convolution(&padded, data_arr.view_mut(), neumaier);
            });
        }
    }

    array_d_to_py_array(py, data_arr)
}
