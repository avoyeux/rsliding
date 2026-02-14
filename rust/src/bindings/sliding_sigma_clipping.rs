//! Python bindings for the sliding sigma clipping operation.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;

// local
use crate::bindings::utils::{array_d_to_py_array, py_array_to_array_d};
use crate::core::padding::{PaddingMode, SlidingWorkspace};
use crate::core::sliding_sigma_clipping::{CenterMode, sliding_sigma_clipping};

/// N-dimensional sliding sigma clipping of an input array with a kernel.
/// The clipped values are set to the corresponding final sliding mode value.
/// The sigma clipping stops after 'max_iterations' number of times or when no more clipping was
/// done.
/// NaN values in the input are ignored in the calculation.
/// If no valid values in the kernel window, the output is set to NaN.
/// Kernel can contain weights.
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
///   Input N-dimensional array.
/// kernel : numpy.ndarray[float64]
///    Kernel (weights) array with the same number of dimensions as ``data``.
/// center_mode: str
///   the sliding mode to use for the clipping. Can be 'mean' or 'median'.
/// pad_mode: str
///    the padding mode to use. Can be 'constant', 'reflect' or 'replicate'.
/// pad_value : float64
///    Constant value used to pad the borders of ``data``. Used when pad_mode is set to 'constant'.
/// neumaier: bool
///    Whether to use Neumaier summation for the sliding mean and standard deviation calculations.
///    This can improve the numerical stability of the calculations, especially for large kernels or
///    data with large values. However, it it will be slightly slower than the standard summation.
/// sigma_upper: float | None
///    the upper sigma coefficient to use for the clipping. If None, no upper clipping is done.
/// sigma_lower: float | None
///    the lower sigma coefficient to use for the clipping. If None, no lower clipping is done.
/// max_iterations: int | None
///    the maximum number of iterations to perform the sigma clipping. If None, iterations continue until convergence.
/// num_threads: int | None
///    The number of threads to used for the sliding operation. If set to None, all available
///    logical units are used.
///
/// Returns
/// -------
/// tuple[numpy.ndarray[float64], numpy.ndarray[bool]]
///     Array with the same shape as ``data`` containing the sigma clipped result.
///     Bool array with the clipped positions.
#[pyfunction(name = "sliding_sigma_clipping")]
pub fn py_sliding_sigma_clipping<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    center_mode: &str,
    pad_mode: &str,
    pad_value: f64,
    neumaier: bool,
    sigma_upper: Option<f64>,
    sigma_lower: Option<f64>,
    max_iterations: Option<usize>,
    num_threads: Option<usize>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<bool>>)> {
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

    // center mode
    let center_mode = match center_mode {
        "mean" => CenterMode::Mean,
        "median" => CenterMode::Median,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid center_mode, expected 'mean' or 'median'",
            ));
        }
    };

    // threads
    let changed_mask = match num_threads {
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

                    // sliding sigma clipping
                    sliding_sigma_clipping(
                        &mut padded,
                        data_arr.view_mut(),
                        &sigma_upper,
                        &sigma_lower,
                        &center_mode,
                        &max_iterations,
                        neumaier,
                    )
                })
            })
        }
        None => {
            py.allow_threads(|| {
                // padding
                let mut padded =
                    SlidingWorkspace::new(data_arr.shape(), kernel_arr, padding_mode).unwrap();
                padded.pad_input(data_arr.view());

                // sliding sigma clipping
                sliding_sigma_clipping(
                    &mut padded,
                    data_arr.view_mut(),
                    &sigma_upper,
                    &sigma_lower,
                    &center_mode,
                    &max_iterations,
                    neumaier,
                )
            })
        }
    };

    let sigma_clipped = array_d_to_py_array(py, data_arr)?;
    let changed_mask = array_d_to_py_array(py, changed_mask)?;
    Ok((sigma_clipped, changed_mask))
}
