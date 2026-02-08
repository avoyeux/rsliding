use ndarray::{ArrayD, IxDyn};
use numpy::{PyArray1, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

// local
use rsliding_core::{
    sliding_mean, sliding_median, sliding_sigma_clipping, sliding_standard_deviation, CenterMode,
    PaddingMode, PaddingWorkspace,
};

/// Read-only NumPy array → ndarray 0.17 ArrayD
fn py_array_to_array_d(arr: &PyReadonlyArrayDyn<'_, f64>) -> PyResult<ArrayD<f64>> {
    let shape = arr.shape().to_vec();
    let data = arr.as_slice().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Input array is not contiguous")
    })?;
    ArrayD::from_shape_vec(IxDyn(&shape), data.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

/// ndarray 0.17 ArrayD → NumPy array (via flat Vec + Python-side reshape)
fn array_d_to_py_array<'py>(
    py: Python<'py>,
    arr: ArrayD<f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let shape = arr.shape().to_vec();
    let (flat, _offset) = arr.into_raw_vec_and_offset();
    let arr1 = PyArray1::from_vec_bound(py, flat);
    let shape_tuple = PyTuple::new_bound(py, shape);
    arr1.call_method1("reshape", (shape_tuple,))?
        .downcast_into::<PyArrayDyn<f64>>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))
}

/// Compute the sliding (weighted) mean over an N-dimensional array.
///
/// NaN values in ``data`` are ignored in the mean calculation.
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
///     Input N-dimensional array.
/// kernel : numpy.ndarray[float64]
///     Kernel (weights) array with the same number of dimensions as ``data``.
/// pad_value : float
///     Constant value used to pad the borders of ``data``.
///
/// Returns
/// -------
/// numpy.ndarray[float64]
///     Array with the same shape as ``data`` containing the sliding mean.
#[pyfunction(name = "sliding_mean", text_signature = "(data, kernel, pad_value)")]
fn py_sliding_mean<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    pad_value: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let mut data_arr = py_array_to_array_d(&data)?;
    let kernel_arr = py_array_to_array_d(&kernel)?;

    let pad_mode = PaddingMode::Constant(pad_value);
    let mut padded = PaddingWorkspace::new(data_arr.shape(), kernel_arr.shape(), pad_mode)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    padded.pad_input(data_arr.view());

    sliding_mean(&padded, data_arr.view_mut(), kernel_arr.view());

    array_d_to_py_array(py, data_arr)
}

/// Compute the sliding median over an N-dimensional array.
///
/// NaN values in ``data`` are ignored. For an even number of valid
/// elements the average of the two middle values is returned.
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
///     Input N-dimensional array.
/// kernel : numpy.ndarray[float64]
///     Kernel (weights) array with the same number of dimensions as ``data``.
/// pad_value : float
///     Constant value used to pad the borders of ``data``.
///
/// Returns
/// -------
/// numpy.ndarray[float64]
///     Array with the same shape as ``data`` containing the sliding median.
#[pyfunction(name = "sliding_median", text_signature = "(data, kernel, pad_value)")]
fn py_sliding_median<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    pad_value: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let mut data_arr = py_array_to_array_d(&data)?;
    let kernel_arr = py_array_to_array_d(&kernel)?;

    let pad_mode = PaddingMode::Constant(pad_value);
    let mut padded = PaddingWorkspace::new(data_arr.shape(), kernel_arr.shape(), pad_mode)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    padded.pad_input(data_arr.view());

    sliding_median(&padded, data_arr.view_mut(), kernel_arr.view());
    array_d_to_py_array(py, data_arr)
}

/// Compute the sliding standard deviation (and mean) over an N-dimensional array.
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
///     Input N-dimensional array.
/// kernel : numpy.ndarray[float64]
///     Kernel (weights) array with the same number of dimensions as ``data``.
/// pad_value : float
///     Constant value used to pad the borders of ``data``.
///
/// Returns
/// -------
/// tuple[numpy.ndarray[float64], numpy.ndarray[float64]]
///     ``(standard_deviation, mean)`` arrays, both with the same shape as ``data``.
#[pyfunction(name = "sliding_standard_deviation", text_signature = "(data, kernel, pad_value)")]
fn py_sliding_standard_deviation<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    pad_value: f64,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    let mut data_arr = py_array_to_array_d(&data)?;
    let kernel_arr = py_array_to_array_d(&kernel)?;

    let pad_mode = PaddingMode::Constant(pad_value);
    let mut padded = PaddingWorkspace::new(data_arr.shape(), kernel_arr.shape(), pad_mode)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    padded.pad_input(data_arr.view());
    let mut mean_buffer = ArrayD::zeros(padded.valid_shape.clone());

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

/// Iterative sliding sigma-clipping over an N-dimensional array.
///
/// At each position the local centre and standard deviation are computed;
/// values further than ``sigma_upper`` / ``sigma_lower`` standard
/// deviations from the centre are replaced with NaN. The process repeats
/// until convergence or ``max_iterations`` is reached.
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64]
///     Input N-dimensional array.
/// kernel : numpy.ndarray[float64]
///     Kernel (weights) array with the same number of dimensions as ``data``.
/// sigma_upper : float
///     Number of standard deviations above the centre for clipping.
/// sigma_lower : float
///     Number of standard deviations below the centre for clipping.
/// center_mode : ``"mean"`` or ``"median"``
///     Statistic used as the centre of each sliding window.
/// max_iterations : int
///     Maximum number of clipping iterations.
/// pad_value : float
///     Constant value used to pad the borders of ``data``.
///
/// Returns
/// -------
/// numpy.ndarray[float64]
///     Sigma-clipped array (clipped values are set to NaN).
#[pyfunction(name = "sliding_sigma_clipping", text_signature = "(data, kernel, sigma_upper, sigma_lower, center_mode, max_iterations, pad_value)")]
fn py_sliding_sigma_clipping<'py>(
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

    let pad_mode = PaddingMode::Constant(pad_value);
    let mut padded = PaddingWorkspace::new(data_arr.shape(), kernel_arr.shape(), pad_mode)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    padded.pad_input(data_arr.view());

    let center_mode = match center_mode {
        "mean" => CenterMode::Mean,
        "median" => CenterMode::Median,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Invalid center_mode, expected 'mean' or 'median'",
            ))
        }
    };

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

#[pymodule]
fn _rsliding(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_sliding_mean, m)?)?;
    m.add_function(wrap_pyfunction!(py_sliding_median, m)?)?;
    m.add_function(wrap_pyfunction!(py_sliding_standard_deviation, m)?)?;
    m.add_function(wrap_pyfunction!(py_sliding_sigma_clipping, m)?)?;
    Ok(())
}
