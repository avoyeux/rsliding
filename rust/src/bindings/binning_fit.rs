// ! Python bindings for the binning fit operation
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::bindings::utils::{array_d_to_py_array, py_array_to_array_d};
use crate::core::binning_fit::binning_fit;

#[pyfunction(name = "binning_fit")]
pub fn py_binning_fit<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    error: PyReadonlyArrayDyn<'py, f64>,
    waves: PyReadonlyArrayDyn<'py, f64>,
    fit_axis: usize,
    bins: Vec<usize>,
    init_guess: PyReadonlyArrayDyn<'py, f64>,
    lower_bounds: PyReadonlyArrayDyn<'py, f64>,
    upper_bounds: PyReadonlyArrayDyn<'py, f64>,
    x_scales: PyReadonlyArrayDyn<'py, f64>,
    num_threads: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
)> {
    if let Some(n) = num_threads {
        ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    }

    let data_arr = py_array_to_array_d(&data)?;
    let error_arr = py_array_to_array_d(&error)?;
    let waves_arr = py_array_to_array_d(&waves)?;
    let init_guess_arr = py_array_to_array_d(&init_guess)?;
    let lower_bounds_arr = py_array_to_array_d(&lower_bounds)?;
    let upper_bounds_arr = py_array_to_array_d(&upper_bounds)?;
    let x_scales_arr = py_array_to_array_d(&x_scales)?;

    let waves_slice = waves_arr.as_slice().unwrap();
    let init_slice = init_guess_arr.as_slice().unwrap();
    let lower_slice = lower_bounds_arr.as_slice().unwrap();
    let upper_slice = upper_bounds_arr.as_slice().unwrap();
    let x_scales_slice = x_scales_arr.as_slice().unwrap();

    let (params, errors, cost) = binning_fit(
        data_arr.view(),
        error_arr.view(),
        waves_slice,
        fit_axis,
        bins,
        init_slice,
        lower_slice,
        upper_slice,
        x_scales_slice,
    );

    let py_params = array_d_to_py_array(py, params)?;
    let py_errors = array_d_to_py_array(py, errors)?;
    let py_cost = array_d_to_py_array(py, cost)?;

    Ok((py_params, py_errors, py_cost))
}
