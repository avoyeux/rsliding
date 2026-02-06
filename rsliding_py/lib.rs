use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use ndarray::{ArrayD, IxDyn};

// local
use rsliding_core::{PaddingMode, PaddingWorkspace, sliding_mean};

// ---------------------------------------------------------------------------
// Helpers to bridge numpy (ndarray 0.15) <-> rsliding_core (ndarray 0.17)
// by going through raw &[f64] / Vec<f64>, avoiding cross-version types.
// ---------------------------------------------------------------------------

/// Read-only NumPy array → ndarray 0.17 ArrayD
fn pyarray_to_arrayd(arr: &PyReadonlyArrayDyn<'_, f64>) -> PyResult<ArrayD<f64>> {
    let shape = arr.shape().to_vec();
    let data = arr
        .as_slice()
        .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Input array is not contiguous"))?;
    ArrayD::from_shape_vec(IxDyn(&shape), data.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

/// ndarray 0.17 ArrayD → NumPy array (via flat Vec + Python-side reshape)
fn arrayd_to_pyarray<'py>(py: Python<'py>, arr: ArrayD<f64>) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let shape = arr.shape().to_vec();
    let (flat, _offset) = arr.into_raw_vec_and_offset();
    let arr1 = PyArray1::from_vec_bound(py, flat);
    let shape_tuple = PyTuple::new_bound(py, shape);
    arr1.call_method1("reshape", (shape_tuple,))?
        .downcast_into::<PyArrayDyn<f64>>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))
}

// ---------------------------------------------------------------------------
// Python-facing functions
// ---------------------------------------------------------------------------

#[pyfunction(name = "sliding_mean")]
fn py_sliding_mean<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    kernel: PyReadonlyArrayDyn<'py, f64>,
    pad_value: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let mut data_arr = pyarray_to_arrayd(&data)?;
    let kernel_arr = pyarray_to_arrayd(&kernel)?;

    let pad_mode = PaddingMode::Constant(pad_value);
    let mut padded = PaddingWorkspace::new(data_arr.shape(), kernel_arr.shape(), pad_mode)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    padded.pad_input(data_arr.view());

    sliding_mean(&padded, data_arr.view_mut(), kernel_arr.view());

    arrayd_to_pyarray(py, data_arr)
}

#[pymodule]
fn sliding_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_sliding_mean, m)?)?;
    Ok(())
}