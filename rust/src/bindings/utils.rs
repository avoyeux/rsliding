// ! Utilities to be used across the bindings.

use ndarray::{ArrayD, IxDyn};
use numpy::{PyArray1, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Read-only NumPy array → ndarray 0.17 ArrayD
pub fn py_array_to_array_d(arr: &PyReadonlyArrayDyn<'_, f64>) -> PyResult<ArrayD<f64>> {
    let shape = arr.shape().to_vec();
    let data = arr.as_slice().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Input array is not contiguous")
    })?;
    ArrayD::from_shape_vec(IxDyn(&shape), data.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

/// ndarray 0.17 ArrayD → NumPy array (via flat Vec + Python-side reshape)
pub fn array_d_to_py_array<'py>(
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
