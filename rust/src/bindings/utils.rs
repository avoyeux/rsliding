// ! Utilities to be used across the bindings.

use ndarray::{ArrayD, IxDyn};
use numpy::{PyArray1, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Converts a read-only NumPy `ndarray` (`float64`, dynamic dimension) into an owned
/// `ndarray::ArrayD<f64>`.
///
/// The input must be C-contiguous in memory. The function copies the data into a new
/// Rust-owned buffer, preserving shape.
///
/// # Parameters
/// - `arr`: Read-only NumPy array view (`PyReadonlyArrayDyn<f64>`).
///
/// # Returns
/// - `Ok(ArrayD<f64>)`: Owned n-dimensional Rust array with the same shape and values.
/// - `Err(PyValueError)`: If the input NumPy array is not contiguous, or if shape/data
///   reconstruction fails.
///
/// # Notes
/// - This performs a data copy (`to_vec()`), so the returned array is independent from
///   the original Python object.
pub fn py_array_to_array_d(arr: &PyReadonlyArrayDyn<'_, f64>) -> PyResult<ArrayD<f64>> {
    let shape = arr.shape().to_vec();
    let data = arr.as_slice().map_err(|_| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>("Input array is not contiguous")
    })?;
    ArrayD::from_shape_vec(IxDyn(&shape), data.to_vec())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

/// Converts an owned `ndarray::ArrayD<T>` into a NumPy dynamic-dimensional array
/// (`numpy.ndarray`) and returns it as `PyArrayDyn<T>`.
///
/// The conversion flattens the Rust array into a contiguous `Vec<T>`, creates a 1D NumPy
/// array, then reshapes it back to the original n-dimensional shape.
///
/// # Type Parameters
/// - `T`: Element type compatible with NumPy (`numpy::Element`).
///
/// # Parameters
/// - `py`: Active Python GIL token.
/// - `arr`: Owned Rust n-dimensional array.
///
/// # Returns
/// - `Ok(Bound<PyArrayDyn<T>>)`: NumPy array with the same shape and values as `arr`.
/// - `Err(PyTypeError)`: If reshaping/downcasting to `PyArrayDyn<T>` fails.
///
/// # Notes
/// - Ownership of array data is transferred into Python.
/// - The resulting NumPy array is newly allocated and independent from the original Rust
///   `ArrayD`.
pub fn array_d_to_py_array<'py, T>(
    py: Python<'py>,
    arr: ArrayD<T>,
) -> PyResult<Bound<'py, PyArrayDyn<T>>>
where
    T: numpy::Element,
{
    let shape = arr.shape().to_vec();
    let (flat, _offset) = arr.into_raw_vec_and_offset();
    let arr1 = PyArray1::from_vec_bound(py, flat);
    let shape_tuple = PyTuple::new_bound(py, shape);
    arr1.call_method1("reshape", (shape_tuple,))?
        .downcast_into::<PyArrayDyn<T>>()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))
}
