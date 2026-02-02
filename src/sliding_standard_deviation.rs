//! N-dimensional sliding standard deviation operation.

use ndarray::{ArrayD, ArrayViewD};
use num_traits::{Float, Zero};
use std::ops::AddAssign;

/// N-dimensional sliding standard deviation
/// also keeps the sliding mean as it might be chosen for the sigma clipping
// todo add the sliding mean to the return signature.
pub fn sliding_standard_deviation<T>(data: ArrayViewD<'_, T>, kernel: ArrayViewD<'_, T>) -> ArrayD<T>
where
    T: Float + Zero + AddAssign,
{

}