//! To compute the sliding mean of an n-dimensional array with possible NaN values.

use ndarray::{ArrayD, ArrayViewD};
use num_traits::{Float, Zero};
use std::ops::AddAssign;

// local
use crate::convolution::convolution;


pub fn sliding_mean<T>(data: ArrayViewD<'_, T>, kernel: ArrayViewD<'_, T>) -> ArrayD<T>
where
    T: Float + Zero + AddAssign,
{
    // normalize kernel
    let kernel_sum = kernel.sum();
    let normalised_kernel = kernel.mapv(|x| x / kernel_sum);
    
    // sliding mean
    convolution(data, normalised_kernel.view())
}
