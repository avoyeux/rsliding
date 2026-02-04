//! To compute the sliding mean of an n-dimensional array with possible NaN values.

use ndarray::{ArrayViewD, IxDyn};

// local
use crate::padding::PaddingWorkspace;


/// N-dimensional sliding mean operation.
/// Handles NaN values by ignoring them in the mean calculation.
pub fn sliding_mean<'a>(mut padded: PaddingWorkspace, kernel: ArrayViewD<'a, f64>) -> PaddingWorkspace {
    let mut padded_idx = vec![0usize; padded.ndim];
    let kernel_raw_dim = kernel.raw_dim();

    // iterate over input indices
    for input_idx in ndarray::indices(padded.valid_shape.clone()) {
        // count
        let mut acc = 0.;
        let mut weighted_sum = 0.;

        // iterate over kernel
        for k_idx in ndarray::indices(kernel_raw_dim.clone()) {
            // compute padded index
            for d in 0..padded.ndim {
                padded_idx[d] = input_idx[d] + k_idx[d];
            }

            // no bounds check
            unsafe {
                let input_val = *padded.padded_buffer.uget(IxDyn(&padded_idx));
                let kernel_val = *kernel.uget(k_idx);

                if !input_val.is_nan() {
                    acc += input_val * kernel_val;
                    weighted_sum += kernel_val;
                }
            }
        }
        // no bounds check
        unsafe {
            *padded.output_buffer.uget_mut(input_idx) = 
                if weighted_sum == 0. { f64::NAN } else { acc / weighted_sum };
        }
    }
    padded
}