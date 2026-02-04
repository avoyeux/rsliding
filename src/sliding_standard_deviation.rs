//! N-dimensional sliding standard deviation operation.

use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};

// local
use crate::padding::PaddingWorkspace;

/// N-dimensional sliding standard deviation
/// also keeps the sliding mean as it might be chosen for the sigma clipping
pub fn sliding_standard_deviation<'a>(
    mut padded: PaddingWorkspace,
    mut mean_buffer: ArrayViewMutD<'a, f64>,
    kernel: ArrayViewD<'a, f64>,
) -> PaddingWorkspace {
    let mut padded_idx = vec![0usize; padded.ndim];
    let kernel_raw_dim = kernel.raw_dim();

    // iterate over input indices
    for input_idx in ndarray::indices(padded.valid_shape.clone()) {
        // count
        let mut n = 0_usize;
        let mut mean = 0.;
        let mut m2 = 0.;

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

                if kernel_val == 0. || input_val.is_nan() {
                    continue;
                }

                n += 1;
                let delta = input_val - mean;
                mean += delta * kernel_val / (n as f64); // ? not sure about this
                let delta2 = input_val - mean;
                m2 += delta * delta2 * kernel_val;
            }
        }
        // no bounds check
        unsafe {
            // mean
            *mean_buffer.uget_mut(&input_idx) = if n == 0 { f64::NAN } else { mean };

            // std
            *padded.output_buffer.uget_mut(input_idx) = if n == 0 {
                f64::NAN
            } else {
                (m2 / (n as f64)).sqrt()
            };
        }
    }
    padded
}
