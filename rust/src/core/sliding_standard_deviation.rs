//! N-dimensional sliding standard deviation operation.

use ndarray::ArrayViewMutD;
use rayon::prelude::*;

// local
use crate::core::padding::SlidingWorkspace;

/// N-dimensional sliding standard deviation
/// also keeps the sliding mean as it might be chosen for the sigma clipping
pub fn sliding_standard_deviation<'a>(
    padded: &SlidingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    mut mean_buffer: ArrayViewMutD<'a, f64>,
) {
    // reset kernel index buffer
    let has_nan = data.iter().any(|v| v.is_nan());
    let padded_strides = padded.padded_buffer.strides();
    // Assume everything is contiguous and abort early if it is not.
    let padded_slice = padded
        .padded_buffer
        .as_slice_memory_order()
        .expect("Padding buffer must be contiguous");
    let out_slice = data
        .as_slice_memory_order_mut()
        .expect("Output view must be contiguous");
    let mean_slice = mean_buffer
        .as_slice_memory_order_mut()
        .expect("Mean buffer must be contiguous");

    let k_offsets = &padded.kernel_offsets;
    let k_weights = &padded.kernel_weights;

    out_slice
        .par_iter_mut()
        .zip(mean_slice.par_iter_mut())
        .enumerate()
        .for_each(|(out_linear, (out, mean_out))| {
            let base = padded.base_offset_from_linear(out_linear, padded_strides);

            let mut n = 0usize;
            let mut mean = 0.0;
            let mut m2 = 0.0;

            if has_nan {
                for i in 0..k_offsets.len() {
                    let value = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                    let kernel_value = k_weights[i];
                    if !value.is_nan() {
                        n += 1;
                        let delta = value - mean;
                        mean += delta * kernel_value / (n as f64);
                        let delta2 = value - mean;
                        m2 += delta * delta2 * kernel_value;
                    }
                }
            } else {
                for i in 0..k_offsets.len() {
                    let value = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                    let kernel_value = k_weights[i];
                    n += 1;
                    let delta = value - mean;
                    mean += delta * kernel_value / (n as f64);
                    let delta2 = value - mean;
                    m2 += delta * delta2 * kernel_value;
                }
            }

            *out = if n == 0 {
                f64::NAN
            } else {
                (m2 / (n as f64)).sqrt()
            };
            *mean_out = if n == 0 { f64::NAN } else { mean };
        });
}
