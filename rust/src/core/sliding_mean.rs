//! To compute the sliding mean of an n-dimensional array with possible NaN values.

use ndarray::ArrayViewMutD;
use rayon::prelude::*;

// local
use crate::core::padding::SlidingWorkspace;

// todo update all docstring

// / N-dimensional sliding mean operation.
// / NaN values are ignored.
// / Kernel can contain weights.
// / Data, kernel, and the padded buffer must be contiguous; this function will panic otherwise.
pub fn sliding_mean<'a>(padded: &SlidingWorkspace, mut data: ArrayViewMutD<'a, f64>) {
    let has_nan = data.iter().any(|v| v.is_nan());
    let padded_strides = padded.padded_buffer.strides();
    let padded_slice = padded
        .padded_buffer
        .as_slice_memory_order()
        .expect("Padding buffer must be contiguous");
    let out_slice = data
        .as_slice_memory_order_mut()
        .expect("Output view must be contiguous");

    let k_offsets = &padded.kernel_offsets;
    let k_weights = &padded.kernel_weights;

    out_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(out_linear, out)| {
            let base = padded.base_offset_from_linear(out_linear, padded_strides);

            let mut acc = 0.0;
            let mut weight_sum = 0.0;

            if has_nan {
                for i in 0..k_offsets.len() {
                    let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                    if !v.is_nan() {
                        acc += v * k_weights[i];
                        weight_sum += k_weights[i];
                    }
                }
            } else {
                for i in 0..k_offsets.len() {
                    let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                    acc += v * k_weights[i];
                    weight_sum += k_weights[i];
                }
            }

            *out = if weight_sum == 0.0 {
                f64::NAN
            } else {
                acc / weight_sum
            };
        });
}
