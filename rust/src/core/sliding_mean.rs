//! To compute the sliding mean of an n-dimensional array with possible NaN values.

use ndarray::ArrayViewMutD;

// local
use crate::core::padding::SlidingWorkspace;

// todo update all docstring

/// N-dimensional sliding mean operation.
/// NaN values are ignored.
/// Kernel can contain weights.
/// Data, kernel, and the padded buffer must be contiguous; this function will panic otherwise.
pub fn sliding_mean<'a>(padded: &mut SlidingWorkspace, mut data: ArrayViewMutD<'a, f64>) {
    // reset kernel index buffer
    padded.idx.fill(0);

    let has_nan = data.iter().any(|v| v.is_nan());
    let mut base = 0isize;
    let mut out_linear = 0usize;
    let padded_strides = padded.padded_buffer.strides();
    let padded_slice = padded
        .padded_buffer
        .as_slice_memory_order()
        .expect("Padding buffer must be contiguous");
    let out_slice = data
        .as_slice_memory_order_mut()
        .expect("Output view must be contiguous");

    loop {
        let mut acc = 0.0;
        let mut weight_sum = 0.0;

        if has_nan {
            for i in 0..padded.kernel_offsets.len() {
                let v = unsafe {
                    *padded_slice
                        .as_ptr()
                        .offset(base + padded.kernel_offsets[i])
                };
                if !v.is_nan() {
                    acc += v * padded.kernel_weights[i];
                    weight_sum += padded.kernel_weights[i];
                }
            }
        } else {
            for i in 0..padded.kernel_offsets.len() {
                let v = unsafe {
                    *padded_slice
                        .as_ptr()
                        .offset(base + padded.kernel_offsets[i])
                };
                acc += v * padded.kernel_weights[i];
                weight_sum += padded.kernel_weights[i];
            }
        }

        out_slice[out_linear] = if weight_sum == 0.0 {
            f64::NAN
        } else {
            acc / weight_sum
        };
        out_linear += 1;

        // increment output index
        let mut d = padded.ndim;
        loop {
            d -= 1;

            padded.idx[d] += 1;
            base += padded_strides[d];

            if padded.idx[d] < padded.out_shape[d] {
                break;
            }

            padded.idx[d] = 0;
            base -= (padded.out_shape[d] as isize) * padded_strides[d];
            if d == 0 {
                return;
            }
        }
    }
}
