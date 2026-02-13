//! To compute the sliding mean of an n-dimensional array with possible NaN values and kernel
//! weights. Not actually used in the sliding sigma clipping implementation as the sliding
//! standard deviation function already computes the sliding mean.

use ndarray::ArrayViewMutD;
use rayon::prelude::*;

// local
use crate::core::padding::SlidingWorkspace;

/// Neumaier addition
#[inline(always)]
pub fn neumaier_add(sum: &mut f64, c: &mut f64, value: f64) {
    let t = *sum + value;
    if sum.abs() >= value.abs() {
        *c += (*sum - t) + value;
    } else {
        *c += (value - t) + *sum;
    }
    *sum = t;
}

/// N-dimensional sliding mean operation with NaN values and a weighted kernel.
/// The NaN values are ignored.
/// If no valid data inside a kernel, the corresponding output is set to NaN.
pub fn sliding_mean<'a>(padded: &SlidingWorkspace, mut data: ArrayViewMutD<'a, f64>) {
    let padded_strides = padded.padded_buffer.strides();
    let padded_slice = padded.padded_buffer.as_slice_memory_order().unwrap();
    let has_nan = padded_slice.iter().any(|v| v.is_nan());
    let out_slice = data.as_slice_memory_order_mut().unwrap();

    let k_offsets = &padded.kernel_offsets;
    let k_weights = &padded.kernel_weights;

    out_slice
        .par_iter_mut()
        .enumerate()
        .for_each(|(out_linear, out)| {
            let base = padded.base_offset_from_linear(out_linear, padded_strides);

            // let mut acc = 0.0;
            let mut weight_sum = 0.0;
            let mut sum = 0.0;
            let mut c = 0.0;
            let mut c_w = 0.0;

            if has_nan {
                for i in 0..k_offsets.len() {
                    let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                    if !v.is_nan() {
                        let kernel_value = k_weights[i];
                        // acc += v * k_weights[i];
                        neumaier_add(&mut sum, &mut c, v * kernel_value);
                        neumaier_add(&mut weight_sum, &mut c_w, kernel_value);
                        // weight_sum += k_weights[i];
                    }
                }
            } else {
                for i in 0..k_offsets.len() {
                    let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                        let kernel_value = k_weights[i];
                        // acc += v * k_weights[i];
                        neumaier_add(&mut sum, &mut c, v * kernel_value);
                        neumaier_add(&mut weight_sum, &mut c_w, kernel_value);
                        // weight_sum += k_weights[i];
                }
            }

            *out = if weight_sum == 0.0 {
                f64::NAN
            } else {
                (sum + c) / (weight_sum + c_w)
            };

            // *out = if weight_sum == 0.0 {
            //     f64::NAN
            // } else {
            //     acc / weight_sum
            // };
        });
}
