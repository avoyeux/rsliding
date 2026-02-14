//! To compute the sliding mean of an n-dimensional array with possible NaN values and kernel
//! weights. Not actually used in the sliding sigma clipping implementation as the sliding
//! standard deviation function already computes the sliding mean.

use ndarray::ArrayViewMutD;
use rayon::prelude::*;

// local
use crate::core::padding::SlidingWorkspace;
use crate::core::utils::neumaier_add;

/// N-dimensional sliding mean operation with NaN values and a weighted kernel.
/// The NaN values are ignored.
/// If no valid data inside a kernel, the corresponding output is set to NaN.
pub fn sliding_mean<'a>(
    workspace: &SlidingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    neumaier: bool,
) {
    let padded_strides = workspace.padded.strides();
    let padded_slice = workspace.padded.as_slice_memory_order().unwrap();
    let has_nan = padded_slice.iter().any(|v| v.is_nan());
    let out_slice = data.as_slice_memory_order_mut().unwrap();

    let k_offsets = &workspace.kernel_offsets;
    let k_weights = &workspace.kernel_weights;

    // a little less stable
    if !neumaier {
        // NaN check (outside of loop for efficiency)
        if has_nan {
            out_slice
                .par_iter_mut()
                .enumerate()
                .for_each(|(out_linear, out)| {
                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;
                    let base = workspace.base_offset_from_linear(out_linear, padded_strides);

                    for i in 0..k_offsets.len() {
                        let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                        if !v.is_nan() {
                            let kernel_value = k_weights[i];
                            sum += v * kernel_value;
                            weight_sum += kernel_value;
                        }
                    }

                    *out = if weight_sum == 0.0 {
                        f64::NAN
                    } else {
                        sum / weight_sum
                    };
                });
        } else {
            out_slice
                .par_iter_mut()
                .enumerate()
                .for_each(|(out_linear, out)| {
                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;
                    let base = workspace.base_offset_from_linear(out_linear, padded_strides);

                    for i in 0..k_offsets.len() {
                        let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                        let kernel_value = k_weights[i];
                        sum += v * kernel_value;
                        weight_sum += kernel_value;
                    }

                    *out = if weight_sum == 0.0 {
                        f64::NAN
                    } else {
                        sum / weight_sum
                    };
                });
        }
    // most stable version possible (uses Neumaier summation)
    } else {
        // NaN check (outside of loop for efficiency)
        if has_nan {
            out_slice
                .par_iter_mut()
                .enumerate()
                .for_each(|(out_linear, out)| {
                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;
                    let mut c = 0.0;
                    let mut c_w = 0.0;
                    let base = workspace.base_offset_from_linear(out_linear, padded_strides);

                    for i in 0..k_offsets.len() {
                        let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                        if !v.is_nan() {
                            let kernel_value = k_weights[i];
                            neumaier_add(&mut sum, &mut c, v * kernel_value);
                            neumaier_add(&mut weight_sum, &mut c_w, kernel_value);
                        }
                    }

                    *out = if weight_sum == 0.0 {
                        f64::NAN
                    } else {
                        (sum + c) / (weight_sum + c_w)
                    };
                });
        } else {
            out_slice
                .par_iter_mut()
                .enumerate()
                .for_each(|(out_linear, out)| {
                    let mut sum = 0.0;
                    let mut weight_sum = 0.0;
                    let mut c = 0.0;
                    let mut c_w = 0.0;
                    let base = workspace.base_offset_from_linear(out_linear, padded_strides);

                    for i in 0..k_offsets.len() {
                        let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                        let kernel_value = k_weights[i];
                        neumaier_add(&mut sum, &mut c, v * kernel_value);
                        neumaier_add(&mut weight_sum, &mut c_w, kernel_value);
                    }

                    *out = if weight_sum == 0.0 {
                        f64::NAN
                    } else {
                        (sum + c) / (weight_sum + c_w)
                    };
                });
        }
    }
}
