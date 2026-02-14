//! N-dimensional sliding standard deviation operation with NaN values and a weighted kernel.
//! Also computes the sliding mean at the same time.

use ndarray::ArrayViewMutD;
use rayon::prelude::*;

// local
use crate::core::padding::SlidingWorkspace;
use crate::core::sliding_mean::sliding_mean;
use crate::core::utils::neumaier_add;

/// N-dimensional sliding standard deviation operation with NaN values and a weighted kernel.
/// NaN values are ignored.
/// If no valid values inside a kernel window, the output is set to NaN.
/// Gives the sliding standard deviation and the sliding mean at the same time.
pub fn sliding_standard_deviation<'a>(
    workspace: &SlidingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    mut mean_buffer: ArrayViewMutD<'a, f64>,
    neumaier: bool,
) {
    // update mean buffer
    sliding_mean(workspace, mean_buffer.view_mut(), neumaier);

    // reset kernel index buffer
    let padded_strides = workspace.padded.strides();
    // Assume everything is contiguous and abort early if it is not.
    let padded_slice = workspace.padded.as_slice_memory_order().unwrap();
    let has_nan = padded_slice.iter().any(|v| v.is_nan());
    let out_slice = data.as_slice_memory_order_mut().unwrap();
    let mean_slice = mean_buffer.as_slice_memory_order().unwrap();

    let k_offsets = &workspace.kernel_offsets;
    let k_weights = &workspace.kernel_weights;

    // a little less stable
    if !neumaier {
        // NaN check (outside of loop for efficiency)
        if has_nan {
            out_slice
                .par_iter_mut()
                .zip(mean_slice)
                .enumerate()
                .for_each(|(out_linear, (out, mean))| {
                    let mut sum = 0.0;
                    let mut sum_weights = 0.0;
                    let base = workspace.base_offset_from_linear(out_linear, padded_strides);

                    for i in 0..k_offsets.len() {
                        let value = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                        if !value.is_nan() {
                            let kernel_value = k_weights[i];
                            let delta = value - *mean;
                            sum += kernel_value * delta * delta;
                            sum_weights += kernel_value;
                        }
                    }

                    *out = if sum_weights == 0.0 {
                        f64::NAN
                    } else {
                        (sum / sum_weights).sqrt()
                    };
                });
        } else {
            out_slice
                .par_iter_mut()
                .zip(mean_slice)
                .enumerate()
                .for_each(|(out_linear, (out, mean))| {
                    let mut sum = 0.0;
                    let mut sum_weights = 0.0;
                    let base = workspace.base_offset_from_linear(out_linear, padded_strides);

                    for i in 0..k_offsets.len() {
                        let value = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                        let kernel_value = k_weights[i];
                        let delta = value - *mean;
                        sum += kernel_value * delta * delta;
                        sum_weights += kernel_value;
                    }

                    *out = if sum_weights == 0.0 {
                        f64::NAN
                    } else {
                        (sum / sum_weights).sqrt()
                    };
                });
        }
    // most stable version possible (uses Neumaier summation)
    } else {
        // NaN check (outside of loop for efficiency)
        if has_nan {
            out_slice
                .par_iter_mut()
                .zip(mean_slice)
                .enumerate()
                .for_each(|(out_linear, (out, mean))| {
                    let mut sum = 0.0;
                    let mut sum_weights = 0.0;
                    let mut c = 0.0;
                    let mut c_w = 0.0;
                    let base = workspace.base_offset_from_linear(out_linear, padded_strides);

                    for i in 0..k_offsets.len() {
                        let value = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                        if !value.is_nan() {
                            let kernel_value = k_weights[i];
                            let delta = value - *mean;
                            neumaier_add(&mut sum, &mut c, kernel_value * delta * delta);
                            neumaier_add(&mut sum_weights, &mut c_w, kernel_value);
                        }
                    }

                    *out = if sum_weights == 0.0 {
                        f64::NAN
                    } else {
                        ((sum + c) / (sum_weights + c_w)).sqrt()
                    };
                });
        } else {
            out_slice
                .par_iter_mut()
                .zip(mean_slice)
                .enumerate()
                .for_each(|(out_linear, (out, mean))| {
                    let mut sum = 0.0;
                    let mut sum_weights = 0.0;
                    let mut c = 0.0;
                    let mut c_w = 0.0;
                    let base = workspace.base_offset_from_linear(out_linear, padded_strides);

                    for i in 0..k_offsets.len() {
                        let value = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                        let kernel_value = k_weights[i];
                        let delta = value - *mean;
                        neumaier_add(&mut sum, &mut c, kernel_value * delta * delta);
                        neumaier_add(&mut sum_weights, &mut c_w, kernel_value);
                    }

                    *out = if sum_weights == 0.0 {
                        f64::NAN
                    } else {
                        ((sum + c) / (sum_weights + c_w)).sqrt()
                    };
                });
        }
    }
}
