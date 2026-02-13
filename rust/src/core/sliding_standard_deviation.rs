//! N-dimensional sliding standard deviation operation with NaN values and a weighted kernel.
//! Also computes the sliding mean at the same time.

use ndarray::ArrayViewMutD;
use rayon::prelude::*;

// local
use crate::core::padding::SlidingWorkspace;

/// N-dimensional sliding standard deviation operation with NaN values and a weighted kernel.
/// NaN values are ignored.
/// If no valid values inside a kernel window, the output is set to NaN.
/// Gives the sliding standard deviation and the sliding mean at the same time.
pub fn sliding_standard_deviation_old<'a>(
    padded: &SlidingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    mut mean_buffer: ArrayViewMutD<'a, f64>,
) {
    // reset kernel index buffer
    let padded_strides = padded.padded_buffer.strides();
    // Assume everything is contiguous and abort early if it is not.
    let padded_slice = padded.padded_buffer.as_slice_memory_order().unwrap();
    let has_nan = padded_slice.iter().any(|v| v.is_nan());
    let out_slice = data.as_slice_memory_order_mut().unwrap();
    let mean_slice = mean_buffer.as_slice_memory_order_mut().unwrap();

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

            *out = if n == 0 { // wrong, should be sum of weights
                f64::NAN
            } else {
                (m2 / (n as f64)).sqrt()
            };
            *mean_out = if n == 0 { f64::NAN } else { mean };
        });
}

use crate::core::sliding_mean::{neumaier_add, sliding_mean};

// trying more stable approach
pub fn sliding_standard_deviation<'a>(
    padded: &SlidingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    mut mean_buffer: ArrayViewMutD<'a, f64>,
) {
    // update mean buffer
    sliding_mean(padded, mean_buffer.view_mut());

    // reset kernel index buffer
    let padded_strides = padded.padded_buffer.strides();
    // Assume everything is contiguous and abort early if it is not.
    let padded_slice = padded.padded_buffer.as_slice_memory_order().unwrap();
    let has_nan = padded_slice.iter().any(|v| v.is_nan());
    let out_slice = data.as_slice_memory_order_mut().unwrap();
    let mean_slice = mean_buffer.as_slice_memory_order().unwrap();

    let k_offsets = &padded.kernel_offsets;
    let k_weights = &padded.kernel_weights;

    out_slice
        .par_iter_mut()
        .zip(mean_slice)
        .enumerate()
        .for_each(|(out_linear, (out, mean))| {
            let base = padded.base_offset_from_linear(out_linear, padded_strides);

            let mut sum_weights = 0.0;
            let mut sum = 0.0;
            let mut c = 0.0;
            let mut c_w = 0.0;

            if has_nan {
                for i in 0..k_offsets.len() {
                    let value = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                    if !value.is_nan() {
                        let delta = value - *mean;
                        // sum += k_weights[i] * delta * delta; // ! old definition was wrong
                        let kernel_value = k_weights[i];
                        neumaier_add(&mut sum_weights, &mut c_w, kernel_value);
                        neumaier_add(&mut sum, &mut c, kernel_value * delta * delta);
                    }
                }
            } else {
                for i in 0..k_offsets.len() {
                    let value = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                    let delta = value - *mean;
                    // sum += k_weights[i] * delta * delta;
                    let kernel_value = k_weights[i];
                    neumaier_add(&mut sum_weights, &mut c_w, kernel_value);
                    neumaier_add(&mut sum, &mut c, kernel_value * delta * delta);
                }
            }

            // *out = if sum_weights == 0.0 {
            //     f64::NAN
            // } else {
            //     (sum / (sum_weights as f64)).sqrt()
            // };
            *out = if sum_weights == 0.0 {
                f64::NAN
            } else {
                ((sum + c) / (sum_weights + c_w)).sqrt()
            };
        });
}