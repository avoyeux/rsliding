//! N-dimensional sliding standard deviation operation.

use ndarray::ArrayViewMutD;

// local
use crate::core::padding::SlidingWorkspace;

/// N-dimensional sliding standard deviation
/// also keeps the sliding mean as it might be chosen for the sigma clipping
pub fn sliding_standard_deviation<'a>(
    padded: &mut SlidingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    mut mean_buffer: ArrayViewMutD<'a, f64>,
) {
    // reset kernel index buffer
    padded.idx.fill(0);
    let has_nan = data.iter().any(|v| v.is_nan());
    let mut base = 0isize;
    let mut out_linear = 0usize;
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

    loop {
        let mut n = 0usize;
        let mut mean = 0.;
        let mut m2 = 0.;

        if has_nan {
            for i in 0..padded.kernel_offsets.len() {
                let value = unsafe {
                    *padded_slice
                        .as_ptr()
                        .offset(base + padded.kernel_offsets[i])
                };
                let kernel_value = padded.kernel_weights[i];

                // std
                if !value.is_nan() {
                    n += 1;
                    let delta = value - mean;
                    mean += delta * kernel_value / (n as f64);
                    let delta2 = value - mean;
                    m2 += delta * delta2 * kernel_value;
                }
            }
        } else {
            for i in 0..padded.kernel_offsets.len() {
                let value = unsafe {
                    *padded_slice
                        .as_ptr()
                        .offset(base + padded.kernel_offsets[i])
                };
                let kernel_value = padded.kernel_weights[i];

                // std
                n += 1;
                let delta = value - mean;
                mean += delta * kernel_value / (n as f64);
                let delta2 = value - mean;
                m2 += delta * delta2 * kernel_value;
            }
        }

        out_slice[out_linear] = if n == 0 {
            f64::NAN
        } else {
            (m2 / (n as f64)).sqrt()
        };
        mean_slice[out_linear] = if n == 0 { f64::NAN } else { mean };
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
