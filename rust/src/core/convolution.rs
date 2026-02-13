//! N-dimensional convolution operation with NaN handling and kernel weights.
//! Actually does a correlation operation so keep in mind to flip the kernel prior to using the
//! function.

use ndarray::ArrayViewMutD;
use rayon::prelude::*;

// local
use crate::core::padding::SlidingWorkspace;

// todo need to see if the results are right.

/// N-dimensional convolution for a kernel with weights and an input array with NaNs.
/// The NaN values in the input are ignored in the convolution operation.
/// If no valid values in the kernel window, the output is set to NaN.
/// Keep in mind that the kernel is not flipped and as such should be flipped before creating
/// the sliding workspace if you do not want a cross-correlation operation.
pub fn convolution<'a>(padded: &SlidingWorkspace, mut data: ArrayViewMutD<'a, f64>) {
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
            let mut acc = 0.0;

            if has_nan {
                let mut has_valid = false;

                for i in 0..k_offsets.len() {
                    let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                    if !v.is_nan() {
                        acc += v * k_weights[i];
                        has_valid = true;
                    }
                }
                *out = if !has_valid { f64::NAN } else { acc };
            } else {
                for i in 0..k_offsets.len() {
                    let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                    acc += v * k_weights[i];
                }
                *out = acc;
            }
        });
}
