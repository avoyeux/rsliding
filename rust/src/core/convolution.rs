// ! N-dimensional convolution operation.

use ndarray::ArrayViewMutD;

// local
use crate::core::padding::SlidingWorkspace;

// todo need to see if the results are right.

/// N-dimensional convolution for a kernel with weights and an input array with NaNs.
/// The NaN values in the input are ignored in the convolution operation.
/// If no valid values in the kernel window, the output is set to NaN.
pub fn convolution<'a>(padded: &mut SlidingWorkspace, mut data: ArrayViewMutD<'a, f64>) {
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
        let mut has_valid = false;
        let mut acc = 0.0;

        if has_nan {
            for i in 0..padded.kernel_offsets.len() {
                let v = unsafe {
                    *padded_slice
                        .as_ptr()
                        .offset(base + padded.kernel_offsets[i])
                };
                if !v.is_nan() {
                    acc += v * padded.kernel_weights[i];
                    has_valid = true;
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
            }
        }

        out_slice[out_linear] = if !has_valid { f64::NAN } else { acc };
        out_linear += 1;

        // increment output index
        let mut d = padded.ndim;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;

            padded.idx[d] += 1;
            base += padded_strides[d];

            if padded.idx[d] < padded.out_shape[d] {
                break;
            }

            padded.idx[d] = 0;
            base -= (padded.out_shape[d] as isize) * padded_strides[d];
        }
    }
}
