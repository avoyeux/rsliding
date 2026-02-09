//! To compute the sliding mean of an n-dimensional array with possible NaN values.

use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};

// local
use crate::core::padding::PaddingWorkspace;

/// N-dimensional sliding mean operation.
/// Handles NaN values by ignoring them in the mean calculation.
pub fn slow_mean<'a>(
    padded: &PaddingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    kernel: ArrayViewD<'a, f64>,
) {
    let mut padded_idx = vec![0usize; padded.ndim];
    let kernel_raw_dim = kernel.raw_dim();

    // iterate over input indices
    for input_idx in ndarray::indices(padded.valid_shape.clone()) {
        // count
        let mut acc = 0.;
        let mut weighted_sum = 0.;

        // iterate over kernel
        for k_idx in ndarray::indices(kernel_raw_dim.clone()) {
            // compute padded index
            for d in 0..padded.ndim {
                padded_idx[d] = input_idx[d] + k_idx[d];
            }

            // no bounds check
            unsafe {
                let input_val = *padded.padded_buffer.uget(IxDyn(&padded_idx));
                let kernel_val = *kernel.uget(k_idx);

                if !input_val.is_nan() && kernel_val != 0. {
                    // ? should I add a 0. check for kernel_val ?
                    acc += input_val * kernel_val;
                    weighted_sum += kernel_val;
                }
            }
        }
        // no bounds check
        unsafe {
            *data.uget_mut(input_idx) = if weighted_sum == 0. {
                f64::NAN
            } else {
                acc / weighted_sum
            };
        }
    }
}

// pub fn sliding_mean<'a>(
//     padded: &PaddingWorkspace,
//     mut data: ArrayViewMutD<'a, f64>,
//     kernel: ArrayViewD<'a, f64>,
// ) {
//     let ndim = data.ndim();
//     let out_shape = data.shape().to_vec();

//     // Fast path requires contiguous buffers.
//     let Some(padded_slice) = padded.padded_buffer.as_slice_memory_order() else {
//         // Fallback to current implementation if not contiguous.
//         return slow_mean(padded, data, kernel);
//     };
//     let Some(out_slice) = data.as_slice_memory_order_mut() else {
//         return slow_mean(padded, data, kernel);
//     };

//     // Strides are in elements (isize) in ndarray.
//     let pstrides = padded.padded_buffer.strides();

//     // Compute pad from kernel shape (pad = k/2).
//     let kshape = kernel.shape();
//     let mut pad = Vec::with_capacity(ndim);
//     for &k in kshape {
//         pad.push(k / 2);
//     }

//     // Precompute kernel offsets + weights (skip zeros).
//     let mut k_offsets: Vec<isize> = Vec::with_capacity(kernel.len());
//     let mut k_weights: Vec<f64> = Vec::with_capacity(kernel.len());
//     for k_idx in ndarray::indices(kernel.raw_dim()) {
//         let mut off = 0isize;
//         for d in 0..ndim {
//             off += (k_idx[d] as isize) * pstrides[d];
//         }

//         let w = unsafe { *kernel.uget(k_idx.clone()) };
//         if w != 0.0 {
//             k_offsets.push(off);
//             k_weights.push(w);
//         }
//     }

//     // Base offset in padded buffer for output index = 0.
//     let mut base = 0isize;

//     // Manual N-D index increment (row-major).
//     let mut idx = vec![0usize; ndim];
//     let mut out_linear = 0usize;

//     loop {
//         let mut acc = 0.0;
//         let mut wsum = 0.0;

//         for i in 0..k_offsets.len() {
//             let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
//             if !v.is_nan() {
//                 acc += v * k_weights[i];
//                 wsum += k_weights[i];
//             }
//         }

//         out_slice[out_linear] = if wsum == 0.0 { f64::NAN } else { acc / wsum };
//         out_linear += 1;

//         // Increment multi-index and base offset.
//         let mut d = ndim;
//         loop {
//             if d == 0 {
//                 return;
//             }
//             d -= 1;

//             idx[d] += 1;
//             base += pstrides[d];

//             if idx[d] < out_shape[d] {
//                 break;
//             }

//             // Reset this dimension and carry.
//             idx[d] = 0;
//             base -= (out_shape[d] as isize) * pstrides[d];
//         }
//     }
// }


pub fn sliding_mean<'a>(
    padded: &PaddingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    kernel: ArrayViewD<'a, f64>,
) {
    let ndim = data.ndim();
    let out_shape = data.shape().to_vec();
    let kshape = kernel.shape().to_vec();
    let has_nan = data.iter().any(|v| v.is_nan());

    let Some(padded_slice) = padded.padded_buffer.as_slice_memory_order() else {
        return slow_mean(padded, data, kernel);
    };
    let Some(out_slice) = data.as_slice_memory_order_mut() else {
        return slow_mean(padded, data, kernel);
    };
    let Some(kernel_slice) = kernel.as_slice_memory_order() else {
        return slow_mean(padded, data, kernel);
    };

    let pstrides = padded.padded_buffer.strides(); // element strides
    let kstrides = kernel.strides();               // element strides

    // Precompute kernel offsets + weights (skip zeros).
    let mut k_offsets: Vec<isize> = Vec::with_capacity(kernel_slice.len());
    let mut k_weights: Vec<f64> = Vec::with_capacity(kernel_slice.len());

    // Manual multi-index over kernel
    let mut k_idx = vec![0usize; ndim];
    let mut k_base = 0isize;
    loop {
        let w = kernel_slice[k_base as usize];
        if w != 0.0 {
            // Convert kernel multi-index into padded offset via strides
            let mut off = 0isize;
            for d in 0..ndim {
                off += (k_idx[d] as isize) * pstrides[d];
            }
            k_offsets.push(off);
            k_weights.push(w);
        }

        // increment kernel index
        let mut d = ndim;
        loop {
            if d == 0 {
                break;
            }
            d -= 1;

            k_idx[d] += 1;
            k_base += kstrides[d];

            if k_idx[d] < kshape[d] {
                break;
            }

            k_idx[d] = 0;
            k_base -= (kshape[d] as isize) * kstrides[d];
            if d == 0 {
                break;
            }
        }

        if k_idx.iter().all(|&x| x == 0) {
            // we wrapped around after finishing the last index
            break;
        }
    }

    // Optional: skip NaN checks if there are none (extra pass)
    

    // Output loop (manual N-D)
    let mut idx = vec![0usize; ndim];
    let mut base = 0isize;
    let mut out_linear = 0usize;

    loop {
        let mut acc = 0.0;
        let mut wsum = 0.0;

        if has_nan {
            for i in 0..k_offsets.len() {
                let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                if !v.is_nan() {
                    acc += v * k_weights[i];
                    wsum += k_weights[i];
                }
            }
        } else {
            for i in 0..k_offsets.len() {
                let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
                acc += v * k_weights[i];
                wsum += k_weights[i];
            }
        }

        out_slice[out_linear] = if wsum == 0.0 { f64::NAN } else { acc / wsum };
        out_linear += 1;

        // increment output index
        let mut d = ndim;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;

            idx[d] += 1;
            base += pstrides[d];

            if idx[d] < out_shape[d] {
                break;
            }

            idx[d] = 0;
            base -= (out_shape[d] as isize) * pstrides[d];
        }
    }
}