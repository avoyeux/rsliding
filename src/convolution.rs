// ! N-dimensional convolution operation.

use ndarray::{IxDyn, ArrayViewD};

// local
use crate::padding::PaddingWorkspace;


/// N-dimensional convolution for a kernel with weights and an input array with NaNs.
/// The NaN values in the input are ignored in the convolution operation.
/// If no valid values in the kernel window, the output is set to NaN.
pub fn convolution<'a>(mut padded: PaddingWorkspace, kernel: ArrayViewD<'a, f64>) -> PaddingWorkspace {

    let mut padded_idx = vec![0usize; padded.ndim];
    let kernel_raw_dim = kernel.raw_dim();

    // iterate over input indices
    for input_idx in ndarray::indices(padded.valid_shape.clone()) {
        // count
        let mut acc = 0.;
        let mut has_valid = false;

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

                if !input_val.is_nan() {
                    acc += input_val * kernel_val;
                    has_valid = true;
                }
            }
        }
        // no bounds check
        unsafe {
            *padded.output_buffer.uget_mut(input_idx) = 
                if !has_valid { f64::NAN } else { acc };
        }
    }
    padded
}
