// ! N-dimensional convolution operation.

use ndarray::{ArrayD, IxDyn};
use num_traits::{Zero, Float};
use std::ops::AddAssign;

// local
use crate::utils::{PaddingMode, PaddingWorkspace};


/// N-dimensional convolution for a kernel with weights and an input array with NaNs.
pub fn convolution<T>(padded: PaddingWorkspace<T>) -> PaddingWorkspace<T>
where
    T: Float + Zero + AddAssign,
{

    // padded data
    let mut output = padded.data;
    let mut in_idx = vec![0usize; padded.ndim];
    let out_raw_dim = output.raw_dim();
    let kernel_raw_dim = padded.kernel.raw_dim();

    // iterate over output
    for out_idx in ndarray::indices(out_raw_dim) {
        // count
        let mut acc = T::zero();
        let mut has_valid = false;

        // iterate over kernel
        for k_idx in ndarray::indices(kernel_raw_dim.clone()) {
            // input index
            for d in 0..padded.ndim {
                in_idx[d] = out_idx[d] + k_idx[d];
            }

            // no bounds check
            unsafe {
                let input_val = *padded.data.uget(IxDyn(&in_idx));
                let kernel_val = *padded.kernel.uget(k_idx);

                if !input_val.is_nan() {
                    acc += input_val * kernel_val;
                    has_valid = true;
                }
            }
        }
        // no bounds check
        unsafe {
            *output.uget_mut(out_idx) = if has_valid { acc } else { T::nan() };
        }
    }
    padded
}

pub fn testing_something<T: Default + Float + Zero>(data: ArrayD<T>, kernel: ArrayD<T>) -> Result<(), String> {
    let padding_mode = PaddingMode::Constant(T::zero());
    let mut pad_ws = PaddingWorkspace::new(data.shape(), kernel.shape(), padding_mode)?;
    pad_ws.pad_input(data.view());

    // convolution writes directly into the reusable output buffer
    let mut output = pad_ws.output_view_mut();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn convolution_1d_valid() {
        // Simple 1D example: [1, 2, 3, 4] * [1, 0, -1] (valid)
        let input = array![1.0f32, 2.0, 3.0, 4.0].into_dyn();
        let kernel = array![1.0f32, 0.0, -1.0].into_dyn();

        let result = convolution(input.view(), kernel.view());

        // Manual computation:
        // pos0: 1*1 + 2*0 + 3*(-1) = -2
        // pos1: 2*1 + 3*0 + 4*(-1) = -2
        let expected = array![-2.0f32, -2.0f32].into_dyn();
        assert_eq!(result, expected);
    }

    #[test]
    fn convolution_2d_valid() {
        // 2D input 3x3, kernel 2x2, valid convolution → output 2x2
        let input = array![
            [1.0f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
        .into_dyn();

        let kernel = array![
            [1.0f32, 0.0],
            [0.0, -1.0],
        ]
        .into_dyn();

        let result = convolution(input.view(), kernel.view());

        // Manual computation:
        // top-left: 1*1 + 2*0 + 4*0 + 5*(-1) = -4
        // top-right: 2*1 + 3*0 + 5*0 + 6*(-1) = -4
        // bottom-left: 4*1 + 5*0 + 7*0 + 8*(-1) = -4
        // bottom-right: 5*1 + 6*0 + 8*0 + 9*(-1) = -4
        let expected = array![
            [-4.0f32, -4.0f32],
            [-4.0f32, -4.0f32],
        ]
        .into_dyn();

        assert_eq!(result, expected);
    }
}
