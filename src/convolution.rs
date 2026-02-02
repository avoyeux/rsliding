// ! N-dimensional convolution operation.

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use num_traits::{Zero, Float};
use std::ops::AddAssign;

// local
use crate::utils::ConvolutionSetup;

// todo add checks on shapes and ndim later
// todo add padding and stride (?) later

/// N-dimensional convolution for a kernel with weights and an input array with NaNs.
pub fn convolution<T>(data: ArrayViewD<'_, T>, kernel: ArrayViewD<'_, T>) -> ArrayD<T>
where
    T: Float + Zero + AddAssign,
{
    let convolution_setup = ConvolutionSetup::new(data.view(), kernel.view());

    let mut output = unsafe {
        ArrayD::<T>::uninit(convolution_setup.out_raw_dim.clone()).assume_init()
    };
    let mut in_idx = vec![0usize; convolution_setup.ndim];

    for out_idx in ndarray::indices(convolution_setup.out_raw_dim) {
        // count
        let mut acc = T::zero();
        let mut has_valid = false;

        for k_idx in ndarray::indices(convolution_setup.kernel_raw_dim.clone()) {
            // input index
            for d in 0..convolution_setup.ndim {
                in_idx[d] = out_idx[d] + k_idx[d];
            }

            // no bounds check
            unsafe {
                let input_val = *data.uget(IxDyn(&in_idx));
                let kernel_val = *kernel.uget(k_idx);

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
    output
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
