use ndarray::{ArrayD, ArrayViewD, IxDyn};
use num_traits::Zero;
use std::ops::{AddAssign, Mul};

// todo add checks on shapes and ndim later
// todo add padding and stride (?) later

// N dimensional convolution
pub fn convolution<T>(input: ArrayViewD<'_, T>, kernel: ArrayViewD<'_, T>) -> ArrayD<T>
where
    T: Copy + Zero + AddAssign + Mul<Output = T>,  // ? Do I need Copy ?
{
    let ndim = input.ndim();

    let input_shape = input.shape();
    let kernel_shape = kernel.shape();
    let mut out_shape_vec = Vec::with_capacity(ndim);
    for d in 0..ndim { // ? after padding, is this needed ?
        let n = input_shape[d];
        let k = kernel_shape[d];
        out_shape_vec.push(n - k + 1);
    }

    let out_shape = IxDyn(&out_shape_vec);
    let mut output = ArrayD::<T>::zeros(out_shape.clone());

    let mut in_idx = vec![0usize; ndim];

    for out_idx in ndarray::indices(out_shape.clone()) {
        let mut acc = T::zero();

        for k_idx in ndarray::indices(kernel.raw_dim()) {
            for d in 0..ndim {
                in_idx[d] = out_idx[d] + k_idx[d];
            }

            let input_val = *input.get(IxDyn(&in_idx)).unwrap();
            let kernel_val = *kernel.get(k_idx).unwrap();
            acc += input_val * kernel_val;
        }
        *output.get_mut(out_idx).unwrap() = acc;
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
