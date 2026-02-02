//! has some utility functions to be used across the crate.
use ndarray::{ArrayViewD, Dim, IxDyn, IxDynImpl};
use num_traits::Float;

pub struct ConvolutionSetup {
    pub ndim: usize,
    pub out_raw_dim: Dim<IxDynImpl>,
    pub kernel_raw_dim: Dim<IxDynImpl>,
}

impl ConvolutionSetup {
    pub fn new<T> (data: ArrayViewD<'_, T>, kernel: ArrayViewD<'_, T>) -> ConvolutionSetup
    where
        T: Float,
    {
        let ndim = data.ndim();
        let input_shape = data.shape();
        let kernel_shape = kernel.shape();

        let mut out_shape_vec = Vec::with_capacity(ndim);
        for d in 0..ndim { // ? after padding, is this needed ?
            out_shape_vec.push(input_shape[d] - kernel_shape[d] + 1);
        }

        let out_raw_dim = IxDyn(&out_shape_vec);
        let kernel_raw_dim = kernel.raw_dim();

        ConvolutionSetup {
            ndim,
            out_raw_dim,
            kernel_raw_dim,
        }
    }
}
