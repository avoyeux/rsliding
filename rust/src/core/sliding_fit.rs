// ! To compute the sliding fit of an n-dimensional array with possible NaN values and kernel
// ! weights.

use ndarray::ArrayViewMutD;
use rayon::prelude::*;

// local
use crate::core::padding::SlidingWorkspace;
use crate::core::sliding_sigma_clipping::CenterMode;

// todo add docstring
pub fn sliding_fit<'a>(
    workspace: &SlidingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    center_mode: &CenterMode,
    neumaier: bool,
) {
    // center values

    let padded_strides = workspace.padded.strides();
    let padded_slice = workspace.padded.as_slice_memory_order().unwrap();
    let had_nan = padded_slice.iter().any(|v| v.is_nan());
    let out_slice = data.as_slice_memory_order_mut().unwrap();

    let k_offsets = &workspace.kernel_offsets;
    let k_weights = &workspace.kernel_weights;
}
