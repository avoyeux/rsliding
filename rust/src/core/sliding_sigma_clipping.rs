//! N-dimensional sliding sigma clipping operations with NaN handling and kernel weights.

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use rayon::prelude::*;

// local
use crate::core::padding::SlidingWorkspace;
use crate::core::sliding_median::sliding_median;
use crate::core::sliding_standard_deviation::sliding_standard_deviation;

/// Gives the different mode options.
/// Can be Mean or Median (i.e. uses the sliding mean or the sliding median).
pub enum CenterMode {
    Mean,
    Median,
}

/// N-dimensional sigma clipping.
/// The data can contain NaN values and the kernel weights.
/// Values clipped (given the sigma coefficients) are set to the corresponding sliding mode value.
/// Final data array can contain NaN values if the corresponding final mode value is set to NaN.
/// At the same time, initial NaN values might become an f64 value if the corresponding mode is not
/// a NaN.
/// Returns a mask giving the positions where a value was swapped to the corresponding mode.
pub fn sliding_sigma_clipping<'a>(
    padded: &mut SlidingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    sigma_upper: &Option<f64>,
    sigma_lower: &Option<f64>,
    center_mode: &CenterMode,
    max_iterations: &Option<usize>,
) -> ArrayD<bool> {
    let mut iterations: usize = 0;
    let mut mode_buffer = data.to_owned();
    let mut std_buffer = data.to_owned();

    loop {
        // std
        sliding_standard_deviation(padded, std_buffer.view_mut(), mode_buffer.view_mut());

        // center
        match center_mode {
            CenterMode::Mean => (),
            CenterMode::Median => {
                sliding_median(padded, mode_buffer.view_mut());
            }
        }

        // update data n changed
        let changed = clipping(
            data.view_mut(),
            mode_buffer.view(),
            std_buffer.view(),
            sigma_lower,
            sigma_upper,
        );

        // stop if no change or max iterations reached
        if !changed {
            break;
        }
        iterations += 1;
        if let Some(max_iter) = max_iterations {
            if iterations >= *max_iter {
                break;
            }
        }

        // update padded buffer
        padded.pad_input(data.view());
    }
    // mask of changed values
    fill_n_mask(data.view_mut(), mode_buffer.view())
}

/// Clips the data given the mode, the sigma_lower and sigma_upper coefficients.
/// The clipped data is set as NaN.
/// If something is clipped, returns true, else returns false.
fn clipping(
    mut data: ArrayViewMutD<f64>,
    mode: ArrayViewD<f64>,
    std: ArrayViewD<f64>,
    sigma_lower: &Option<f64>,
    sigma_upper: &Option<f64>,
) -> bool {
    let data_slice = data.as_slice_memory_order_mut().unwrap();
    let mode_slice = mode.as_slice_memory_order().unwrap();
    let std_slice = std.as_slice_memory_order().unwrap();

    data_slice
        .par_iter_mut()
        .zip(mode_slice)
        .zip(std_slice)
        .map(|((x, &mu), &s)| {
            let diff = *x - mu;
            let mut outlier = false;

            if let Some(sl) = sigma_lower {
                outlier |= diff < -sl * s;
            }
            if let Some(su) = sigma_upper {
                outlier |= diff > su * s;
            }

            if outlier {
                *x = f64::NAN;
            }

            outlier
        })
        .reduce(|| false, |a, b| a || b)
}

/// To update the data and mask after the sigma clipping.
/// The data values that have NaN values are swapped with the final mode of the corresponding
/// window.
/// The mask represents the positions where the data was changed to the corresponding mode.
/// Keep in mind that the mode might still be a NaN value.
fn fill_n_mask(mut data: ArrayViewMutD<f64>, mode: ArrayViewD<f64>) -> ArrayD<bool> {
    // mask
    let mut mask = ArrayD::from_elem(data.raw_dim(), false);
    let mask_slice = mask.as_slice_memory_order_mut().unwrap();
    let data_slice = data.as_slice_memory_order_mut().unwrap();
    let mode_slice = mode.as_slice_memory_order().unwrap();

    mask_slice
        .par_iter_mut()
        .zip(data_slice)
        .zip(mode_slice)
        .for_each(|((m, x), &mu)| {
            let is_nan = x.is_nan();

            *m = is_nan;
            if is_nan {
                *x = mu;
            }
        });
    mask
}
