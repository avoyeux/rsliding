//! N-dimensional sliding sigma clipping operations with NaN handling.

use ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Zip};

// local
use crate::core::padding::PaddingWorkspace;
use crate::core::sliding_median::sliding_median;
use crate::core::sliding_standard_deviation::sliding_standard_deviation;

pub enum CenterMode {
    Mean,
    Median,
}

pub fn sliding_sigma_clipping<'a>(
    padded: &mut PaddingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    kernel: ArrayViewD<'a, f64>,
    sigma_upper: &Option<f64>,
    sigma_lower: &Option<f64>,
    center_mode: &CenterMode,
    max_iterations: &Option<usize>,
) {
    let mut iterations: usize = 0;
    let mut mode_buffer = data.to_owned();
    let mut std_buffer = data.to_owned();
    let mut mask_buffer = ArrayD::from_elem(data.raw_dim(), false);

    loop {
        // std
        sliding_standard_deviation(
            padded,
            std_buffer.view_mut(),
            mode_buffer.view_mut(),
            kernel.view(),
        );

        // center
        match center_mode {
            CenterMode::Mean => (),
            CenterMode::Median => {
                sliding_median(padded, mode_buffer.view_mut(), kernel.view());
            }
        }

        // update data n mask
        clipping(
            mask_buffer.view_mut(),
            data.view_mut(),
            mode_buffer.view(),
            std_buffer.view(),
            sigma_lower,
            sigma_upper,
        );

        // stop if no change or max iterations reached
        let changed = mask_buffer.iter().any(|&m| m);
        if !changed {
            break;
        }
        iterations += 1;
        if let Some(max_iter) = max_iterations {
            if iterations >= *max_iter {
                break;
            }
        }

        // reset mask
        mask_buffer.fill(false);

        // update padded buffer
        padded.pad_input(data.view());
    }
}

fn clipping(
    mask: ArrayViewMutD<bool>,
    data: ArrayViewMutD<f64>,
    mode: ArrayViewD<f64>,
    std: ArrayViewD<f64>,
    sigma_lower: &Option<f64>,
    sigma_upper: &Option<f64>,
) {
    Zip::from(mask)
        .and(data)
        .and(mode)
        .and(std)
        .for_each(|m, x, &mu, &s| {
            let diff = *x - mu;
            let mut outlier = false;

            if let Some(sl) = sigma_lower {
                outlier |= diff < -sl * s;
            }
            if let Some(su) = sigma_upper {
                outlier |= diff > su * s;
            }

            *m = *m || outlier;

            if outlier {
                *x = f64::NAN;
            }
        });
}
