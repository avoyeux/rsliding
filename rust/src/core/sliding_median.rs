//! N-dimensional sliding weighted median operation for arrays with NaN values and a weighted kernel.

use ndarray::ArrayViewMutD;
use rayon::prelude::*;
use std::cmp::Ordering;

// local
use crate::core::padding::SlidingWorkspace;

/// N-dimensional sliding **weighted** median operation.
/// Uses kernel values as non-negative weights and ignores NaNs.
/// Kernel entries equal to 0 act as a mask (weight 0).
/// Uses a non weighted partition when all weights (expect the filtered 0. values) are the same.
pub fn sliding_median<'a>(workspace: &SlidingWorkspace, mut data: ArrayViewMutD<'a, f64>) {
    let padded_strides = workspace.padded.strides();
    let padded_slice = workspace.padded.as_slice_memory_order().unwrap();
    let out_slice = data.as_slice_memory_order_mut().unwrap();
    let k_offsets = &workspace.kernel_offsets;
    let k_weights = &workspace.kernel_weights;

    out_slice.par_iter_mut().enumerate().for_each_init(
        || {
            (
                // buffer for each thread
                Vec::with_capacity(k_offsets.len()),
                Vec::with_capacity(k_offsets.len()),
            )
        },
        |(window_vals, window_weights), (out_linear, out)| {
            // reset thread window buffers
            window_vals.clear();
            window_weights.clear();

            let base = workspace.base_offset_from_linear(out_linear, padded_strides);

            for i in 0..k_offsets.len() {
                let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };

                if !v.is_nan() {
                    window_vals.push(v);
                    window_weights.push(k_weights[i]);
                }
            }

            let median = if window_vals.is_empty() {
                f64::NAN
            } else if weights_all_equal(&window_weights) {
                median_partition(window_vals)
            } else {
                weighted_median_partition(window_vals, window_weights)
            };

            *out = median;
        },
    );
}

/// Checks if the kernel weights (expect the filtered 0. values) are all equal.
fn weights_all_equal(weights: &[f64]) -> bool {
    if weights.len() <= 1 {
        return true;
    }
    let w0 = weights[0];
    let max_w = weights.iter().map(|w| w.abs()).fold(1.0, f64::max);
    let eps = f64::EPSILON * max_w;
    for &w in &weights[1..] {
        if (w - w0).abs() > eps * (1.0 + w0.abs()) {
            return false;
        }
    }
    true
}

/// Select the (unweighted) median using partitioning.
/// For an even number of elements, returns the average of the two middle values.
/// Way more efficient than the weighted case.
fn median_partition(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }

    let mid = n / 2;

    // partition
    values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));

    if n % 2 == 1 {
        values[mid]
    } else {
        // average is n even
        let mut lower_max = values[0];
        for &v in &values[..mid] {
            if v > lower_max {
                lower_max = v;
            }
        }
        0.5 * (lower_max + values[mid])
    }
}

/// Weighted median with "midpoint on exact half-mass" behavior:
/// if cumulative weight hits exactly 50%, returns average of current and next value.
/// This matches the usual unweighted even-length median when all weights are 1.
fn weighted_median_partition(values: &mut [f64], weights: &mut [f64]) -> f64 {
    if values.is_empty() || values.len() != weights.len() {
        return f64::NAN;
    }

    let mut pairs: Vec<(f64, f64)> = values
        .iter()
        .copied()
        .zip(weights.iter().copied())
        .filter(|(_, w)| *w > 0.0 && w.is_finite())
        .collect();

    if pairs.is_empty() {
        return f64::NAN;
    }

    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let total_weight: f64 = pairs.iter().map(|(_, w)| *w).sum();
    if total_weight <= 0.0 || !total_weight.is_finite() {
        return f64::NAN;
    }

    let half = 0.5 * total_weight;
    let eps = f64::EPSILON * total_weight.max(1.0);

    let mut cum = 0.0;
    for i in 0..pairs.len() {
        cum += pairs[i].1;

        if cum + eps < half {
            continue;
        }

        // Exact half-mass: return midpoint with next value if available.
        if (cum - half).abs() <= eps && i + 1 < pairs.len() {
            return 0.5 * (pairs[i].0 + pairs[i + 1].0);
        }

        return pairs[i].0;
    }

    pairs[pairs.len() - 1].0
}
