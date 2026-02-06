//! N-dimensional sliding median operation for arrays with possible NaN values.

use ndarray::{ArrayViewD, ArrayViewMutD, IxDyn};
use std::cmp::Ordering;

// local
use crate::padding::PaddingWorkspace;

/// Select the (unweighted) median using partitioning.
/// For an even number of elements, returns the average of the two middle values.
fn median_partition(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }

    let mid = n / 2;

    // Partially partition so that values[mid] is the element that would be there in a full sort.
    values.select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Less));

    if n % 2 == 1 {
        values[mid]
    } else {
        // For even n, median is the average of the two middle order statistics.
        let mut lower_max = values[0];
        for &v in &values[..mid] {
            if v > lower_max {
                lower_max = v;
            }
        }
        0.5 * (lower_max + values[mid])
    }
}

/// Select a weighted median using partitioning (quickselect-style).
/// We use the "half-mass" definition: find m such that the cumulative
/// weight to the left is < total_weight/2 and to the right is <= total_weight/2.
fn weighted_median_partition(values: &mut [f64], weights: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }

    let mut total_weight = 0.0;
    for &w in weights.iter() {
        total_weight += w;
    }
    if total_weight == 0.0 {
        return f64::NAN;
    }

    let mut left = 0usize;
    let mut right = n;
    let mut target = 0.5 * total_weight;

    loop {
        let len = right - left;
        if len == 0 {
            return f64::NAN;
        } else if len == 1 {
            return values[left];
        }

        let pivot_index = left + len / 2;
        let pivot_value = values[pivot_index];

        // 3-way partition: [left..lt)=<pivot, [lt..gt)==pivot, [gt..right)>pivot
        let mut lt = left;
        let mut i = left;
        let mut gt = right;

        while i < gt {
            let v = values[i];
            let ord = v.partial_cmp(&pivot_value).unwrap_or(Ordering::Equal);
            match ord {
                Ordering::Less => {
                    values.swap(lt, i);
                    weights.swap(lt, i);
                    lt += 1;
                    i += 1;
                }
                Ordering::Greater => {
                    gt -= 1;
                    values.swap(i, gt);
                    weights.swap(i, gt);
                }
                Ordering::Equal => {
                    i += 1;
                }
            }
        }

        let mut w_left = 0.0;
        for &w in &weights[left..lt] {
            w_left += w;
        }
        let mut w_pivot = 0.0;
        for &w in &weights[lt..gt] {
            w_pivot += w;
        }

        if target < w_left {
            right = lt;
        } else if target <= w_left + w_pivot {
            return pivot_value;
        } else {
            target -= w_left + w_pivot;
            left = gt;
        }
    }
}

fn weights_all_equal(weights: &[f64]) -> bool {
    if weights.len() <= 1 {
        return true;
    }
    let w0 = weights[0];
    const EPS: f64 = 1e-12;
    for &w in &weights[1..] {
        if (w - w0).abs() > EPS * (1.0 + w0.abs()) {
            return false;
        }
    }
    true
}

/// N-dimensional sliding **weighted** median operation.
/// Uses kernel values as non-negative weights and ignores NaNs.
/// Kernel entries equal to 0 act as a mask (weight 0).
pub fn sliding_weighted_median<'a>(
    padded: &PaddingWorkspace,
    mut data: ArrayViewMutD<'a, f64>,
    kernel: ArrayViewD<'a, f64>,
) {
    let mut padded_idx = vec![0usize; padded.ndim];
    let kernel_raw_dim = kernel.raw_dim();

    let kernel_len = kernel.len();
    let mut values: Vec<f64> = Vec::with_capacity(kernel_len);
    let mut weights: Vec<f64> = Vec::with_capacity(kernel_len);

    for input_idx in ndarray::indices(padded.valid_shape.clone()) {
        values.clear();
        weights.clear();

        for k_idx in ndarray::indices(kernel_raw_dim.clone()) {
            for d in 0..padded.ndim {
                padded_idx[d] = input_idx[d] + k_idx[d];
            }

            unsafe {
                let input_val = *padded.padded_buffer.uget(IxDyn(&padded_idx));
                let kernel_val = *kernel.uget(k_idx);

                if !input_val.is_nan() && kernel_val > 0.0 {
                    values.push(input_val);
                    weights.push(kernel_val);
                }
            }
        }

        let median = if values.is_empty() {
            f64::NAN
        } else if weights_all_equal(&weights) {
            // All (positive) weights equal -> fall back to standard sample median,
            // which for even counts returns the average of the two middle values.
            median_partition(&mut values)
        } else {
            weighted_median_partition(&mut values, &mut weights)
        };

        unsafe {
            *data.uget_mut(input_idx) = median;
        }
    }
}
