//! N-dimensional sliding median operation for arrays with possible NaN values.

use ndarray::ArrayViewMutD;
use std::cmp::Ordering;

// local
use crate::core::padding::SlidingWorkspace;

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
pub fn sliding_median<'a>(padded: &mut SlidingWorkspace, mut data: ArrayViewMutD<'a, f64>) {
    padded.idx.fill(0);

    let padded_slice = padded
        .padded_buffer
        .as_slice_memory_order()
        .expect("Padding buffer must be contiguous");
    let out_slice = data
        .as_slice_memory_order_mut()
        .expect("Output view must be contiguous");

    let k_offsets = &padded.kernel_offsets;
    let k_weights = &padded.kernel_weights;
    assert_eq!(k_offsets.len(), k_weights.len());

    let pstrides = padded.padded_buffer.strides();

    let mut window_vals = Vec::with_capacity(k_offsets.len());
    let mut window_weights = Vec::with_capacity(k_offsets.len());

    let mut base = 0isize;
    let mut out_linear = 0usize;

    loop {
        window_vals.clear();
        window_weights.clear();

        for i in 0..k_offsets.len() {
            let w = k_weights[i];
            if w == 0.0 {
                continue;
            }
            let v = unsafe { *padded_slice.as_ptr().offset(base + k_offsets[i]) };
            if v.is_nan() {
                continue;
            }
            window_vals.push(v);
            window_weights.push(w);
        }

        let median = if window_vals.is_empty() {
            f64::NAN
        } else if weights_all_equal(&window_weights) {
            median_partition(&mut window_vals)
        } else {
            weighted_median_partition(&mut window_vals, &mut window_weights)
        };

        out_slice[out_linear] = median;
        out_linear += 1;

        // advance N-D index just like in sliding_mean
        let mut d = padded.ndim;
        loop {
            if d == 0 {
                return;
            }
            d -= 1;

            padded.idx[d] += 1;
            base += pstrides[d];

            if padded.idx[d] < padded.out_shape[d] {
                break;
            }

            padded.idx[d] = 0;
            base -= (padded.out_shape[d] as isize) * pstrides[d];
        }
    }
}
