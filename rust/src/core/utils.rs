//! Neumaier summation for improved numerical stability in floating-point summation.

/// Neumaier summation
#[inline(always)]
pub fn neumaier_add(sum: &mut f64, c: &mut f64, value: f64) {
    let t = *sum + value;
    if sum.abs() >= value.abs() {
        *c += (*sum - t) + value;
    } else {
        *c += (value - t) + *sum;
    }
    *sum = t;
}
