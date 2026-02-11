//! 'sliding' library containing different sliding window operations.
//! Done to replace the 'sliding' python library that I had created before.

use pyo3::prelude::*;

// Local modules
mod bindings;
mod core;

// Re-exports
pub use core::convolution::convolution;
pub use core::padding::{PaddingMode, SlidingWorkspace};
pub use core::sliding_mean::sliding_mean;
pub use core::sliding_median::sliding_median;
pub use core::sliding_sigma_clipping::{CenterMode, sliding_sigma_clipping};
pub use core::sliding_standard_deviation::sliding_standard_deviation;

// Python bindings
#[pymodule]
fn _bindings(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bindings::padding::py_padding, m)?)?;
    m.add_function(wrap_pyfunction!(bindings::convolution::py_convolution, m)?)?;
    m.add_function(wrap_pyfunction!(
        bindings::sliding_mean::py_sliding_mean,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::sliding_median::py_sliding_median,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::sliding_standard_deviation::py_sliding_standard_deviation,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bindings::sliding_sigma_clipping::py_sliding_sigma_clipping,
        m
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{ArrayD, arr2};

    fn own_data() -> (ArrayD<f64>, ArrayD<f64>, ArrayD<f64>) {
        let data = arr2(&[
            [f64::NAN, 3.0, 1.0, 0.0],
            [5.0, 2.0, f64::NAN, 4.0],
            [1.0, f64::NAN, 5.0, 3.0],
            [1.0, 0.0, 3.0, 4.0],
        ])
        .into_dyn();
        let kernel1 = arr2(&[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]).into_dyn();
        let kernel2 = arr2(&[[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]).into_dyn();
        (data, kernel1, kernel2)
    }

    fn std_population(xs: &[f64]) -> f64 {
        let n = xs.len();
        if n == 0 {
            return f64::NAN;
        }
        let mean = xs.iter().copied().sum::<f64>() / n as f64;
        let var = xs
            .iter()
            .map(|&x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        var.sqrt()
    }

    #[test]
    fn check_mean_zero() {
        // prepare data
        let (mut data, kernel, _) = own_data();
        let pad_mode = PaddingMode::Constant(0.);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        sliding_mean(&mut padded, data.view_mut());

        // compare
        let expected_mean = arr2(&[
            [1.25, 4. / 3., 9. / 7., 5. / 7.],
            [1., 3., 18. / 7., 9. / 7.],
            [8. / 7., 17. / 7., 8. / 3., 16. / 7.],
            [1. / 7., 10. / 7., 12. / 7., 11. / 8.],
        ])
        .into_dyn();
        assert_abs_diff_eq!(data, expected_mean, epsilon = 1e-8);
    }

    #[test]
    fn check_mean_zero_asym() {
        // prepare data
        let (mut data, _, kernel) = own_data();
        let pad_mode = PaddingMode::Constant(0.);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        sliding_mean(&padded, data.view_mut());

        // compare
        let expected_mean = arr2(&[
            [7. / 6., 2., 5. / 3., 5. / 6.],
            [1.5, 2.8, 2.6, 13. / 6.],
            [4. / 7., 2., 3., 2.5],
            [1. / 6., 1., 1., 12. / 7.],
        ])
        .into_dyn();
        assert_abs_diff_eq!(data, expected_mean, epsilon = 1e-8);
    }

    #[test]
    fn check_mean_nan() {
        // prepare data
        let (mut data, kernel, _) = own_data();
        let pad_mode = PaddingMode::Constant(f64::NAN);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        sliding_mean(&mut padded, data.view_mut());

        // compare
        let expected_mean = arr2(&[
            [10. / 3., 8. / 3., 9. / 4., 2.5],
            [2., 3., 18. / 7., 9. / 4.],
            [2., 17. / 7., 8. / 3., 4.],
            [0.5, 2.5, 3., 11. / 3.],
        ])
        .into_dyn();
        assert_abs_diff_eq!(data, expected_mean, epsilon = 1e-8);
    }

    #[test]
    fn check_median_zero() {
        let (mut data, kernel, _) = own_data();
        let pad_mode = PaddingMode::Constant(0.);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        sliding_median(&mut padded, data.view_mut());

        // compare
        let expected_median = arr2(&[
            [0., 0.5, 0., 0.],
            [0.5, 3., 3., 0.],
            [0., 2., 3., 3.],
            [0., 1., 0., 0.],
        ])
        .into_dyn();
        assert_abs_diff_eq!(data, expected_median, epsilon = 1e-8);
    }

    #[test]
    fn check_median_zero_asym() {
        let (mut data, _, kernel) = own_data();
        let pad_mode = PaddingMode::Constant(0.);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        sliding_median(&mut padded, data.view_mut());

        // compare
        let expected_median = arr2(&[
            [0., 2., 1.5, 0.],
            [0.5, 2., 3., 2.],
            [0., 1., 3.5, 3.],
            [0., 0., 0., 0.],
        ])
        .into_dyn();
        assert_abs_diff_eq!(data, expected_median, epsilon = 1e-8);
    }

    #[test]
    fn check_median_nan() {
        let (mut data, kernel, _) = own_data();
        let pad_mode = PaddingMode::Constant(f64::NAN);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        sliding_median(&mut padded, data.view_mut());

        // compare
        let expected_median = arr2(&[
            [3., 2., 2.5, 2.5],
            [2., 3., 3., 2.],
            [1.5, 2., 3., 4.],
            [0.5, 2., 3.5, 3.],
        ])
        .into_dyn();
        assert_abs_diff_eq!(data, expected_median, epsilon = 1e-8);
    }

    #[test]
    fn check_mean_std_zero() {
        let (mut data, kernel, _) = own_data();
        let pad_mode = PaddingMode::Constant(0.);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        let mut mean_buffer = ArrayD::zeros(data.shape());
        sliding_standard_deviation(&mut padded, data.view_mut(), mean_buffer.view_mut());

        // compare
        let expected_mean = arr2(&[
            [1.25, 4. / 3., 9. / 7., 5. / 7.],
            [1., 3., 18. / 7., 9. / 7.],
            [8. / 7., 17. / 7., 8. / 3., 16. / 7.],
            [1. / 7., 10. / 7., 12. / 7., 11. / 8.],
        ])
        .into_dyn();
        assert_abs_diff_eq!(mean_buffer, expected_mean, epsilon = 1e-8);
    }

    #[test]
    fn check_mean_std_zero_asym() {
        let (mut data, _, kernel) = own_data();
        let pad_mode = PaddingMode::Constant(0.);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        let mut mean_buffer = ArrayD::zeros(data.shape());
        sliding_standard_deviation(&mut padded, data.view_mut(), mean_buffer.view_mut());

        // compare
        // compare
        let expected_mean = arr2(&[
            [7. / 6., 2., 5. / 3., 5. / 6.],
            [1.5, 2.8, 2.6, 13. / 6.],
            [4. / 7., 2., 3., 2.5],
            [1. / 6., 1., 1., 12. / 7.],
        ])
        .into_dyn();
        assert_abs_diff_eq!(mean_buffer, expected_mean, epsilon = 1e-8);
    }

    #[test]
    fn check_mean_std_nan() {
        let (mut data, kernel, _) = own_data();
        let pad_mode = PaddingMode::Constant(f64::NAN);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        let mut mean_buffer = ArrayD::zeros(data.shape());
        sliding_standard_deviation(&mut padded, data.view_mut(), mean_buffer.view_mut());

        // compare
        let expected_mean = arr2(&[
            [10. / 3., 8. / 3., 9. / 4., 2.5],
            [2., 3., 18. / 7., 9. / 4.],
            [2., 17. / 7., 8. / 3., 4.],
            [0.5, 2.5, 3., 11. / 3.],
        ])
        .into_dyn();
        assert_abs_diff_eq!(mean_buffer, expected_mean, epsilon = 1e-8);
    }

    #[test]
    fn check_standard_deviation_zero() {
        // prepare data
        let (mut data, kernel, _) = own_data();
        let pad_mode = PaddingMode::Constant(0.);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        let mut mean_buffer = ArrayD::zeros(data.shape());
        sliding_standard_deviation(&mut padded, data.view_mut(), mean_buffer.view_mut());

        // compare
        let std_0_0 = std_population(&[0., 0., 0., 0., 0., 2., 3., 5.]);
        let std_0_1 = std_population(&[0., 0., 0., 1., 2., 5.]);
        let std_0_2 = std_population(&[0., 0., 0., 0., 2., 3., 4.]);
        let std_0_3 = std_population(&[0., 0., 0., 0., 0., 1., 4.]);
        let std_1_0 = std_population(&[0., 0., 0., 1., 2., 3.]);
        let std_1_1 = std_population(&[1., 1., 3., 5., 5.]);
        let std_1_2 = std_population(&[0., 1., 2., 3., 3., 4., 5.]);
        let std_1_3 = std_population(&[0., 0., 0., 0., 1., 3., 5.]);
        let std_2_0 = std_population(&[0., 0., 0., 0., 1., 2., 5.]);
        let std_2_1 = std_population(&[0., 1., 1., 2., 3., 5., 5.]);
        let std_2_2 = std_population(&[0., 2., 3., 3., 4., 4.]);
        let std_2_3 = std_population(&[0., 0., 0., 3., 4., 4., 5.]);
        let std_3_0 = std_population(&[0., 0., 0., 0., 0., 0., 1.]);
        let std_3_1 = std_population(&[0., 0., 0., 1., 1., 3., 5.]);
        let std_3_2 = std_population(&[0., 0., 0., 0., 3., 4., 5.]);
        let std_3_3 = std_population(&[0., 0., 0., 0., 0., 3., 3., 5.]);
        let expected_std = arr2(&[
            [std_0_0, std_0_1, std_0_2, std_0_3],
            [std_1_0, std_1_1, std_1_2, std_1_3],
            [std_2_0, std_2_1, std_2_2, std_2_3],
            [std_3_0, std_3_1, std_3_2, std_3_3],
        ])
        .into_dyn();
        assert_abs_diff_eq!(data, expected_std, epsilon = 1e-8);
    }

    #[test]
    fn check_standard_deviation_zero_asym() {
        // prepare data
        let (mut data, _, kernel) = own_data();
        let pad_mode = PaddingMode::Constant(0.);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        let mut mean_buffer = ArrayD::zeros(data.shape());
        sliding_standard_deviation(&mut padded, data.view_mut(), mean_buffer.view_mut());

        // compare
        let std_0_0 = std_population(&[0., 0., 0., 0., 2., 5.]);
        let std_0_1 = std_population(&[0., 0., 2., 3., 5.]);
        let std_0_2 = std_population(&[0., 0., 1., 2., 3., 4.]);
        let std_0_3 = std_population(&[0., 0., 0., 0., 1., 4.]);
        let std_1_0 = std_population(&[0., 0., 0., 1., 3., 5.]);
        let std_1_1 = std_population(&[1., 1., 2., 5., 5.]);
        let std_1_2 = std_population(&[0., 2., 3., 3., 5.]);
        let std_1_3 = std_population(&[0., 0., 1., 3., 4., 5.]);
        let std_2_0 = std_population(&[0., 0., 0., 0., 1., 1., 2.]);
        let std_2_1 = std_population(&[0., 1., 1., 3., 5.]);
        let std_2_2 = std_population(&[0., 2., 3., 4., 4., 5.]);
        let std_2_3 = std_population(&[0., 0., 3., 3., 4., 5.]);
        let std_3_0 = std_population(&[0., 0., 0., 0., 0., 1.]);
        let std_3_1 = std_population(&[0., 0., 0., 0., 1., 1., 5.]);
        let std_3_2 = std_population(&[0., 0., 0., 0., 3., 3.]);
        let std_3_3 = std_population(&[0., 0., 0., 0., 3., 4., 5.]);
        let expected_std = arr2(&[
            [std_0_0, std_0_1, std_0_2, std_0_3],
            [std_1_0, std_1_1, std_1_2, std_1_3],
            [std_2_0, std_2_1, std_2_2, std_2_3],
            [std_3_0, std_3_1, std_3_2, std_3_3],
        ])
        .into_dyn();
        assert_abs_diff_eq!(data, expected_std, epsilon = 1e-8);
    }

    #[test]
    fn check_standard_deviation_nan() {
        // prepare data
        let (mut data, kernel, _) = own_data();
        let pad_mode = PaddingMode::Constant(f64::NAN);
        let mut padded = SlidingWorkspace::new(data.shape(), kernel, pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        let mut mean_buffer = ArrayD::zeros(data.shape());
        sliding_standard_deviation(&mut padded, data.view_mut(), mean_buffer.view_mut());

        // compare
        let std_0_0 = std_population(&[2., 3., 5.]);
        let std_0_1 = std_population(&[1., 2., 5.]);
        let std_0_2 = std_population(&[0., 2., 3., 4.]);
        let std_0_3 = std_population(&[1., 4.]);
        let std_1_0 = std_population(&[1., 2., 3.]);
        let std_1_1 = std_population(&[1., 1., 3., 5., 5.]);
        let std_1_2 = std_population(&[0., 1., 2., 3., 3., 4., 5.]);
        let std_1_3 = std_population(&[0., 1., 3., 5.]);
        let std_2_0 = std_population(&[0., 1., 2., 5.]);
        let std_2_1 = std_population(&[0., 1., 1., 2., 3., 5., 5.]);
        let std_2_2 = std_population(&[0., 2., 3., 3., 4., 4.]);
        let std_2_3 = std_population(&[3., 4., 4., 5.]);
        let std_3_0 = std_population(&[0., 1.]);
        let std_3_1 = std_population(&[1., 1., 3., 5.]);
        let std_3_2 = std_population(&[0., 3., 4., 5.]);
        let std_3_3 = std_population(&[3., 3., 5.]);
        let expected_std = arr2(&[
            [std_0_0, std_0_1, std_0_2, std_0_3],
            [std_1_0, std_1_1, std_1_2, std_1_3],
            [std_2_0, std_2_1, std_2_2, std_2_3],
            [std_3_0, std_3_1, std_3_2, std_3_3],
        ])
        .into_dyn();
        assert_abs_diff_eq!(data, expected_std, epsilon = 1e-8);
    }
}
