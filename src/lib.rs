//! 'sliding' library containing different sliding window operations.
//! Done to replace the 'sliding' python library that I had created before.

// Local modules
mod convolution;
mod padding;
mod sliding_mean;
mod sliding_standard_deviation;

// Re-exports
pub use convolution::convolution;
pub use padding::{PaddingMode, PaddingWorkspace};
pub use sliding_mean::sliding_mean;
pub use sliding_standard_deviation::sliding_standard_deviation;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{ArrayD, arr2};

    fn own_data() -> (ArrayD<f64>, ArrayD<f64>) {
        let data = arr2(&[
            [f64::NAN, 3.0, 1.0, 0.0],
            [5.0, 2.0, f64::NAN, 4.0],
            [1.0, f64::NAN, 5.0, 3.0],
            [1.0, 0.0, 3.0, 4.0],
        ])
        .into_dyn();
        let kernel = arr2(&[[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]).into_dyn();
        (data, kernel)
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
    fn check_dims() {
        let input_shape = [5, 5];
        let kernel_shape = [3, 3];
        let pad_mode = PaddingMode::Constant(0.);

        let padded = PaddingWorkspace::new(&input_shape, &kernel_shape, pad_mode).unwrap();
        assert_eq!(padded.ndim, 2);
    }

    #[test]
    fn check_mean() {
        // prepare data
        let (data, kernel) = own_data();
        let pad_mode = PaddingMode::Constant(0.0f64);
        let mut padded = PaddingWorkspace::new(data.shape(), kernel.shape(), pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        let computed = sliding_mean(padded, kernel.view());

        // compare
        let expected_mean = arr2(&[
            [1.25, 4. / 3., 9. / 7., 5. / 7.],
            [1., 3., 18. / 7., 9. / 7.],
            [8. / 7., 17. / 7., 8. / 3., 16. / 7.],
            [1. / 7., 10. / 7., 12. / 7., 11. / 8.],
        ])
        .into_dyn();
        assert_abs_diff_eq!(computed.output_buffer, expected_mean, epsilon = 1e-8);
    }

    #[test]
    fn check_mean_std() {
        let (data, kernel) = own_data();
        let pad_mode = PaddingMode::Constant(0.0f64);
        let mut padded = PaddingWorkspace::new(data.shape(), kernel.shape(), pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        let mut mean_buffer = ArrayD::zeros(padded.valid_shape.clone());
        let _ = sliding_standard_deviation(padded, mean_buffer.view_mut(), kernel.view());

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
    fn check_standard_deviation() {
        // prepare data
        let (data, kernel) = own_data();
        let pad_mode = PaddingMode::Constant(0.0f64);
        let mut padded = PaddingWorkspace::new(data.shape(), kernel.shape(), pad_mode).unwrap();
        padded.pad_input(data.view());

        // compute
        let mut mean_buffer = ArrayD::zeros(padded.valid_shape.clone());
        let computed = sliding_standard_deviation(padded, mean_buffer.view_mut(), kernel.view());

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
        assert_abs_diff_eq!(computed.output_buffer, expected_std, epsilon = 1e-8);
    }
}
