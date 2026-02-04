//! 'sliding' library containing different sliding window operations.
//! Done to replace the 'sliding' python library that I had created before.

// Local modules
mod padding;
mod convolution;
mod sliding_mean;
// mod sliding_standard_deviation;

// Re-exports
pub use padding::{PaddingMode, PaddingWorkspace};
pub use convolution::convolution;
pub use sliding_mean::sliding_mean;


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;
    use approx::assert_abs_diff_eq;

    #[test]
    fn check_dims() {

        let input_shape = [5, 5];
        let kernel_shape = [3, 3];
        let pad_mode = PaddingMode::Constant(0.);

        let padded = PaddingWorkspace::new(
            &input_shape,
            &kernel_shape,
            pad_mode,
        ).unwrap();

        assert_eq!(padded.ndim, 2);
    }

    #[test]
    fn check_mean() {
        let data = arr2(&[
            [f64::NAN, 3.0, 1.0, 0.0],
            [5.0, 2.0, f64::NAN, 4.0],
            [1.0, f64::NAN, 5.0, 3.0],
            [1.0, 0.0, 3.0, 4.0],
        ]).into_dyn();
        let kernel = arr2(&[
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]).into_dyn();
        let expected_mean = arr2(&[
            [1.25, 4. / 3., 9. / 7., 5. / 7.],
            [1., 3., 18. / 7., 9. / 7.],
            [8. / 7., 17. / 7., 8. / 3., 16. / 7.],
            [1. / 7., 10. / 7., 12. / 7., 11. / 8.],
        ]).into_dyn();

        let pad_mode = PaddingMode::Constant(0.0f64);
        let mut padded = PaddingWorkspace::new(
            data.shape(),
            kernel.shape(),
            pad_mode,
        ).unwrap();

        // load input data into the padded workspace
        padded.pad_input(data.view());

        // run sliding mean and compare the output buffer to expected mean
        let computed = sliding_mean(padded, kernel.view());
        assert_abs_diff_eq!(computed.output_buffer, expected_mean, epsilon = 1e-8);
    }
}
