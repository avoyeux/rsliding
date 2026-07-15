// ! To compute the binning fit of an n-dimensional array with possible NaN values.
// ! Used to improve the SNR value when fitting while loosing resolution.

use ndarray::{Array1, ArrayD, ArrayViewD, IxDyn};

use super::fit::least_squares::fit_spectrum;

fn flat_to_multi(mut idx: usize, dims: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; dims.len()];
    for i in (0..dims.len()).rev() {
        coords[i] = idx % dims[i];
        idx /= dims[i];
    }
    coords
}

fn section_start_indices(section_coords: &[usize], bin_factors: &[usize]) -> Vec<usize> {
    section_coords
        .iter()
        .zip(bin_factors.iter())
        .map(|(&c, &b)| c * b)
        .collect()
}

fn extract_spectrum(
    data: &ArrayViewD<f64>,
    non_fit_axes: &[usize],
    bin_factors: &[usize],
    section_coords: &[usize],
    fit_axis: usize,
) -> Array1<f64> {
    let ndim = data.ndim();
    let start = section_start_indices(section_coords, bin_factors);

    let mut sum = Array1::zeros(data.shape()[fit_axis]);
    let n_pixels: usize = bin_factors.iter().product();

    let total_combinations: usize = bin_factors.iter().product();

    for combo in 0..total_combinations {
        let mut remaining = combo;
        let mut view_idx = vec![0; ndim];
        view_idx[fit_axis] = 0;

        for (axis_idx, (&axis, &factor)) in non_fit_axes.iter().zip(bin_factors.iter()).enumerate()
        {
            let offset = remaining % factor;
            remaining /= factor;
            view_idx[axis] = start[axis_idx] + offset;
        }

        for w in 0..data.shape()[fit_axis] {
            view_idx[fit_axis] = w;
            sum[w] += data[view_idx.as_slice()];
        }
    }

    sum / n_pixels as f64
}

pub fn binning_fit(
    data: ArrayViewD<f64>,
    error: ArrayViewD<f64>,
    waves: &[f64],
    fit_axis: usize,
    bins: Vec<usize>,
    init_guess: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    x_scales: &[f64],
) -> (ArrayD<f64>, ArrayD<f64>, ArrayD<f64>) {
    let ndim = data.ndim();
    let non_fit_axes: Vec<usize> = (0..ndim).filter(|&a| a != fit_axis).collect();
    let input_shape = data.shape().to_vec();
    for (&axis, &factor) in non_fit_axes.iter().zip(bins.iter()) {
        if input_shape[axis] % factor != 0 {
            panic!(
                "axis {} size {} not divisible by bin factor {}",
                axis, input_shape[axis], factor
            );
        }
    }

    let n_params = init_guess.len();
    let mut sections_per_axis = Vec::new();
    for (&axis, &factor) in non_fit_axes.iter().zip(bins.iter()) {
        sections_per_axis.push(input_shape[axis] / factor);
    }
    let n_sections: usize = sections_per_axis.iter().product();

    let mut out_shape_params = Vec::new();
    let mut out_shape_errors = Vec::new();
    let mut out_shape_cost = Vec::new();
    let mut non_fit_idx = 0;
    for a in 0..ndim {
        if a == fit_axis {
            out_shape_params.push(n_params);
            out_shape_errors.push(n_params);
            out_shape_cost.push(1);
        } else {
            let n_sec = sections_per_axis[non_fit_idx];
            out_shape_params.push(n_sec);
            out_shape_errors.push(n_sec);
            out_shape_cost.push(n_sec);
            non_fit_idx += 1;
        }
    }

    let mut params_out = ArrayD::from_elem(IxDyn(&out_shape_params), f64::NAN);
    let mut errors_out = ArrayD::from_elem(IxDyn(&out_shape_errors), f64::NAN);
    let mut cost_out = ArrayD::from_elem(IxDyn(&out_shape_cost), f64::NAN);

    for section_idx in 0..n_sections {
        let section_coords = flat_to_multi(section_idx, &sections_per_axis);

        let spectrum = extract_spectrum(&data, &non_fit_axes, &bins, &section_coords, fit_axis);

        let err_spectrum =
            extract_spectrum(&error, &non_fit_axes, &bins, &section_coords, fit_axis);

        let has_nan =
            spectrum.iter().any(|v| v.is_nan()) || err_spectrum.iter().any(|v| v.is_nan());

        if has_nan {
            continue;
        }

        let result = fit_spectrum(
            waves,
            spectrum.as_slice().unwrap(),
            err_spectrum.as_slice().unwrap(),
            init_guess,
            lower_bounds,
            upper_bounds,
            x_scales,
        );

        let mut write_coords = Vec::new();
        let mut non_fit_idx2 = 0;
        for a in 0..ndim {
            if a == fit_axis {
                write_coords.push(0);
            } else {
                write_coords.push(section_coords[non_fit_idx2]);
                non_fit_idx2 += 1;
            }
        }

        for p in 0..n_params {
            let mut pc = write_coords.clone();
            pc[fit_axis] = p;
            params_out[pc.as_slice()] = result.params[p];
            errors_out[pc.as_slice()] = result.errors[p];
        }
        {
            let mut cc = write_coords.clone();
            cc[fit_axis] = 0;
            cost_out[cc.as_slice()] = result.cost;
        }
    }

    (params_out, errors_out, cost_out)
}

#[cfg(test)]
mod tests {
    use super::binning_fit;
    use approx::assert_abs_diff_eq;
    use ndarray::{ArrayD, IxDyn};

    fn double_gaussian(waves: &[f64], params: &[f64]) -> Vec<f64> {
        let (a0, mu0, sig0) = (params[0], params[1], params[2]);
        let (a1, mu1, sig1) = (params[3], params[4], params[5]);
        let bg = params[6];
        waves
            .iter()
            .map(|&w| {
                bg + a0 * (-0.5 * ((w - mu0) / sig0).powi(2)).exp()
                    + a1 * (-0.5 * ((w - mu1) / sig1).powi(2)).exp()
            })
            .collect()
    }

    fn triple_gaussian(waves: &[f64], params: &[f64]) -> Vec<f64> {
        let (a0, mu0, sig0) = (params[0], params[1], params[2]);
        let (a1, mu1, sig1) = (params[3], params[4], params[5]);
        let (a2, mu2, sig2) = (params[6], params[7], params[8]);
        let bg = params[9];
        waves
            .iter()
            .map(|&w| {
                bg + a0 * (-0.5 * ((w - mu0) / sig0).powi(2)).exp()
                    + a1 * (-0.5 * ((w - mu1) / sig1).powi(2)).exp()
                    + a2 * (-0.5 * ((w - mu2) / sig2).powi(2)).exp()
            })
            .collect()
    }

    #[test]
    fn check_binning_fit_ones_noiseless() {
        let true_params = vec![10.0, 500.0, 20.0, 5.0, 600.0, 15.0, 2.0];
        let n_params = true_params.len();

        let nw = 200;
        let waves: Vec<f64> = (0..nw).map(|i| 400.0 + i as f64 * 2.0).collect();
        let spectrum = double_gaussian(&waves, &true_params);

        let nx = 3;
        let ny = 2;
        let mut data_flat = Vec::with_capacity(nx * ny * nw);
        for _ in 0..(nx * ny) {
            data_flat.extend_from_slice(&spectrum);
        }
        let data = ArrayD::from_shape_vec(IxDyn(&[nx, ny, nw]), data_flat).unwrap();
        let error = ArrayD::from_elem(IxDyn(&[nx, ny, nw]), 0.1);

        let init_guess = vec![8.0, 498.0, 22.0, 4.0, 598.0, 17.0, 1.5];
        let lower_bounds = vec![0.0, 400.0, 1.0, 0.0, 400.0, 1.0, 0.0];
        let upper_bounds = vec![100.0, 800.0, 50.0, 100.0, 800.0, 50.0, 10.0];
        let x_scales = vec![1.0; n_params];

        let (params, _errors, _cost) = binning_fit(
            data.view(),
            error.view(),
            &waves,
            2,
            vec![1, 1],
            &init_guess,
            &lower_bounds,
            &upper_bounds,
            &x_scales,
        );

        for i in 0..nx {
            for j in 0..ny {
                for p in 0..n_params {
                    assert_abs_diff_eq!(params[[i, j, p]], true_params[p], epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn check_binning_fit_with_bins_noiseless() {
        let true_params = vec![12.0, 520.0, 18.0, 7.0, 580.0, 12.0, 3.0, 640.0, 22.0, 1.5];
        let n_params = true_params.len();

        let nw = 200;
        let waves: Vec<f64> = (0..nw).map(|i| 400.0 + i as f64 * 2.0).collect();
        let spectrum = triple_gaussian(&waves, &true_params);

        let nx = 4;
        let ny = 4;
        let mut data_flat = Vec::with_capacity(nx * ny * nw);
        for _ in 0..(nx * ny) {
            data_flat.extend_from_slice(&spectrum);
        }
        let data = ArrayD::from_shape_vec(IxDyn(&[nx, ny, nw]), data_flat).unwrap();
        let error = ArrayD::from_elem(IxDyn(&[nx, ny, nw]), 0.1);

        let init_guess = vec![10.0, 518.0, 20.0, 5.0, 578.0, 14.0, 2.0, 638.0, 24.0, 1.0];
        let lower_bounds = vec![0.0, 400.0, 1.0, 0.0, 400.0, 1.0, 0.0, 400.0, 1.0, 0.0];
        let upper_bounds = vec![
            100.0, 800.0, 50.0, 100.0, 800.0, 50.0, 100.0, 800.0, 50.0, 10.0,
        ];
        let x_scales = vec![1.0; n_params];

        let (params, _errors, _cost) = binning_fit(
            data.view(),
            error.view(),
            &waves,
            2,
            vec![2, 2],
            &init_guess,
            &lower_bounds,
            &upper_bounds,
            &x_scales,
        );

        assert_eq!(params.shape(), &[2, 2, n_params]);
        for i in 0..2 {
            for j in 0..2 {
                for p in 0..n_params {
                    assert_abs_diff_eq!(params[[i, j, p]], true_params[p], epsilon = 1e-6);
                }
            }
        }
    }
}
