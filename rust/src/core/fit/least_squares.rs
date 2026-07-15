// core/fit/least_squares.rs — clean version with correct predicted reduction

use ndarray::{Array1, Array2};

use super::FitResult;
use super::gaussian::{gaussian_jacobian, gaussian_profile};

fn cholesky_decompose(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    Some(l)
}

fn cholesky_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[[i, j]] * y[j];
        }
        y[i] = sum / l[[i, i]];
    }
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[[j, i]] * x[j];
        }
        x[i] = sum / l[[i, i]];
    }
    x
}

fn compute_residuals(waves: &[f64], data: &[f64], error: &[f64], params: &[f64]) -> Array1<f64> {
    let model = gaussian_profile(waves, params);
    let n = data.len();
    let mut r = Array1::zeros(n);
    for i in 0..n {
        r[i] = (data[i] - model[i]) / error[i];
    }
    r
}

fn compute_cost(waves: &[f64], data: &[f64], error: &[f64], params: &[f64]) -> f64 {
    let r = compute_residuals(waves, data, error, params);
    0.5 * r.mapv(|v| v * v).sum()
}

pub fn fit_spectrum(
    waves: &[f64],
    data: &[f64],
    error: &[f64],
    init_guess: &[f64],
    lower_bounds: &[f64],
    upper_bounds: &[f64],
    x_scales: &[f64],
) -> FitResult {
    let n_params = init_guess.len();
    let n_data = data.len();
    let ndof = n_data.saturating_sub(n_params);

    let mut params: Vec<f64> = init_guess.to_vec();
    let mut lambda = 1e-3;
    let mut nu = 2.0;
    let max_iter = 100;
    let gtol = 1e-8;
    let xtol = 1e-8;
    let mut converged = false;

    let mut current_cost = compute_cost(waves, data, error, &params);

    for _iter in 0..max_iter {
        let j = gaussian_jacobian(waves, &params, error);
        let residuals = compute_residuals(waves, data, error, &params);

        let jt = j.t();
        let jtj = jt.dot(&j);
        let jtr = jt.dot(&residuals);

        let mut augmented = jtj.clone();
        for p in 0..n_params {
            augmented[[p, p]] += lambda * jtj[[p, p]].max(1.0);
        }

        let l = match cholesky_decompose(&augmented) {
            Some(l) => l,
            None => {
                lambda *= nu;
                nu *= 2.0;
                continue;
            }
        };

        let neg_jtr = -&jtr;
        let delta_raw = cholesky_solve(&l, &neg_jtr);

        let mut delta = delta_raw.clone();
        for p in 0..n_params {
            delta[p] /= x_scales[p];
        }

        let mut new_params = params.clone();
        for p in 0..n_params {
            new_params[p] += delta[p];
            new_params[p] = new_params[p].clamp(lower_bounds[p], upper_bounds[p]);
        }

        let new_cost = compute_cost(waves, data, error, &new_params);
        let actual_reduction = current_cost - new_cost;

        let j_delta = j.dot(&delta);
        let predicted_reduction = -delta.dot(&jtr) - 0.5 * j_delta.mapv(|x| x * x).sum();

        let rho = if predicted_reduction.abs() > 1e-15 {
            actual_reduction / predicted_reduction
        } else {
            0.0
        };

        if rho > 0.0 {
            let param_change = delta.mapv(|d| d * d).sum().sqrt();
            params = new_params;
            current_cost = new_cost;

            let x_norm = params.iter().map(|&p| p * p).sum::<f64>().sqrt().max(1.0);
            if param_change < xtol * x_norm {
                converged = true;
                break;
            }

            let gain = (2.0 * rho - 1.0).powi(3);
            lambda *= (1.0 - gain).max(1.0 / 3.0);
            nu = 2.0;
        } else {
            lambda *= nu;
            nu *= 2.0;
        }

        if jtr.mapv(|g| g * g).sum().sqrt() < gtol {
            converged = true;
            break;
        }
    }

    let j_final = gaussian_jacobian(waves, &params, error);
    let jtj_final = j_final.t().dot(&j_final);
    let sigma2 = if ndof > 0 {
        2.0 * current_cost / ndof as f64
    } else {
        f64::NAN
    };

    let errors: Vec<f64> = if let Some(l) = cholesky_decompose(&jtj_final) {
        let mut cov_diag = vec![0.0; n_params];
        for i in 0..n_params {
            let mut e_i = Array1::zeros(n_params);
            e_i[i] = 1.0;
            let col = cholesky_solve(&l, &e_i);
            cov_diag[i] = sigma2 * col[i];
        }
        cov_diag
            .iter()
            .map(|&v| if v > 0.0 { v.sqrt() } else { f64::NAN })
            .collect()
    } else {
        vec![f64::NAN; n_params]
    };

    FitResult {
        params,
        errors,
        cost: current_cost,
        converged,
    }
}
