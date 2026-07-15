// core/fit/gaussian.rs

use ndarray::{Array1, Array2};

/// Evaluate multi-Gaussian profile: A_i * exp(-0.5 * ((λ - μ_i) / σ_i)²) + continuum
/// params layout: [A_0, μ_0, σ_0, A_1, μ_1, σ_1, ..., bg]
pub fn gaussian_profile(waves: &[f64], params: &[f64]) -> Array1<f64> {
    let nb_lines = (params.len() - 1) / 3;
    let mut profile = Array1::from_elem(waves.len(), params[params.len() - 1]);
    if nb_lines > 0 {
        for i in 0..nb_lines {
            let (a, mu, sig) = (params[3 * i], params[3 * i + 1], params[3 * i + 2]);
            for (j, &w) in waves.iter().enumerate() {
                let dw = (w - mu) / sig;
                profile[j] += a * (-0.5 * dw * dw).exp();
            }
        }
    }
    profile
}

/// Analytic Jacobian: d(residual)/d(param) for weighted residuals
/// residual = (data - model) / error
pub fn gaussian_jacobian(waves: &[f64], params: &[f64], error: &[f64]) -> Array2<f64> {
    let nb_lines = (params.len() - 1) / 3;
    let nw = waves.len();
    let n_params = params.len();
    let mut j = Array2::zeros((nw, n_params));
    let inv_err: Vec<f64> = error.iter().map(|&e| -1.0 / e).collect();

    for jj in 0..nw {
        j[[jj, n_params - 1]] = inv_err[jj]; // d(res)/d(bg)
    }

    for i in 0..nb_lines {
        let (a, mu, sig) = (params[3 * i], params[3 * i + 1], params[3 * i + 2]);
        for jj in 0..nw {
            let dw = waves[jj] - mu;
            let exp_term = (-0.5 * (dw / sig).powi(2)).exp();

            j[[jj, 3 * i]] = inv_err[jj] * exp_term;
            j[[jj, 3 * i + 1]] = inv_err[jj] * a * dw / (sig * sig) * exp_term;
            j[[jj, 3 * i + 2]] = inv_err[jj] * a * (dw * dw) / (sig.powi(3)) * exp_term;
        }
    }
    j
}
