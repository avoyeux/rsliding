// core/fit/mod.rs

pub mod gaussian;
pub mod least_squares;

pub struct FitResult {
    pub params: Vec<f64>,
    pub errors: Vec<f64>,
    pub cost: f64,
    pub converged: bool,
}
