//! 'sliding' library containing different sliding window operations.
//! Done to replace the 'sliding' python library that I had created before.

// Local modules
mod utils;
mod convolution;
mod sliding_mean;
mod sliding_standard_deviation;

// Re-exports
pub use convolution::convolution;
pub use sliding_mean::sliding_mean;