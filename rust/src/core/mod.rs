//! Rust code for sliding operations.
//! Has a convolution, sliding_mean, sliding_median, sliding_sigma_clipping and
//! sliding_standard_deviation function. Also has a padding struct (i.e. 'SlidingWorkspace') but
//! most likely not needed outside the actual create. You never know, so leaving it public.

// Local modules
pub mod convolution;
pub mod padding;
pub mod sliding_mean;
pub mod sliding_median;
pub mod sliding_sigma_clipping;
pub mod sliding_standard_deviation;
mod utils;
