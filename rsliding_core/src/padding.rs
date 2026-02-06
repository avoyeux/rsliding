//! Paddding (change the filename to padding.rs later)
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn, Slice};

// todo see if strides implementation is needed later
// todo add proper padding mode implementation

pub enum PaddingMode {
    Constant(f64),
    Reflect,
    Replicate,
    Wrap,
}

pub struct PaddingWorkspace {
    pub ndim: usize,                // number of dimensions
    pad: Vec<usize>,                // per-dimension padding
    padding_mode: PaddingMode,      // padding mode
    pub valid_shape: IxDyn,         // initial input shape
    pub padded_buffer: ArrayD<f64>, // reused by padding operations
    filled: bool,
}

impl PaddingWorkspace {
    /// Creates a new PaddingWorkspace.
    /// Check the kernel validity and setups the padded and output buffers.
    pub fn new(
        input_shape: &[usize],
        kernel_shape: &[usize],
        padding_mode: PaddingMode,
    ) -> Result<Self, String> {
        //check
        Self::check_kernel(input_shape, kernel_shape)?;

        let ndim = input_shape.len();
        let pad: Vec<usize> = kernel_shape.iter().map(|&k| k / 2).collect();
        let padded_shape = IxDyn(
            input_shape
                .iter()
                .zip(&pad)
                .map(|(&n, &p)| n + 2 * p)
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let valid_shape = IxDyn(input_shape);

        Ok(Self {
            ndim,
            pad,
            padding_mode,
            valid_shape: valid_shape.clone(),
            padded_buffer: ArrayD::zeros(padded_shape),
            filled: false,
        })
    }

    /// Check if the kernel is valid for the given data.
    /// # Errors
    /// Returns an error if the kernel is not valid.
    fn check_kernel(input_shape: &[usize], kernel_shape: &[usize]) -> Result<(), String> {
        // todo add checks if zero or negative values

        // dims
        if input_shape.len() != kernel_shape.len() {
            return Err("Data and kernel must have the same number of dimensions.".to_string());
        }

        // odd values
        for &dim in kernel_shape {
            if dim % 2 == 0 {
                return Err("Kernel dimensions must be odd.".to_string());
            }
        }
        Ok(())
    }

    /// Pad the input data into the padded buffer.
    /// input data shape must match the valid shape (and not the padded shape).
    pub fn pad_input<'a>(&mut self, input: ArrayViewD<'a, f64>) {
        // reuse existing buffer; fill depending on mode
        match self.padding_mode {
            PaddingMode::Constant(value) => {
                if !self.filled {
                    self.padded_buffer.fill(value)
                }
            }
            _ => panic!("Mode not added yet"),
        }
        self.filled = true;

        let input_shape = input.shape();
        let mut window = self.padded_buffer.view_mut();
        for (axis, p) in self.pad.iter().enumerate() {
            let start = *p as isize;
            let end = start + input_shape[axis] as isize;
            window = window.slice_axis_move(Axis(axis), Slice::from(start..end));
        }
        window.assign(&input);
    }
}
