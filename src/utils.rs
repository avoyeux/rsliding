//! Paddding (change the filename to padding.rs later)
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn, Slice, ArrayViewMutD};
use num_traits::{Zero, Float};

// todo see if strides implementation is needed later

pub enum PaddingMode<T>
where
    T: Default + Float + Zero,
{
    Constant(T),
    Reflect,
    Replicate,
    Wrap,
}

pub struct PaddingWorkspace<T>
where
    T: Default + Float + Zero,
{
    ndim: usize,
    pad: Vec<usize>,
    padded_shape: IxDyn,
    valid_shape: IxDyn,
    padded_buffer: ArrayD<T>,
    output_buffer: ArrayD<T>,      // reused by convolution/sliding ops
}

impl<T> PaddingWorkspace<T>
where
    T: Default + Float + Zero,
{
    pub fn new(
        input_shape: &[usize],
        kernel_shape: &[usize],
        padding_mode: PaddingMode<T>,
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
            padded_shape: padded_shape.clone(),
            valid_shape,
            padded_buffer: ArrayD::zeros(padded_shape),
            output_buffer: ArrayD::zeros(IxDyn(input_shape)),
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

    pub fn pad_input<'a>(&mut self, input: ArrayViewD<'a, T>) {
        // reuse existing buffer; fill depending on mode
        self.padded_buffer.fill(T::zero());
        let mut window = self.padded_buffer.view_mut();
        for (axis, p) in self.pad.iter().enumerate() {
            let start = *p as isize;
            let end = start + input.shape()[axis] as isize;
            window = window.slice_axis_move(Axis(axis), Slice::from(start..end));
        }
        window.assign(&input);
    }

    pub fn padded_view(&self) -> ArrayViewD<'_, T> {
        self.padded_buffer.view()
    }

    pub fn output_view_mut(&mut self) -> ArrayViewMutD<'_, T> {
        self.output_buffer.view_mut()
    }

    pub fn take_output(&mut self) -> ArrayD<T> {
        std::mem::take(&mut self.output_buffer)
    }
}
