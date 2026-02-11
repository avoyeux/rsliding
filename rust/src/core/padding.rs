//! Implements the workspace for all sliding operations.
//! Done this way to avoid allocations and redundant calculations.
use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn, Slice};

// todo update/change all docstring

pub enum PaddingMode {
    Constant(f64),
    Reflect,
    Replicate,
}

pub struct SlidingWorkspace {
    pub ndim: usize,                // number of dimensions
    pad: Vec<usize>,                // per-dimension padding
    padding_mode: PaddingMode,      // padding mode
    pub padded_buffer: ArrayD<f64>, // reused by padding operations
    pub out_shape: Vec<usize>,      // shape of the output
    kernel_shape: Vec<usize>,       // shape of the kernel
    pub idx: Vec<usize>,
    kernel: ArrayD<f64>,
    pub kernel_offsets: Vec<isize>,
    pub kernel_weights: Vec<f64>,
    filled: bool,
}

impl SlidingWorkspace {
    /// Creates a new PaddingWorkspace.
    /// Check the kernel validity and setups the padded and output buffers.
    pub fn new(
        input_shape: &[usize],
        kernel: ArrayD<f64>,
        padding_mode: PaddingMode,
    ) -> Result<Self, String> {
        // check kernel validity
        Self::check_kernel(input_shape, kernel.shape())?;

        let ndim = input_shape.len();
        let pad: Vec<usize> = kernel.shape().iter().map(|&k| k / 2).collect();
        let padded_shape = IxDyn(
            input_shape
                .iter()
                .zip(&pad)
                .map(|(&n, &p)| n + 2 * p)
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let out_shape = input_shape.to_vec();
        let kernel_shape = kernel.shape().to_vec();
        let idx = vec![0usize; ndim];

        // kernel offsets and weights (skip zeros)
        let kernel_slice = kernel
            .as_slice_memory_order()
            .expect("Kernel must be contiguous");
        let kernel_offsets = Vec::with_capacity(kernel_slice.len());
        let kernel_weights = Vec::with_capacity(kernel_slice.len());

        let mut instance = SlidingWorkspace {
            ndim,
            pad,
            padding_mode,
            padded_buffer: ArrayD::zeros(padded_shape),
            out_shape,
            kernel_shape,
            idx,
            kernel,
            kernel_offsets,
            kernel_weights,
            filled: false,
        };
        instance.create_offsets();
        Ok(instance)
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
                    self.padded_buffer.fill(value);
                    self.filled = true; // only fill once since constant padding
                }
            }
            _ => (), // other padding choices are done when data is already in the buffer.
        }

        let mut window = self.padded_buffer.view_mut();
        for (axis, p) in self.pad.iter().enumerate() {
            let start = *p as isize;
            let end = start + self.out_shape[axis] as isize;
            window = window.slice_axis_move(Axis(axis), Slice::from(start..end));
        }
        window.assign(&input);

        // padding
        match self.padding_mode {
            PaddingMode::Constant(_) => (), // already done
            PaddingMode::Reflect => self.fill_reflect(),
            PaddingMode::Replicate => self.fill_replicate(),
        }
    }

    fn fill_reflect(&mut self) {
        for axis_idx in 0..self.ndim {
            let pad = self.pad[axis_idx];
            if pad == 0 {
                continue;
            }

            let core_len = self.out_shape[axis_idx];
            let axis = Axis(axis_idx);
            let padded = self.padded_buffer.view_mut();
            let (mut left_pad, tail) = padded.split_at(axis, pad);
            let (core, mut right_pad) = tail.split_at(axis, core_len);

            for offset in 0..pad {
                let src_idx = Self::even_mirror_index(offset, core_len);
                let dst_idx = pad - 1 - offset;
                let src = core.index_axis(axis, src_idx);
                let mut dst = left_pad.index_axis_mut(axis, dst_idx);
                dst.assign(&src);
            }

            for offset in 0..pad {
                let src_idx = core_len - 1 - Self::even_mirror_index(offset, core_len);
                let dst_idx = offset;
                let src = core.index_axis(axis, src_idx);
                let mut dst = right_pad.index_axis_mut(axis, dst_idx);
                dst.assign(&src);
            }
        }
    }

    #[inline]
    fn even_mirror_index(distance: usize, len: usize) -> usize {
        if len <= 1 {
            return 0;
        }
        let period = 2 * len - 2;
        let mut d = distance % period;
        if d >= len {
            d = period - d;
        }
        d
    }

    fn fill_replicate(&mut self) {
        for axis_idx in 0..self.ndim {
            let pad = self.pad[axis_idx];
            if pad == 0 {
                continue;
            }

            let core_len = self.out_shape[axis_idx];
            let axis = Axis(axis_idx);
            let padded = self.padded_buffer.view_mut();
            let (mut left_pad, tail) = padded.split_at(axis, pad);
            let (core, mut right_pad) = tail.split_at(axis, core_len);

            let left_edge = core.index_axis(axis, 0);
            let right_edge = core.index_axis(axis, core_len - 1);

            for i in 0..pad {
                let mut dst = left_pad.index_axis_mut(axis, i);
                dst.assign(&left_edge);
            }

            for i in 0..pad {
                let mut dst = right_pad.index_axis_mut(axis, i);
                dst.assign(&right_edge);
            }
        }
    }

    fn create_offsets(&mut self) {
        let padded_strides = self.padded_buffer.strides();
        let kernel_strides = self.kernel.strides();
        let kernel_slice = self
            .kernel
            .as_slice_memory_order()
            .expect("Kernel must be contiguous");

        // manual multi-indexing
        let mut kernel_base = 0isize;
        loop {
            let weight = kernel_slice[kernel_base as usize];

            if weight != 0. {
                // padded offset
                let mut offset = 0isize;
                for d in 0..self.ndim {
                    offset += (self.idx[d] as isize) * padded_strides[d];
                }
                self.kernel_offsets.push(offset);
                self.kernel_weights.push(weight);
            }

            // increment index
            let mut d = self.ndim;
            loop {
                d -= 1;

                self.idx[d] += 1;
                kernel_base += kernel_strides[d];

                if self.idx[d] < self.kernel_shape[d] {
                    break;
                }

                self.idx[d] = 0;
                kernel_base -= (self.kernel_shape[d] as isize) * kernel_strides[d];
                if d == 0 {
                    break;
                }
            }

            if self.idx.iter().all(|&x| x == 0) {
                break;
            }
        }
    }
}
