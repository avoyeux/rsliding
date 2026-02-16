//! Contains the workspace struct used by all sliding operations.
//! This struct does the padding and keeps the padded buffer and offsets needed for all sliding
//! operations.

use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn, Slice};

/// The different padding modes implemented in the SlidingWorkspace struct.
/// Constant puts constant values as the padding.
/// Reflect, reflects the values at the border.
/// Replicate, replicates the border values.
pub enum PaddingMode {
    Constant(f64),
    Reflect,
    Replicate,
}

/// Workspace used in all sliding operations.
/// The workspace adds the padding and contains the different buffers needed for the sliding
/// operations.
pub struct SlidingWorkspace {
    pub padded: ArrayD<f64>,        // reused by padding operations
    pub kernel_offsets: Vec<isize>, // the offsets of the kernel elements in the padded buffer
    pub kernel_weights: Vec<f64>,   // the weights of the kernel
    ndim: usize,                    // number of dimensions
    pad: Vec<usize>,                // per-dimension padding
    padding_mode: PaddingMode,      // padding mode
    out_shape: Vec<usize>,          // shape of the output
    kernel: ArrayD<f64>,            // the actual kernel
    filled: bool, // used when the padding is set to constant (so the padding is only done once)
}

impl SlidingWorkspace {
    /// Creates a new SlidingWorkspace.
    /// Computes the offsets needed during the multithreaded sliding operations.
    /// Also creates the padded data (use the pad_input method to populate the padded buffer and
    /// compute the padding (if needed).
    /// Keep in mind that the kernel needs to be contiguous.
    pub fn new(
        input_shape: &[usize],
        kernel: ArrayD<f64>,
        padding_mode: PaddingMode,
    ) -> Result<Self, String> {
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

        // kernel offsets and weights (skip zeros)
        let kernel_offsets = Vec::with_capacity(kernel.len());
        let kernel_weights = Vec::with_capacity(kernel.len());

        let mut instance = SlidingWorkspace {
            ndim,
            pad,
            padding_mode,
            padded: ArrayD::zeros(padded_shape),
            out_shape,
            kernel,
            kernel_offsets,
            kernel_weights,
            filled: false,
        };
        instance.create_offsets();
        Ok(instance)
    }

    /// Pad the input data into the padded buffer.
    /// No shape checks are done so make sure that the input data shape matches the valid shape
    /// (and not the padded shape), i.e. must match 'input_shape' given in new().
    pub fn pad_input<'a>(&mut self, input: ArrayViewD<'a, f64>) {
        // fill once if padding mode is constant
        match self.padding_mode {
            PaddingMode::Constant(value) => {
                if !self.filled {
                    self.padded.fill(value);
                    self.filled = true; // only fill once since constant padding
                }
            }
            _ => (), // other padding choices are done when data is already in the buffer.
        }

        // populate input inside the padded buffer
        let mut window = self.padded.view_mut();
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

    /// Computes the offset needed to get the corresponding element in the padded buffer from a
    /// linear index in the output.
    /// Used in the multithreaded sliding operations.
    /// The inline macro is most likely useless given that it is only used inside this crate, but
    /// kept for the reader.
    #[inline]
    pub fn base_offset_from_linear(&self, mut linear: usize, padded_strides: &[isize]) -> isize {
        let out_shape = &self.out_shape;
        let mut base = 0isize;

        for d in (0..out_shape.len()).rev() {
            let dim = out_shape[d];
            let idx = linear % dim;
            linear /= dim;
            base += (idx as isize) * padded_strides[d];
        }
        base
    }

    /// Does the reflect mode padding.
    fn fill_reflect(&mut self) {
        for axis_idx in 0..self.ndim {
            let pad = self.pad[axis_idx];
            if pad == 0 {
                continue;
            }

            let core_len = self.out_shape[axis_idx];
            let axis = Axis(axis_idx);
            let padded = self.padded.view_mut();
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

    /// Mirrors the index for even-length reflection padding.
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

    /// Does the replicate mode padding.
    fn fill_replicate(&mut self) {
        for axis_idx in 0..self.ndim {
            let pad = self.pad[axis_idx];
            if pad == 0 {
                continue;
            }

            let core_len = self.out_shape[axis_idx];
            let axis = Axis(axis_idx);
            let padded = self.padded.view_mut();
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

    /// Creates the kernel offsets and weights for the sliding operation.
    fn create_offsets(&mut self) {
        let mut idx = vec![0usize; self.ndim];
        let kernel_shape = self.kernel.shape().to_vec();
        let padded_strides = self.padded.strides();
        let kernel_strides = self.kernel.strides();
        let kernel_slice = self.kernel.as_slice_memory_order().unwrap();

        // manual multi-indexing
        let mut kernel_base = 0isize;
        loop {
            let weight = kernel_slice[kernel_base as usize];

            if weight != 0. {
                // padded offset
                let mut offset = 0isize;
                for d in 0..self.ndim {
                    offset += (idx[d] as isize) * padded_strides[d];
                }
                self.kernel_offsets.push(offset);
                self.kernel_weights.push(weight);
            }

            // increment index
            let mut d = self.ndim;
            loop {
                d -= 1;

                idx[d] += 1;
                kernel_base += kernel_strides[d];

                if idx[d] < kernel_shape[d] {
                    break;
                }

                idx[d] = 0;
                kernel_base -= (kernel_shape[d] as isize) * kernel_strides[d];
                if d == 0 {
                    break;
                }
            }

            if idx.iter().all(|&x| x == 0) {
                break;
            }
        }
    }
}
