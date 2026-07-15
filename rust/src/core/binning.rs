//! Contains the workspace struct used by all binning operations.
//! This struct does the binning.

pub struct BinningWorkspace {
    pub bin_factors: Vec<usize>,
    pub non_fit_axes: Vec<usize>,
    pub fit_axis: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub n_sections: usize,
    pub sections_per_axis: Vec<usize>,
}

impl BinningWorkspace {
    pub fn new(
        input_shape: &[usize],
        fit_axis: usize,
        bin_factors: Vec<usize>,
    ) -> Result<Self, String> {
        let ndim = input_shape.len();

        if fit_axis >= ndim {
            return Err(format!(
                "fit_axis {} out of bounds for {}D data",
                fit_axis, ndim
            ));
        }
        if bin_factors.len() != ndim - 1 {
            return Err(format!(
                "bins length {} must equal ndim-1 ({})",
                bin_factors.len(),
                ndim - 1
            ));
        }

        let non_fit_axes: Vec<usize> = (0..ndim).filter(|&a| a != fit_axis).collect();

        for (&axis, &factor) in non_fit_axes.iter().zip(bin_factors.iter()) {
            if input_shape[axis] % factor != 0 {
                return Err(format!(
                    "axis {} size {} not divisible by bin factor {}",
                    axis, input_shape[axis], factor
                ));
            }
        }

        let mut output_shape = input_shape.to_vec();
        let mut sections_per_axis = Vec::with_capacity(non_fit_axes.len());
        for (&axis, &factor) in non_fit_axes.iter().zip(bin_factors.iter()) {
            output_shape[axis] = input_shape[axis] / factor;
            sections_per_axis.push(output_shape[axis]);
        }
        output_shape[fit_axis] = 1;

        let n_sections: usize = sections_per_axis.iter().product();

        Ok(BinningWorkspace {
            bin_factors,
            non_fit_axes,
            fit_axis,
            input_shape: input_shape.to_vec(),
            output_shape,
            n_sections,
            sections_per_axis,
        })
    }
}
