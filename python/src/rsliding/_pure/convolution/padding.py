"""
Code to convert the padding choices from cv2.filter2D to numpy.pad.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# TYPE ANNOTATIONs
import numpy.typing as npt
from typing import Literal, cast, TypeAlias
BorderType: TypeAlias = Literal['reflect', 'constant', 'replicate'] | None
KernelType: TypeAlias = int | tuple[int, ...] | npt.NDArray[np.float64]

# API public
__all__ = ['BorderType', 'Padding']



class Padding:
    """
    To add padding to data according to the border type.
    The border type follows the cv2.filter2D border naming convention.
    """

    def __init__(
            self,
            data: npt.NDArray[np.float64],
            kernel: KernelType,
            borders: BorderType = 'reflect',
            pad_value: float = 0.,
            force_contiguous: bool = True,
        ) -> None:
        """
        Adds padding to the given data according to the border type.
        The border type follows the cv2.filter2D border naming convention. The padding is added
        using np.pad.
        To get the padded data, use the 'padded' property.

        Args:
            data (np.ndarray[tuple[int, ...], np.dtype[np.floating]]): the data to pad.
            kernel (tuple[int, ...]): the kernel size used for the convolution.
            borders (BorderType, optional): the border type to use for padding.
                Defaults to 'reflect'.
            pad_value (float, optional): NOT USED. Here only for API consistency with the
                corresponding Rust struct.
            force_contiguous (bool, optional): NOT USED. Here only for API consistency with the
                corresponding Rust struct.
        """

        self._data = data
        if isinstance(kernel, int):
            kernel = (kernel,) * data.ndim
        elif isinstance(kernel, np.ndarray):
            kernel = tuple(kernel.shape)
        self._kernel = kernel
        self._borders = borders

        # RUN
        self._padded_data = self._add_padding()

    @property
    def padded(self) -> npt.NDArray[np.float64]:
        """
        The padded data using np.pad and the borders choice.

        Returns:
            np.ndarray[tuple[int, ...], np.dtype[np.floating]]: the padded data.
        """
        return self._padded_data

    def _get_padding(self) -> dict:
        """
        Gives a dictionary containing the np.pad padding choices equivalent to the border
        information.

        Raises:
            ValueError: if the border type name is not recognised.

        Returns:
            dict: the dictionary containing the padding choices.
        """

        if self._borders is None:
            # ADAPTATIVE borders
            result = {
                'mode': 'constant',
                'constant_values': np.nan,
            }
        elif self._borders == 'reflect':
            result = {
                'mode': 'symmetric',
                'reflect': 'even',
            }
        elif self._borders == 'constant':
            result = {
                'mode': 'constant',
                'constant_values': 0.,
            }
        elif self._borders == 'replicate':
            result = {'mode': 'edge'}
        else:
            raise ValueError(f"Unknown border type: {self._borders}")
        return result

    def _add_padding(self) -> np.ndarray:
        """
        To add padding to the given data according to the border type.

        Returns:
            np.ndarray: the padded data.
        """

        # WIDTH padding
        pad = tuple((k // 2, k // 2) for k in self._kernel)

        # MODE np.pad
        padding_params = self._get_padding()
        padding_mode = padding_params['mode']
        padding_constant_values = padding_params.get('constant_values', 0)
        padding_reflect_type = padding_params.get('reflect', 'even')

        if padding_mode == 'constant':
            padded = np.pad(
                array=self._data,
                pad_width=pad,
                mode=cast(Literal['edge'], padding_mode),
                constant_values=padding_constant_values,
            )
        elif padding_mode == 'symmetric':
            padded = np.pad(
                array=self._data,
                pad_width=pad,
                mode=cast(Literal['edge'], padding_mode),
                reflect_type=cast(Literal['even'], padding_reflect_type),
            )
        else:
            padded = np.pad(
                array=self._data,
                pad_width=pad,
                mode=cast(Literal['edge'], padding_mode),
            )
        return padded
