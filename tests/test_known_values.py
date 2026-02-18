"""
Just to test if the wheel was properly built for the current platform.
! IMPORTANT: need to install pytest prior to running this test.
"""
from __future__ import annotations


# IMPORTs
import pytest

# IMPORTs alias
import numpy as np

# IMPORTs local
from rsliding import SlidingMean, SlidingMedian, SlidingStandardDeviation



class TestKnownArrays:
    """
    To test the sliding computations on an array with known expected results.
    Also tests the None borders option.
    """

    @pytest.fixture(scope='class')
    def data_2d(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Provides a 2D array for which the expected results are known.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: the data to use for the tests and two
                possible kernels.
        """

        # KERNEL pre-computed
        kernel1 = np.ones((3, 3), dtype=np.float32)
        kernel2 = kernel1.copy()
        kernel1[1, 1] = 0.
        kernel2[0, 1] = 0.
        kernel2[1, 2] = 0.

        # DATA pre-computed
        computed_data = np.array([
            [np.nan, 3, 1, 0],
            [5, 2, np.nan, 4],
            [1, np.nan, 5, 3],
            [1, 0, 3, 4],
        ], dtype=np.float32)
        return computed_data, kernel1, kernel2

    def _known_mean_constant(self) -> np.ndarray:
        """
        Gives the expected sliding mean results when the borders are set to 'constant'.

        Returns:
            np.ndarray: the expected sliding mean results.
        """

        mean = np.array([
            [1.25, 4/3, 9/7, 5/7],
            [1., 3., 18/7, 9/7],
            [8/7, 17/7, 8/3, 16/7],
            [1/7, 10/7, 12/7, 11/8],
        ], dtype=np.float32)
        return mean

    def _known_mean_constant_asym(self) -> np.ndarray:
        """
        Gives the expected sliding mean results when the borders are set to 'constant'.

        Returns:
            np.ndarray: the expected sliding mean results.
        """

        mean = np.array([
            [7/6, 2, 5/3, 5/6],
            [1.5, 2.8, 2.6, 13/6],
            [4/7, 2, 3, 2.5],
            [1/6, 1, 1, 12/7],
        ], dtype=np.float32)
        return mean

    def _know_mean_none(self) -> np.ndarray:
        """
        Gives the expected sliding mean results when the borders are set to None.

        Returns:
            np.ndarray: the expected sliding mean results.
        """

        mean = np.array([
            [10/3, 8/3, 9/4, 5/2],
            [2., 3., 18/7, 9/4],
            [2., 17/7, 8/3, 4.],
            [.5, 2.5, 3., 11/3],
        ], dtype=np.float32)
        return mean

    def _known_median_constant(self) -> np.ndarray:
        """
        Gives the expected sliding median results when the borders are set to 'constant'.

        Returns:
            np.ndarray: the expected sliding median results.
        """

        median = np.array([
                [0., .5, 0., 0.],
                [.5, 3., 3., 0.],
                [0., 2., 3., 3.],
                [0., 1., 0., 0.],
            ], dtype=np.float32)
        return median

    def _known_median_constant_asym(self) -> np.ndarray:
        """
        Gives the expected sliding median results when the borders are set to 'constant'.

        Returns:
            np.ndarray: the expected sliding median results.
        """

        median = np.array([
                [0, 2, 1.5, 0],
                [.5, 2, 3, 2],
                [0, 1, 3.5, 3],
                [0, 0, 0, 0],
            ], dtype=np.float32)
        return median

    def _known_median_none(self) -> np.ndarray:
        """
        Gives the expected sliding median results when the borders are set to None.

        Returns:
            np.ndarray: the expected sliding median results.
        """

        median = np.array([
            [3., 2., 2.5, 2.5],
            [2., 3., 3., 2.],
            [1.5, 2., 3., 4.],
            [.5, 2., 3.5, 3.],
        ], dtype=np.float32)
        return median

    def _known_std_constant(self) -> np.ndarray:
        """
        Gives the expected sliding standard deviation results when the borders are set to
        'constant'.

        Returns:
            np.ndarray: the expected sliding standard deviation results.
        """

        std_0_0 = np.std([0, 0, 0, 0, 0, 2, 3, 5], dtype=np.float32)
        std_0_1 = np.std([0, 0, 0, 1, 2, 5], dtype=np.float32)
        std_0_2 = np.std([0, 0, 0, 0, 2, 3, 4], dtype=np.float32)
        std_0_3 = np.std([0, 0, 0, 0, 0, 1, 4], dtype=np.float32)
        std_1_0 = np.std([0, 0, 0, 1, 2, 3], dtype=np.float32)
        std_1_1 = np.std([1, 1, 3, 5, 5], dtype=np.float32)
        std_1_2 = np.std([0, 1, 2, 3, 3, 4, 5], dtype=np.float32)
        std_1_3 = np.std([0, 0, 0, 0, 1, 3, 5], dtype=np.float32)
        std_2_0 = np.std([0, 0, 0, 0, 1, 2, 5], dtype=np.float32)
        std_2_1 = np.std([0, 1, 1, 2, 3, 5, 5], dtype=np.float32)
        std_2_2 = np.std([0, 2, 3, 3, 4, 4], dtype=np.float32)
        std_2_3 = np.std([0, 0, 0, 3, 4, 4, 5], dtype=np.float32)
        std_3_0 = np.std([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        std_3_1 = np.std([0, 0, 0, 1, 1, 3, 5], dtype=np.float32)
        std_3_2 = np.std([0, 0, 0, 0, 3, 4, 5], dtype=np.float32)
        std_3_3 = np.std([0, 0, 0, 0, 0, 3, 3, 5], dtype=np.float32)

        std = np.array([
            [std_0_0, std_0_1, std_0_2, std_0_3],
            [std_1_0, std_1_1, std_1_2, std_1_3],
            [std_2_0, std_2_1, std_2_2, std_2_3],
            [std_3_0, std_3_1, std_3_2, std_3_3],
        ], dtype=np.float32)
        return std

    def _known_std_constant_asym(self) -> np.ndarray:
        """
        Gives the expected sliding standard deviation results when the borders are set to
        'constant'.

        Returns:
            np.ndarray: the expected sliding standard deviation results.
        """

        std_0_0 = np.std([0, 0, 0, 0, 2, 5], dtype=np.float32)
        std_0_1 = np.std([0, 0, 2, 3, 5], dtype=np.float32)
        std_0_2 = np.std([0, 0, 1, 2, 3, 4], dtype=np.float32)
        std_0_3 = np.std([0, 0, 0, 0, 1, 4], dtype=np.float32)
        std_1_0 = np.std([0, 0, 0, 1, 3, 5], dtype=np.float32)
        std_1_1 = np.std([1, 1, 2, 5, 5], dtype=np.float32)
        std_1_2 = np.std([0, 2, 3, 3, 5], dtype=np.float32)
        std_1_3 = np.std([0, 0, 1, 3, 4, 5], dtype=np.float32)
        std_2_0 = np.std([0, 0, 0, 0, 1, 1, 2], dtype=np.float32)
        std_2_1 = np.std([0, 1, 1, 3, 5], dtype=np.float32)
        std_2_2 = np.std([0, 2, 3, 4, 4, 5], dtype=np.float32)
        std_2_3 = np.std([0, 0, 3, 3, 4, 5], dtype=np.float32)
        std_3_0 = np.std([0, 0, 0, 0, 0, 1], dtype=np.float32)
        std_3_1 = np.std([0, 0, 0, 0, 1, 1, 5], dtype=np.float32)
        std_3_2 = np.std([0, 0, 0, 0, 3, 3], dtype=np.float32)
        std_3_3 = np.std([0, 0, 0, 0, 3, 4, 5], dtype=np.float32)

        std = np.array([
            [std_0_0, std_0_1, std_0_2, std_0_3],
            [std_1_0, std_1_1, std_1_2, std_1_3],
            [std_2_0, std_2_1, std_2_2, std_2_3],
            [std_3_0, std_3_1, std_3_2, std_3_3],
        ], dtype=np.float32)
        return std

    def _known_std_none(self) -> np.ndarray:
        """
        Gives the expected sliding standard deviation results when the borders are set to None.

        Returns:
            np.ndarray: the expected sliding standard deviation results.
        """

        std_0_0 = np.std([2, 3, 5], dtype=np.float32)
        std_0_1 = np.std([1, 2, 5], dtype=np.float32)
        std_0_2 = np.std([0, 2, 3, 4], dtype=np.float32)
        std_0_3 = np.std([1, 4], dtype=np.float32)
        std_1_0 = np.std([1, 2, 3], dtype=np.float32)
        std_1_1 = np.std([1, 1, 3, 5, 5], dtype=np.float32)
        std_1_2 = np.std([0, 1, 2, 3, 3, 4, 5], dtype=np.float32)
        std_1_3 = np.std([0, 1, 3, 5], dtype=np.float32)
        std_2_0 = np.std([0, 1, 2, 5], dtype=np.float32)
        std_2_1 = np.std([0, 1, 1, 2, 3, 5, 5], dtype=np.float32)
        std_2_2 = np.std([0, 2, 3, 3, 4, 4], dtype=np.float32)
        std_2_3 = np.std([3, 4, 4, 5], dtype=np.float32)
        std_3_0 = np.std([0, 1], dtype=np.float32)
        std_3_1 = np.std([1, 1, 3, 5], dtype=np.float32)
        std_3_2 = np.std([0, 3, 4, 5], dtype=np.float32)
        std_3_3 = np.std([3, 3, 5], dtype=np.float32)

        std = np.array([
            [std_0_0, std_0_1, std_0_2, std_0_3],
            [std_1_0, std_1_1, std_1_2, std_1_3],
            [std_2_0, std_2_1, std_2_2, std_2_3],
            [std_3_0, std_3_1, std_3_2, std_3_3],
        ], dtype=np.float32)
        return std

    def test_constant_mean_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Test the SlidingMean function when borders are set to 'constant'.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, kernel, _ = data_2d
        expected = self._known_mean_constant()

        result = SlidingMean(
            data=data,
            kernel=kernel,
            borders='constant',
            threads=2,
        ).mean

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_constant_mean_sliding_asym(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Test the SlidingMean function when borders are set to 'constant'.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, _, kernel = data_2d
        expected = self._known_mean_constant_asym()

        result = SlidingMean(
            data=data,
            kernel=kernel,
            borders='constant',
            threads=2,
        ).mean

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_none_mean_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Test the SlidingMean function when borders are set to None.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, kernel, _ = data_2d
        expected = self._know_mean_none()

        result = SlidingMean(
            data=data,
            kernel=kernel,
            borders=None,
            threads=2,
        ).mean

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_constant_mean_std(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the mean of the SlidingStandardDeviation class when borders are set to 'constant'.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, kernel, _ = data_2d
        expected = self._known_mean_constant()

        result = SlidingStandardDeviation(
            data=data,
            kernel=kernel,
            borders='constant',
            threads=2,
        ).mean

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_constant_mean_std_asym(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the mean of the SlidingStandardDeviation class when borders are set to 'constant'.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, _, kernel = data_2d
        expected = self._known_mean_constant_asym()

        result = SlidingStandardDeviation(
            data=data,
            kernel=kernel,
            borders='constant',
            threads=2,
        ).mean

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_none_mean_std(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the mean of the SlidingStandardDeviation class when borders are set to None.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, kernel, _ = data_2d
        expected = self._know_mean_none()

        result = SlidingStandardDeviation(
            data=data,
            kernel=kernel,
            borders=None,
            threads=2,
        ).mean

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_constant_median_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the SlidingStandardDeviation class when borders are set to 'constant'.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, kernel, _ = data_2d
        expected = self._known_median_constant()

        result = SlidingMedian(
            data=data,
            kernel=kernel,
            borders='constant',
            threads=2,
        ).median

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_constant_median_sliding_asym(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the SlidingStandardDeviation class when borders are set to 'constant'.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, _, kernel = data_2d
        expected = self._known_median_constant_asym()

        result = SlidingMedian(
            data=data,
            kernel=kernel,
            borders='constant',
            threads=2,
        ).median

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_none_median_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the sliding median results when borders are set to None.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, kernel, _ = data_2d
        expected = self._known_median_none()

        result = SlidingMedian(
            data=data,
            kernel=kernel,
            borders=None,
            threads=2,
        ).median

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_constant_std_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the sliding standard deviation results when borders are set to 'constant'.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, kernel, _ = data_2d
        expected = self._known_std_constant()

        result = SlidingStandardDeviation(
            data=data,
            kernel=kernel,
            borders='constant',
            threads=2,
        ).standard_deviation

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_constant_std_sliding_asym(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the sliding standard deviation results when borders are set to 'constant' and the
        kernel weights are asymmetric.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, _, kernel = data_2d
        expected = self._known_std_constant_asym()

        result = SlidingStandardDeviation(
            data=data,
            kernel=kernel,
            borders='constant',
            threads=2,
        ).standard_deviation

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)

    def test_none_std_sliding(
            self,
            data_2d: tuple[np.ndarray, np.ndarray, np.ndarray],
        ) -> None:
        """
        Tests the SlidingStandardDeviation class when borders are set to None.

        Args:
            data_2d (tuple[np.ndarray, np.ndarray, np.ndarray]): the data and kernels to test for.
        """

        data, kernel, _ = data_2d
        expected = self._known_std_none()

        result = SlidingStandardDeviation(
            data=data,
            kernel=kernel,
            borders=None,
            threads=2,
        ).standard_deviation

        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-5)
