""" test the filterbank_bar_graph module"""

from contextlib import ExitStack as does_not_raise

import numpy as np
import pytest

import music_visualization_system.filterbank_bar_graph as fbg

TEST_DATA = [
    (np.array([1, 2, 3, 4]), 4, np.array([1, 2, 3, 4]), does_not_raise()),
    (np.array([1, 1, 2, 2, 3, 3, 4, 4]), 4, np.array([1, 2, 3, 4]), does_not_raise()),
    (np.array([1]), 3, np.array([1, 1, 1]), does_not_raise()),
    (np.array([1]), 3, np.array([1, 1, 1]), does_not_raise()),
]


@pytest.mark.parametrize("filterbank,width,expected,raises", TEST_DATA)
def test_reshape_filterbank(filterbank, width, expected, raises):
    with raises:
        reshaped_filterbank = fbg._reshape_filterbank(filterbank, width)

        assert reshaped_filterbank.shape[0] == width
        assert np.array_equal(reshaped_filterbank, expected)


TEST_DATA = [
    (1, 1, 1, np.full((1), 1).astype(np.uint32), (1, 1, 3), does_not_raise(),),
    (1, 2, 1, np.full((1), 0).astype(np.uint32), (2, 1, 3), does_not_raise(),),
    (
        100,
        200,
        1,
        np.full((100), 0).astype(np.uint32),
        (200, 100, 3),
        does_not_raise(),
    ),
]


@pytest.mark.parametrize(
    "width,height,max_y_axis,filterbank,expected,raises", TEST_DATA
)
def test_filterbank_graph_size(height, width, max_y_axis, filterbank, expected, raises):
    with raises:
        bar_graph = fbg.one_channel(width, height, max_y_axis, filterbank)

        assert bar_graph.shape == expected
