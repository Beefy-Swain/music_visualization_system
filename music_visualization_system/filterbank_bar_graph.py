import numpy as np


def _reshape_filterbank(filterbank: np.ndarray, width: int) -> np.ndarray:
    common_multiple_filterbank = np.repeat(filterbank, repeats=width).reshape(
        (width, len(filterbank))
    )
    reshaped_filterbank = np.mean(common_multiple_filterbank, axis=1)
    return reshaped_filterbank


def filterbank_bar_graph(
    width: int, height: int, max_y_axis: int, filterbank: np.ndarray
) -> np.ndarray:
    """ Turn each bin of a filterbank into a bar graph

    Args:
        width: width of LED Wall in pixels
        height: height of LED Wall in pixels
        max_y_axis: maximum of Y axis for graphing
        filterbank: filterbank array plot

    Returns:
        np.ndarray: uint8 ndarray with dimensions (width, height, 3)

    Raises:
        NotImplementedError: when the width and length of the filterbank don't match
    """
    if not width == len(filterbank):
        filterbank = _reshape_filterbank(filterbank, width)

    bar_graph = np.zeros((height, width, 3), dtype=np.uint8)

    # create a bar that goes from blue to yellow to red
    full_bar = np.zeros((height, 3), dtype=np.uint8)
    for i in range(height):
        if i > height * 9 / 10:
            full_bar[i, :] = [
                255,
                0,
                0,
            ]
        elif i > height * 1 / 4:
            coefficient = (i - (height * (1 / 4 * 9 / 10))) / (
                height - (height * (1 / 4 * 9 / 10))
            )
            full_bar[i, :] = [
                255,
                0 - (255 * coefficient),
                0,
            ]
        else:
            coefficient = i / (height * 1 / 4)
            full_bar[i, :] = [
                255 / 3 + (255 * coefficient / 3),
                255 / 3 + (255 * coefficient / 3),
                255 / 3 - (255 * coefficient / 3),
            ]

    for i, energy in enumerate(filterbank):
        if not energy:
            bar_height = 0
        else:
            bar_height = int((energy / max_y_axis) * height)

        # dimensions: height, width, RGB bytes
        # set all pixels from the bottom to bar_height full white
        bar_graph[0:bar_height, i, :] = full_bar[0:bar_height, :]

    return np.flipud(bar_graph)
