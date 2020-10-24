import numpy as np  # type:ignore


def _reshape_bin_rms_array(bin_rms_array: np.ndarray, width: int) -> np.ndarray:
    common_multiple_bin_rms_array = np.repeat(bin_rms_array, repeats=width).reshape(
        (width, len(bin_rms_array))
    )
    reshaped_bin_rms_array = np.mean(common_multiple_bin_rms_array, axis=1)
    return reshaped_bin_rms_array


def one_channel(
    width: int, height: int, max_y_axis: float, bin_rms_array: np.ndarray
) -> np.ndarray:
    """ Turn each bin of a bin_rms_array into a bar graph

    Args:
        width: width of LED Wall in pixels
        height: height of LED Wall in pixels
        max_y_axis: maximum of Y axis for graphing
        bin_rms_array: array of bin energies

    Returns:
        np.ndarray: uint8 ndarray with dimensions (width, height, 3)

    Raises:
        NotImplementedError: when the width and length of the bin_rms_array don't match
    """
    if not width == len(bin_rms_array):
        bin_rms_array = _reshape_bin_rms_array(bin_rms_array, width)

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

    for i, energy in enumerate(bin_rms_array):
        if not energy:
            bar_height = 0
        else:
            bar_height = int((energy / max_y_axis) * height)

        bar_graph[0:bar_height, i, :] = full_bar[0:bar_height, :]

    return np.flipud(bar_graph)


def centered_two_channel(
    width: int,
    height: int,
    max_y_axis: float,
    left_channel: np.ndarray,
    right_channel: np.ndarray,
):
    # odd
    if width % 2:
        each_channel_width = int(width / 2) + 1
    # even
    else:
        each_channel_width = int(width / 2)

    left_graph = one_channel(int(each_channel_width), height, max_y_axis, left_channel)
    right_graph = one_channel(
        int(each_channel_width), height, max_y_axis, right_channel
    )

    left_graph = np.fliplr(left_graph)
    output = np.concatenate((left_graph, right_graph), axis=1)

    return output
