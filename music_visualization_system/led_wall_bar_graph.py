import logging
import multiprocessing as mp
import pickle
import queue
import socket
import struct
import sys
import time
from typing import List, Tuple

import numpy as np  # type:ignore

import pymvf

LOGGER = logging.getLogger(__name__)


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
) -> np.ndarray:
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
    frame = np.concatenate((right_graph, left_graph), axis=1)

    return frame


class LEDWall:
    """ Object for controlling and managing the LED Wall"""

    def __init__(
        self,
        led_wall_server: str,
        width: int,
        height: int,
        delay: float,
        bins: List[Tuple[int, int]],
        time_per_buffer: float,
    ):
        self._width = width
        self._height = height
        self._delay = delay
        self._bins = bins
        self._time_per_buffer = time_per_buffer

        server_address = led_wall_server.split(":")[0]
        try:
            server_port = int(led_wall_server.split(":")[1])
        except IndexError:
            sys.exit("must provide a port on the host")

        self.led_wall_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.led_wall_connection.connect((server_address, server_port))

        self._input_queue = mp.Queue()
        self._main_process = pymvf.Process(target=self._main)
        self._main_process.start()

    def _send_frame(self, frame: np.ndarray) -> None:
        pickled_frame = pickle.dumps(frame)

        # https://stackoverflow.com/a/60067126/1342874
        header = struct.pack("!Q", len(pickled_frame))
        self.led_wall_connection.sendall(header)
        self.led_wall_connection.sendall(pickled_frame)

    def _main(self) -> None:
        # each bin energy meter is more logarithmic as this approaches zero
        logarithmic_constant = 0.8
        max_energy_gate_constant = 0.9

        next_frame_time = None

        max_bin_energies = {bin_: None for bin_ in self._bins}
        frames = []
        LOGGER.info("Initialized LEDWall Process")
        while True:
            try:
                max_energy, buffer = self._input_queue.get(block=False)
                max_energy = max_energy * max_energy_gate_constant

                if next_frame_time is None:
                    next_frame_time = buffer.timestamp + self._delay

                left_bin_energy_array = np.array(
                    list(buffer.left_bin_energy_mapping.values())
                )
                right_bin_energy_array = np.array(
                    list(buffer.right_bin_energy_mapping.values())
                )

                # logarithmic scaling bin energies to compensate for the way humans hear
                for energy_array in [left_bin_energy_array, right_bin_energy_array]:
                    for i, energy in enumerate(energy_array):
                        energy_array[i] = energy * (1 + i) ** 0.5

                left_bin_log_energy_array = np.power(
                    left_bin_energy_array, logarithmic_constant
                )
                right_bin_log_energy_array = np.power(
                    right_bin_energy_array, logarithmic_constant
                )
                frame = centered_two_channel(
                    self._width,
                    self._height,
                    max_energy,
                    left_bin_log_energy_array,
                    right_bin_log_energy_array,
                )
                frames.append(frame)

            except queue.Empty:
                if next_frame_time is None:
                    # no frames recieved yet
                    continue

            # when we get close, wait to send the frame
            if time.perf_counter() > next_frame_time - (self._time_per_buffer / 5):
                while time.perf_counter() < next_frame_time:
                    # block until it's time to send the frame
                    continue
                self._send_frame(frames.pop(0))
                next_frame_time += self._time_per_buffer

    def __call__(self, buffer: pymvf.buffer.Buffer) -> None:
        self._input_queue.put(buffer)
