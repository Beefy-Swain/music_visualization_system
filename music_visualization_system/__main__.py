""" __main__ module of music_visualization_system"""

import atexit
import logging
import os
import pickle
import socket
import struct
import sys
import time

import numpy as np  # type:ignore
import psutil  # type:ignore
import typer

import music_visualization_system.led_wall_bar_graph as bar_graph
import pymvf

logging.basicConfig(filename="mvs.log", level=20)
LOGGER = logging.getLogger(__name__)


@atexit.register
def _killtree(including_parent=True):
    LOGGER.critical("Stopping")
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()

    if including_parent:
        parent.kill()

    LOGGER.critical("Stopped")


def main(
    led_wall_server: str = typer.Argument(..., help="HOST:PORT of the LED Wall"),
    width: int = typer.Argument(..., help="width of LED Wall in pixels"),
    height: int = typer.Argument(..., help="height of LED Wall in pixels"),
    delay: float = typer.Option(0, help="height of LED Wall in pixels"),
) -> None:
    """ Main function of music_visualization_system"""

    server_address = led_wall_server.split(":")[0]
    try:
        server_port = int(led_wall_server.split(":")[1])
    except IndexError:
        sys.exit("must provide a port on the host")

    led_wall_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    led_wall_connection.connect((server_address, server_port))

    buffer_discard_qty = 10
    buffer_processor = pymvf.PyMVF(
        bin_edges=pymvf.dsp.generate_bin_edges(40, 16000, 24), buffer_discard_qty=10,
    )

    time_per_buffer = buffer_processor.buffer_size / buffer_processor.sample_rate

    # each bin energy meter is more logarithmic as this approaches zero
    logarithmic_constant = 0.8
    max_energy_gate_constant = 0.9
    max_energy = None
    buffers_since_last_max_energy = 0

    next_frame_time = None
    while True:
        buffer = buffer_processor()

        if next_frame_time is None:
            next_frame_time = buffer.timestamp + delay
        else:
            next_frame_time = next_frame_time + time_per_buffer

        left_bin_energy_array = np.array(list(buffer.left_bin_energy_mapping.values()))
        right_bin_energy_array = np.array(
            list(buffer.right_bin_energy_mapping.values())
        )

        for energy_array in [left_bin_energy_array, right_bin_energy_array]:
            for i, energy in enumerate(energy_array):
                energy_array[i] = energy * (1 + i) ** 0.5

        left_bin_log_energy_array = np.power(
            left_bin_energy_array, logarithmic_constant
        )
        right_bin_log_energy_array = np.power(
            right_bin_energy_array, logarithmic_constant
        )

        current_max_energy = float(
            np.amax(
                np.concatenate((left_bin_log_energy_array, right_bin_log_energy_array))
            )
        )
        if not current_max_energy:
            # silence, skip
            continue

        if max_energy is None:
            # first instance of not silence
            max_energy = current_max_energy * max_energy_gate_constant

        if max_energy < current_max_energy:
            # increase the max energy
            max_energy = current_max_energy * max_energy_gate_constant

            buffers_since_last_max_energy = 0
            LOGGER.debug(f"max energy increased to {max_energy}")
        elif buffers_since_last_max_energy > 100:
            max_energy = max_energy * 0.9
            buffers_since_last_max_energy = 0
            LOGGER.debug(f"max energy decreased to {max_energy}")
        else:
            buffers_since_last_max_energy += 1

        frame = bar_graph.centered_two_channel(
            width,
            height,
            max_energy,
            left_bin_log_energy_array,
            right_bin_log_energy_array,
        )
        pickled_frame = pickle.dumps(frame)

        while time.perf_counter() < next_frame_time:
            # waiting for it to be time to display the next frame
            time.sleep(0.001)

        # https://stackoverflow.com/a/60067126/1342874
        header = struct.pack("!Q", len(pickled_frame))
        led_wall_connection.sendall(header)
        led_wall_connection.sendall(pickled_frame)


if __name__ == "__main__":
    typer.run(main)
