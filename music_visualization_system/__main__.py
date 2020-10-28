""" __main__ module of music_visualization_system"""

import atexit
import logging
import multiprocessing as mp
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
        bin_edges=pymvf.dsp.generate_bin_edges(20, 16000, 24), buffer_discard_qty=10,
    )

    logarithmic_constant = 0.8
    # this prevents the max energy from "running away"
    # why this happens is not understood. I blame transients
    max_energy_gate_constant = 0.8
    max_energy = None
    buffers_since_last_max_energy = 0
    while True:
        start = time.monotonic()
        buffer = buffer_processor()

        left_bin_rms_array = np.array(list(buffer.left_bin_energy_mapping.values()))
        right_bin_rms_array = np.array(list(buffer.right_bin_energy_mapping.values()))

        current_max_energy = float(
            np.amax(np.concatenate((left_bin_rms_array, right_bin_rms_array)))
        )
        if not current_max_energy:
            # silence, skip
            continue

        if not max_energy:
            # first instance of not silence
            max_energy = (
                np.power(current_max_energy, logarithmic_constant)
                * max_energy_gate_constant
            )
        elif max_energy < current_max_energy:
            # increase the max energy
            max_energy = (
                np.power(current_max_energy, logarithmic_constant)
                * max_energy_gate_constant
            )
            buffers_since_last_max_energy = 0
            print(f"max energy increased to {max_energy}")
        elif buffers_since_last_max_energy > 100:
            max_energy = max_energy * 0.9
            buffers_since_last_max_energy = 0
            print(f"max energy decreased to {max_energy}")
        else:
            buffers_since_last_max_energy += 1

        left_bin_log_rms_array = np.power(left_bin_rms_array, logarithmic_constant)
        right_bin_log_rms_array = np.power(right_bin_rms_array, logarithmic_constant)

        frame = bar_graph.centered_two_channel(
            width, height, max_energy, left_bin_log_rms_array, right_bin_log_rms_array,
        )
        pickled_frame = pickle.dumps(frame)

        # https://stackoverflow.com/a/60067126/1342874
        header = struct.pack("!Q", len(pickled_frame))
        led_wall_connection.sendall(header)
        led_wall_connection.sendall(pickled_frame)

        # took = time.monotonic() - start
        # if took > 512 / 44100:
        #     print("." * round(1000 * (took - 512 / 44100)))


if __name__ == "__main__":
    typer.run(main)
