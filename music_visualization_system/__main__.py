""" __main__ module of music_visualization_system"""

import atexit
import logging
import os
import time

import numpy as np  # type:ignore
import psutil  # type:ignore
import typer

import music_visualization_system.led_wall_bar_graph as led
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

    bin_edges = pymvf.dsp.generate_bin_edges(40, 16000, 24)
    # FIXME: this should be what we pass to pymvf, not bin_edges
    bins = []
    previous_edge = bin_edges[0]
    for edge in bin_edges[1:]:
        bins.append((previous_edge, edge))
        previous_edge = edge

    buffer_discard_qty = 10
    buffer_processor = pymvf.PyMVF(
        bin_edges=bin_edges, buffer_discard_qty=buffer_discard_qty,
    )

    time_per_buffer = buffer_processor.buffer_size / buffer_processor.sample_rate

    led_wall = led.LEDWall(led_wall_server, width, height, delay, bins, time_per_buffer)

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

        # logarithmic scaling bin energies to compensate for the way humans hear
        for energy_array in [left_bin_energy_array, right_bin_energy_array]:
            for i, energy in enumerate(energy_array):
                energy_array[i] = energy * (1 + i) ** 0.5

        current_max_energy = float(
            np.amax(np.concatenate((left_bin_energy_array, right_bin_energy_array)))
        )

        if max_energy is None:
            # first instance of not silence
            max_energy = current_max_energy

        if max_energy < current_max_energy:
            # increase the max energy
            max_energy = current_max_energy

            buffers_since_last_max_energy = 0
            LOGGER.info(f"max energy increased to {max_energy}")
        elif buffers_since_last_max_energy > 100:
            max_energy = max_energy * 0.9
            buffers_since_last_max_energy = 0
            LOGGER.info(f"max energy decreased to {max_energy}")
        else:
            buffers_since_last_max_energy += 1

        led_wall((max_energy, buffer))


if __name__ == "__main__":
    typer.run(main)
