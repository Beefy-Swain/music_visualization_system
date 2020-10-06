""" __main__ module of music_visualization_system"""

import atexit
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

import music_visualization_system.filterbank_bar_graph as fbg
import pymvf


@atexit.register
def _killtree(including_parent=True):
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        child.kill()

    if including_parent:
        parent.kill()


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
        sys.exit("must provide a port")

    led_wall_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    led_wall_connection.connect((server_address, server_port))

    pymvf_queue: mp.Queue = mp.Queue()
    pymvf_process = mp.Process(target=pymvf.PyMVF, args=(pymvf_queue,))
    pymvf_process.start()
    next_update = time.monotonic() + 1 / 120

    while True:
        start = time.monotonic()

        buffer = pymvf_queue.get()

        max_energy = np.amax(
            np.concatenate(
                (buffer.left_channel_filterbank, buffer.right_channel_filterbank,)
            )
        ).astype(np.uint32)

        average_filterbank = (
            (buffer.left_channel_filterbank / 2) + (buffer.right_channel_filterbank / 2)
        ).astype(np.uint32)

        frame = fbg.one_channel(width, height, max_energy, average_filterbank[2:-5])
        pickled_frame = pickle.dumps(frame)

        # https://stackoverflow.com/a/60067126/1342874
        header = struct.pack("!Q", len(pickled_frame))
        led_wall_connection.sendall(header)
        led_wall_connection.sendall(pickled_frame)

        took = time.monotonic() - start
        if took > 512 / 44100:
            print("." * round(1000 * (took - 512 / 44100)))


if __name__ == "__main__":
    typer.run(main)
