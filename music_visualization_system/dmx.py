""" esp8266-dmx control object

See: https://github.com/caverage/esp8266-dmx
"""
import socket
from typing import Optional

import numpy as np  # type:ignore


class ESPDMX:
    def __init__(self, host: str, port: int):
        self._connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connection.connect((host, port))

    def __call__(self, channel_values: np.array) -> None:
        assert channel_values.dtype == np.uint8

        self._connection.sendall(channel_values.asbytes())
