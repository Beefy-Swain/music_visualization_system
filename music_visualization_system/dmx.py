""" esp8266-dmx control object

See: https://github.com/caverage/esp8266-dmx
"""
import socket


class ESPDMX:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def __call__(self, channel_values: bytearray) -> None:
        controller_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        controller_connection.connect((self.host, self.port))

        controller_connection.sendall(channel_values)
