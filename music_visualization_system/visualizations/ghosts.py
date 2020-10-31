"""The ghosts at each side of the stage"""

import time

import music_visualization_system as mvs
import pymvf


class Ghosts:
    def __init__(self, dmx: mvs.dmx.ESPDMX):
        pass

    def _main(self) -> None:
        while True:
            time.sleep(1)

    def __call__(self, buffer: pymvf.buffer.Buffer) -> None:
        pass
