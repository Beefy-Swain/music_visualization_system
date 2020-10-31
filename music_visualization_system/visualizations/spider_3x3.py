"""The 3x3 spider light on the cieling

https://www.amazon.com/gp/product/B07XRDB4QT/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1
"""

import logging
import multiprocessing as mp
import time

import numpy as np  # type: ignore

import music_visualization_system as mvs
import pymvf

LOGGER = logging.getLogger(__name__)


class Spider3X3:
    def __init__(self, delay: float):
        self._dmx_device = mvs.dmx.ESPDMX("espdmx1.caverage.lan", 80)
        self._delay = delay

        self.disable()

        self._process_queue = mp.Queue()
        self._drive_queue = mp.Queue()

        self._process_process = pymvf.Process(target=self._process)
        self._process_process.start()

        self._drive_process = pymvf.Process(target=self._drive)
        self._drive_process.start()

    def enable(self) -> None:
        dmx_channels = bytearray()

        for _ in range(1, 9 + 1):
            dmx_channels.append(0)

        dmx_channels.append(220)  # AUTO effect

        for _ in range(11, 12 + 1):
            dmx_channels.append(0)

        try:
            self._dmx_device(dmx_channels)
            LOGGER.info("enabled")
        except OSError:
            LOGGER.critical("cannot reach Spider3X3 DMX controller")

    def disable(self) -> None:
        dmx_channels = bytearray()

        for _ in range(1, 12):
            dmx_channels.append(0)

        try:
            self._dmx_device(dmx_channels)
            LOGGER.info("disabled")
        except OSError:
            LOGGER.critical("cannot reach Spider3X3 DMX controller")

    def _process(self) -> None:
        """ Process incoming data into directions for the `_drive` process"""

        buffers_of_delay = mvs.BUFFERS_PER_SECOND * self._delay
        look_ahead_list = []
        disable_time = None

        while True:
            look_ahead_list.append(self._process_queue.get())

            # wait until we have enough values for the calculations
            # nothing is discarded here, so no issues with timing or sync
            if not len(look_ahead_list) > buffers_of_delay / 2:
                continue

            timestamp, _ = look_ahead_list.pop(0)

            # every intensity for DELAY seconds worth of buffers, approx
            delayed_intensities = np.array(look_ahead_list)[:, 1]

            if disable_time is None:
                disable_time = timestamp

            if np.average(delayed_intensities) > 0.3:
                period = len(delayed_intensities) * mvs.TIME_PER_BUFFER
                if timestamp + period > disable_time:
                    # LOGGER.info(">0.3")
                    disable_time = timestamp + period
            elif (
                np.average(delayed_intensities[: int(len(delayed_intensities) / 2)])
                > 0.4
            ):
                period = (len(delayed_intensities) / 2) * mvs.TIME_PER_BUFFER
                if timestamp + period > disable_time:
                    LOGGER.info(">0.4")
                    disable_time = timestamp + period
            elif (
                np.average(delayed_intensities[: int(len(delayed_intensities) / 4)])
                > 0.6
            ):
                period = (len(delayed_intensities) / 4) * mvs.TIME_PER_BUFFER
                if timestamp + period > disable_time:
                    LOGGER.info(">0.6")
                    disable_time = timestamp + period

            enabled = bool(timestamp < disable_time)

            # LOGGER.info(disable_time - timestamp)
            # LOGGER.info(enabled)

            self._drive_queue.put((timestamp, enabled))

    def _drive(self) -> None:
        """ Drive the DMX device"""
        next_frame_time = None
        current_status = False

        while True:
            timestamp, enabled = self._drive_queue.get()

            if next_frame_time is None:
                next_frame_time = timestamp + self._delay

            # if current_status == enabled:
            #     # leave it be if it's the same
            #     next_frame_time += mvs.TIME_PER_BUFFER
            #     continue

            while time.perf_counter() < next_frame_time:
                # wait for next frame time
                pass

            if enabled:
                self.enable()
            else:
                self.disable()

            current_status = enabled
            next_frame_time += mvs.TIME_PER_BUFFER

    def __call__(self, timestamp: float, mono_intensity: float) -> None:
        """ Add an intensity to the queue for processing.

        Args:
            timestamp: the timestamp of the buffer that the intensity came from.
            mono_intensity: the intensity of the mono channel for the given timestamp.
        """

        assert mono_intensity <= 1
        self._process_queue.put((timestamp, mono_intensity))
