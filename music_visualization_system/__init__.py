from . import dmx, visualizations

BUFFER_SIZE = 512
SAMPLE_RATE = 44100
TIME_PER_BUFFER = BUFFER_SIZE / SAMPLE_RATE
BUFFERS_PER_SECOND = SAMPLE_RATE / BUFFER_SIZE
